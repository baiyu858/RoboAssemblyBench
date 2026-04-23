from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.util import has_display
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions

from toolkits.factory_dual_franka_assembly.planner_primitives import quat_rotate
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_batch
from toolkits.factory_dual_franka_assembly.scene_profiles import DEFAULT_SCENE_PROFILE, list_scene_profiles
from toolkits.factory_dual_franka_assembly.task_specs import list_task_recipes


def _infer_preview_target_spec(task, *, robot_name: str) -> tuple[str | None, dict | str | None]:
    for phase_spec in getattr(task.config, 'phase_specs', []):
        robot_targets = phase_spec.get('robot_targets') or {}
        target_like = robot_targets.get(robot_name)
        if target_like is None:
            continue
        if isinstance(target_like, dict):
            target_name = target_like.get('target') or target_like.get('target_name') or target_like.get('name')
            return target_name, target_like
        if isinstance(target_like, str):
            return target_like, target_like
    return None, None


def _robot_hold_action(obs: dict | None, *, gripper_command: str = 'open') -> dict:
    robot_obs = obs or {}
    current_position = robot_obs.get('eef_position')
    current_orientation = robot_obs.get('eef_orientation')
    return {
        'arm_ik_controller': [
            None if current_position is None else np.asarray(current_position, dtype=float).tolist(),
            None if current_orientation is None else np.asarray(current_orientation, dtype=float).tolist(),
        ],
        'gripper_controller': [str(gripper_command)],
    }


def _move_left_robot_to_preview_target(
    env: Env,
    task,
    initial_obs: dict,
    *,
    target_name: str | None,
    target_like,
    max_steps: int,
    position_tolerance: float,
) -> tuple[dict, dict]:
    if target_like is None:
        return initial_obs, {
            'enabled': False,
            'reason': 'no_target',
            'target_name': None,
        }

    current_left_obs = (initial_obs or {}).get('franka_left', {})
    fallback_orientation = current_left_obs.get('eef_orientation')
    _, target_position, target_orientation, _ = task.resolve_robot_target_pose('franka_left', target_like)
    if target_position is None:
        return initial_obs, {
            'enabled': False,
            'reason': 'unresolved_target',
            'target_name': target_name,
        }
    if target_orientation is None:
        if fallback_orientation is not None:
            target_orientation = np.asarray(fallback_orientation, dtype=float)
        else:
            target_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    obs = initial_obs
    reached = False
    steps_taken = 0
    for step_idx in range(max(int(max_steps), 1)):
        left_obs = (obs or {}).get('franka_left', {})
        right_obs = (obs or {}).get('franka_right', {})
        left_action = {
            'arm_ik_controller': [
                np.asarray(target_position, dtype=float).tolist(),
                np.asarray(target_orientation, dtype=float).tolist(),
            ],
            'gripper_controller': ['open'],
        }
        env_actions = {
            'franka_left': left_action,
            'franka_right': _robot_hold_action(right_obs, gripper_command='open'),
        }
        obs_list, _, terminated, _, _ = env.step([env_actions])
        obs = obs_list[0] if obs_list else obs
        steps_taken = step_idx + 1

        left_obs = (obs or {}).get('franka_left', {})
        current_position = left_obs.get('eef_position')
        if current_position is not None:
            current_position = np.asarray(current_position, dtype=float)
            if np.linalg.norm(current_position - np.asarray(target_position, dtype=float)) <= float(position_tolerance):
                reached = True
                break

        if terminated and terminated[0]:
            break

    return obs, {
        'enabled': True,
        'reason': 'ok' if reached else 'stopped_before_tolerance',
        'target_name': target_name,
        'target_position': np.asarray(target_position, dtype=float).tolist(),
        'target_orientation': np.asarray(target_orientation, dtype=float).tolist(),
        'steps_taken': int(steps_taken),
        'reached': bool(reached),
        'position_tolerance': float(position_tolerance),
    }


def _build_env(task_configs, *, headless: bool):
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            native=False,
            webrtc=False,
        ),
        env_num=1,
        metrics_save_path='none',
        task_configs=task_configs,
    )
    import_extensions()
    return Env(config)


def _gf_quat_to_wxyz(quat) -> list[float]:
    imag = quat.GetImaginary()
    return [
        float(quat.GetReal()),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
    ]


def _matrix_pose(matrix) -> tuple[list[float], list[float], list[float]]:
    from omni.isaac.core.utils.rotations import quat_to_euler_angles

    translation = matrix.ExtractTranslation()
    rotation_quat = matrix.ExtractRotation().GetQuat()
    quat_wxyz = _gf_quat_to_wxyz(rotation_quat)
    euler_xyz = quat_to_euler_angles(np.asarray(quat_wxyz, dtype=float), extrinsic=False).tolist()
    return (
        [float(translation[0]), float(translation[1]), float(translation[2])],
        quat_wxyz,
        [float(value) for value in euler_xyz],
    )


def _round_list(values, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _camera_prim_path(task, camera_spec: dict) -> str:
    prim_path = str(camera_spec.get('prim_path') or '')
    if prim_path.startswith('/'):
        return prim_path
    owner_name = str(camera_spec.get('owner') or camera_spec.get('robot') or '')
    if owner_name not in task.robots:
        raise KeyError(f'Camera {camera_spec.get("name")!r} references unknown owner {owner_name!r}.')
    robot_root = str(task.robots[owner_name].config.prim_path).rstrip('/')
    return f'{robot_root}/{prim_path}'


def _camera_snapshot(task) -> list[dict]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera_rows: list[dict] = []
    for camera_spec in getattr(task.config, 'camera_metadata', []):
        prim_path = _camera_prim_path(task, camera_spec)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            camera_rows.append(
                {
                    'name': camera_spec.get('name'),
                    'view_type': camera_spec.get('view_type'),
                    'prim_path': prim_path,
                    'valid': False,
                }
            )
            continue

        xformable = UsdGeom.Xformable(prim)
        local_matrix, _ = xformable.GetLocalTransformation()
        world_matrix = omni.usd.get_world_transform_matrix(prim)

        local_position, local_quat, local_euler = _matrix_pose(local_matrix)
        world_position, world_quat, world_euler = _matrix_pose(world_matrix)

        view_type = str(camera_spec.get('view_type') or '')
        snippet: dict[str, list[float]] = {}
        if view_type == 'front':
            snippet = {
                'translation': _round_list(world_position),
                'orientation_euler': _round_list(world_euler),
            }
        elif view_type == 'wrist':
            local_forward = quat_rotate(local_quat, [0.0, 0.0, -1.0])
            look_offset = (0.18 * np.asarray(local_forward, dtype=float)).tolist()
            snippet = {
                'translation': _round_list(local_position),
                'orientation_euler': _round_list(local_euler),
                'mount_offset': _round_list(local_position),
                'look_offset': _round_list(look_offset),
            }

        camera_rows.append(
            {
                'name': camera_spec.get('name'),
                'view_type': view_type,
                'owner': camera_spec.get('owner'),
                'prim_path': prim_path,
                'valid': True,
                'world_position': _round_list(world_position),
                'world_orientation': _round_list(world_quat),
                'world_orientation_euler': _round_list(world_euler),
                'local_position': _round_list(local_position),
                'local_orientation': _round_list(local_quat),
                'local_orientation_euler': _round_list(local_euler),
                'yaml_snippet': snippet,
            }
        )
    return camera_rows


def _print_snapshot(task, *, dump_json_path: Path | None = None):
    snapshot = {
        'recipe': getattr(task.config, 'recipe', ''),
        'scene_profile': getattr(task.config, 'scene_profile', ''),
        'workspace_offset': getattr(task.config, 'workspace_offset', []),
        'cameras': _camera_snapshot(task),
    }

    print('\n================ Camera Snapshot ================')
    print(f"recipe        : {snapshot['recipe']}")
    print(f"scene_profile : {snapshot['scene_profile']}")
    print(f"workspace     : {snapshot['workspace_offset']}")
    for camera in snapshot['cameras']:
        print('------------------------------------------------')
        print(f"name          : {camera['name']}")
        print(f"view_type     : {camera['view_type']}")
        print(f"prim_path     : {camera['prim_path']}")
        if not camera['valid']:
            print('status        : invalid prim')
            continue
        print(f"world_pos     : {camera['world_position']}")
        print(f"world_euler   : {camera['world_orientation_euler']}")
        print(f"local_pos     : {camera['local_position']}")
        print(f"local_euler   : {camera['local_orientation_euler']}")
        if camera['yaml_snippet']:
            print('yaml_snippet  :')
            for key, value in camera['yaml_snippet'].items():
                print(f'  {key}: {value}')
    print('=================================================\n')

    if dump_json_path is not None:
        dump_json_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Open a live UI preview for tuning dual-arm assembly cameras.')
    parser.add_argument('--recipe', default='screw_fastening', choices=list_task_recipes())
    parser.add_argument('--scene-profile', default=DEFAULT_SCENE_PROFILE, choices=list_scene_profiles())
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--print-interval',
        type=float,
        default=0.0,
        help='Seconds between camera transform dumps. Use 0 to disable auto-dumping.',
    )
    parser.add_argument('--dump-json-path', type=str, default='')
    parser.add_argument('--warmup-render-steps', type=int, default=5)
    parser.add_argument(
        '--preview-left-target',
        type=str,
        default='auto',
        help='Move the left arm to this named target before preview. Use auto to choose the first left-arm task target, or none to disable.',
    )
    parser.add_argument(
        '--preview-left-max-steps',
        type=int,
        default=120,
        help='Maximum env steps used to move the left arm to the preview target before entering the idle UI loop.',
    )
    parser.add_argument(
        '--preview-left-position-tolerance',
        type=float,
        default=0.03,
        help='Position tolerance for considering the left arm preview pose reached.',
    )
    args = parser.parse_args()

    if not args.headless and not has_display():
        raise RuntimeError('No display detected. Use a desktop session or pass --headless.')

    env = _build_env(
        task_configs=build_dual_franka_assembly_batch(
            recipe=args.recipe,
            seeds=[int(args.seed)],
            scene_profile=args.scene_profile,
            attach_runtime_cameras=True,
        ),
        headless=bool(args.headless),
    )

    dump_json_path = Path(args.dump_json_path).resolve() if args.dump_json_path else None

    try:
        obs_list, task_cfgs = env.reset()
        if not task_cfgs or task_cfgs[0] is None or not env.runner.current_tasks:
            raise RuntimeError('No task was loaded for camera preview.')

        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]

        env.runner.warm_up(steps=max(int(args.warmup_render_steps), 1), render=True, physics=False)

        preview_left_target_arg = str(args.preview_left_target).strip()
        preview_target_name = None
        preview_target_like = None
        if preview_left_target_arg.lower() != 'none':
            if preview_left_target_arg.lower() == 'auto':
                preview_target_name, preview_target_like = _infer_preview_target_spec(task, robot_name='franka_left')
            else:
                preview_target_name = preview_left_target_arg
                preview_target_like = preview_left_target_arg

        initial_obs = obs_list[0] if obs_list else {}
        preview_pose_status = {
            'enabled': False,
            'reason': 'disabled',
            'target_name': None,
        }
        if preview_target_like is not None:
            initial_obs, preview_pose_status = _move_left_robot_to_preview_target(
                env,
                task,
                initial_obs,
                target_name=preview_target_name,
                target_like=preview_target_like,
                max_steps=int(args.preview_left_max_steps),
                position_tolerance=float(args.preview_left_position_tolerance),
            )

        print('Camera preview is live.')
        print('Open the Stage and Property panels, select a camera prim, and drag/rotate it in the UI.')
        if float(args.print_interval) > 0:
            print('Every few seconds the script will print the current transform and a YAML snippet you can copy back.')
        else:
            print('Auto snapshot dumping is disabled by default because the transform query path can crash Isaac on this setup.')
            print('Use the Stage + Property panels to adjust cameras live, then copy local Translate / Rotate XYZ back into YAML.')
            print('For wrist cameras, these map directly to translation / orientation_euler.')
            print('For the front camera, use the world-mounted prim and copy Translate / Rotate XYZ into translation / orientation_euler.')
        if preview_pose_status.get('enabled'):
            print(
                f"Left arm preview pose: {preview_pose_status.get('target_name')} "
                f"(reached={preview_pose_status.get('reached')}, steps={preview_pose_status.get('steps_taken')})"
            )
        else:
            print(f"Left arm preview pose: disabled ({preview_pose_status.get('reason')})")
        print('Front camera prim   : /World/env_0/cameras/third_person_front')
        print('Left wrist prim     : /World/env_0/robots/franka_left/panda_hand/left_wrist_camera')
        print('Right wrist prim    : /World/env_0/robots/franka_right/panda_hand/right_wrist_camera')
        print('Press Ctrl+C to exit.\n')

        auto_dump_enabled = float(args.print_interval) > 0
        last_print = time.time()
        while True:
            env.simulation_app.update()
            now = time.time()
            if auto_dump_enabled and now - last_print >= max(float(args.print_interval), 0.2):
                _print_snapshot(task, dump_json_path=dump_json_path)
                last_print = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        print('\nCamera preview interrupted by user.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
