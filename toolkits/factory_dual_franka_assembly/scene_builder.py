from __future__ import annotations

import copy
import random
from typing import Iterable

import numpy as np

from internutopia_extension.configs.objects import (
    DynamicCompoundCuboidCfg,
    DynamicCubeCfg,
    StaticCubeCfg,
    UsdObjCfg,
    VisualCubeCfg,
)
from internutopia_extension.configs.robots.franka import (
    FrankaRobotCfg,
    arm_ik_cfg as franka_arm_ik_cfg,
    arm_joint_cfg as franka_arm_joint_cfg,
    gripper_cfg as franka_gripper_cfg,
)
from internutopia_extension.configs.robots.ur5e import (
    UR5eRobotCfg,
    arm_ik_cfg as ur5e_arm_ik_cfg,
    arm_joint_cfg as ur5e_arm_joint_cfg,
    gripper_cfg as ur5e_gripper_cfg,
)
from internutopia_extension.configs.sensors import RepCameraCfg
from internutopia_extension.configs.tasks.factory_dual_franka_assembly_task import (
    FactoryDualFrankaAssemblyTaskCfg,
)

from toolkits.factory_dual_franka_assembly.planner_primitives import (
    compose_pose,
    euler_xyz_intrinsic_to_quat,
    euler_xyz_to_quat,
    pose_dict,
    sample_position,
)
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe


def _resolve_orientation(spec: dict) -> np.ndarray:
    if 'orientation' in spec:
        return np.asarray(spec['orientation'], dtype=float)
    return euler_xyz_to_quat(spec.get('orientation_euler', [0.0, 0.0, 0.0]))


def _resolve_camera_orientation(spec: dict) -> np.ndarray:
    if 'orientation' in spec:
        return np.asarray(spec['orientation'], dtype=float)
    return euler_xyz_intrinsic_to_quat(spec.get('orientation_euler', [0.0, 0.0, 0.0]))


def _build_robot_cfgs(recipe_spec: dict) -> tuple[list, tuple[str, ...]]:
    robots = []
    robot_names = []
    for robot_spec in recipe_spec['robots']:
        orientation = _resolve_orientation(robot_spec)
        robot_type = str(robot_spec.get('type', 'FrankaRobot'))
        common_kwargs = dict(
            name=robot_spec['name'],
            prim_path=robot_spec['prim_path'],
            position=tuple(float(value) for value in robot_spec['position']),
            orientation=tuple(float(value) for value in orientation),
            sensors=[],
        )
        if robot_spec.get('usd_path') is not None:
            common_kwargs['usd_path'] = robot_spec['usd_path']
        if robot_spec.get('scale') is not None:
            common_kwargs['scale'] = tuple(float(value) for value in robot_spec['scale'])

        if robot_type in {'FrankaRobot', 'franka', 'Franka'}:
            robots.append(
                FrankaRobotCfg(
                    controllers=[
                        franka_arm_ik_cfg.update(),
                        franka_arm_joint_cfg.update(),
                        franka_gripper_cfg.update(),
                    ],
                    **common_kwargs,
                )
            )
        elif robot_type in {'UR5eRobot', 'ur5e', 'UR5e'}:
            ur5e_kwargs = {}
            for field_name in (
                'end_effector_prim_name',
                'ik_base_prim_name',
                'gripper_dof_name',
                'gripper_open_position',
                'gripper_closed_position',
                'hand_link_name',
                'left_finger_link_name',
                'right_finger_link_name',
                'initial_joint_positions',
                'gripper_xform_orient',
                'gripper_mount_local_pos0',
                'gripper_mount_local_pos1',
                'gripper_mount_local_rot0',
                'gripper_mount_local_rot1',
                'configure_gripper_mount_joint',
                'gripper_base_link_path',
                'gripper_container_path',
                'gripper_container_orient',
                'author_gripper_collision_pads',
            ):
                if field_name in robot_spec:
                    ur5e_kwargs[field_name] = copy.deepcopy(robot_spec[field_name])
            robots.append(
                UR5eRobotCfg(
                    type='UR5eRobot',
                    controllers=[
                        ur5e_arm_ik_cfg.update(),
                        ur5e_arm_joint_cfg.update(),
                        ur5e_gripper_cfg.update(),
                    ],
                    **common_kwargs,
                    **ur5e_kwargs,
                )
            )
        else:
            raise ValueError(f'Unsupported robot type {robot_type!r} for robot {robot_spec["name"]!r}.')
        robot_names.append(robot_spec['name'])
    return robots, tuple(robot_names)


def _normalize_camera_spec(camera_spec: dict, *, force_runtime_sensor: bool = False) -> dict:
    normalized = copy.deepcopy(camera_spec)
    normalized.setdefault('owner', normalized.get('robot', 'franka_left'))
    normalized.setdefault('video_key', f"observation.images.{normalized['name']}")
    normalized.setdefault('attach_runtime_sensor', False)
    if force_runtime_sensor:
        normalized['attach_runtime_sensor'] = True
        normalized['rgba'] = True
    return normalized


def _build_camera_cfg(camera_spec: dict) -> RepCameraCfg:
    normalized_spec = _normalize_camera_spec(camera_spec)
    resolution = normalized_spec.get('resolution', [640, 360])
    orientation = None
    if 'orientation' in normalized_spec or 'orientation_euler' in normalized_spec:
        orientation = tuple(float(value) for value in _resolve_camera_orientation(normalized_spec))

    return RepCameraCfg(
        name=normalized_spec['name'],
        prim_path=normalized_spec['prim_path'],
        resolution=tuple(int(value) for value in resolution),
        rgba=bool(normalized_spec.get('rgba', True)),
        landmarks=bool(normalized_spec.get('landmarks', False)),
        depth=bool(normalized_spec.get('depth', False)),
        pointcloud=bool(normalized_spec.get('pointcloud', False)),
        camera_params=bool(normalized_spec.get('camera_params', False)),
        position=None if normalized_spec.get('position') is None else tuple(float(value) for value in normalized_spec['position']),
        translation=None if normalized_spec.get('translation') is None else tuple(float(value) for value in normalized_spec['translation']),
        orientation=orientation,
        look_at=None if normalized_spec.get('look_at') is None else tuple(float(value) for value in normalized_spec['look_at']),
        focal_length=None
        if normalized_spec.get('focal_length') is None
        else float(normalized_spec['focal_length']),
        horizontal_aperture=None
        if normalized_spec.get('horizontal_aperture') is None
        else float(normalized_spec['horizontal_aperture']),
        vertical_aperture=None
        if normalized_spec.get('vertical_aperture') is None
        else float(normalized_spec['vertical_aperture']),
        clipping_range=None
        if normalized_spec.get('clipping_range') is None
        else tuple(float(value) for value in normalized_spec['clipping_range']),
    )


def _build_object_cfg(object_spec: dict, position: np.ndarray, orientation: np.ndarray):
    kind = object_spec['kind']
    common_kwargs = dict(
        name=object_spec['name'],
        prim_path=object_spec['prim_path'],
        position=tuple(float(value) for value in position),
        orientation=tuple(float(value) for value in orientation),
        scale=tuple(float(value) for value in object_spec.get('scale', [1.0, 1.0, 1.0])),
    )
    if kind == 'dynamic_cube':
        return DynamicCubeCfg(
            color=tuple(object_spec['color']),
            mass=object_spec.get('mass'),
            density=object_spec.get('density'),
            collider=object_spec.get('collider', True),
            static_friction=object_spec.get('static_friction'),
            dynamic_friction=object_spec.get('dynamic_friction'),
            restitution=object_spec.get('restitution'),
            **common_kwargs,
        )
    if kind == 'dynamic_compound_cuboid':
        return DynamicCompoundCuboidCfg(
            color=tuple(object_spec.get('color', [0.5, 0.5, 0.5])),
            parts=copy.deepcopy(object_spec['parts']),
            mass=object_spec.get('mass'),
            density=object_spec.get('density'),
            collider=object_spec.get('collider', True),
            static_friction=object_spec.get('static_friction'),
            dynamic_friction=object_spec.get('dynamic_friction'),
            restitution=object_spec.get('restitution'),
            **common_kwargs,
        )
    if kind == 'visual_cube':
        return VisualCubeCfg(color=list(object_spec['color']), **common_kwargs)
    if kind == 'static_cube':
        return StaticCubeCfg(
            color=list(object_spec['color']),
            static_friction=object_spec.get('static_friction'),
            dynamic_friction=object_spec.get('dynamic_friction'),
            restitution=object_spec.get('restitution'),
            **common_kwargs,
        )
    if kind == 'usd':
        return UsdObjCfg(
            usd_path=object_spec['usd_path'],
            collider=bool(object_spec.get('collider', True)),
            auto_collider=bool(object_spec.get('auto_collider', True)),
            rigid_body=bool(object_spec.get('rigid_body', True)),
            mass=object_spec.get('mass'),
            density=object_spec.get('density'),
            static_friction=object_spec.get('static_friction'),
            dynamic_friction=object_spec.get('dynamic_friction'),
            restitution=object_spec.get('restitution'),
            **common_kwargs,
        )
    raise ValueError(f'Unsupported object kind: {kind}')


def _sample_objects(recipe_spec: dict, rng: random.Random, workspace_offset: np.ndarray):
    object_cfgs = []
    object_states = {}
    tracked_object_names = []
    object_metadata = []
    for object_spec in recipe_spec['objects']:
        position = sample_position(object_spec['position'], object_spec.get('random_xy'), rng)
        if object_spec.get('apply_workspace_offset', True):
            position = position + workspace_offset
        orientation = _resolve_orientation(object_spec)
        object_cfgs.append(_build_object_cfg(object_spec, position=position, orientation=orientation))
        object_states[object_spec['name']] = {
            'position': position.tolist(),
            'orientation': orientation.tolist(),
            'scale': list(object_spec.get('scale', [1.0, 1.0, 1.0])),
            'kind': object_spec['kind'],
        }
        if object_spec.get('tracked', object_spec['kind'] not in {'visual_cube', 'static_cube'}):
            tracked_object_names.append(object_spec['name'])
        object_metadata.append(
            {
                **copy.deepcopy(object_spec),
                'sampled_position': position.tolist(),
                'sampled_orientation': orientation.tolist(),
            }
        )
    return object_cfgs, object_states, tuple(tracked_object_names), object_metadata


def _resolve_target_pose(target_spec: dict, object_states: dict, workspace_offset: np.ndarray) -> dict:
    if target_spec.get('reference', 'world') == 'world':
        base_position = np.asarray(target_spec.get('position', [0.0, 0.0, 0.0]), dtype=float)
        if target_spec.get('apply_workspace_offset', True):
            base_position = base_position + workspace_offset
        base_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        reference_state = object_states[target_spec['reference']]
        base_position = np.asarray(reference_state['position'], dtype=float)
        base_orientation = np.asarray(reference_state['orientation'], dtype=float)

    local_position = np.asarray(target_spec.get('offset', [0.0, 0.0, 0.0]), dtype=float)
    local_orientation = _resolve_orientation(target_spec)
    world_position, world_orientation = compose_pose(
        base_position=base_position,
        base_orientation=base_orientation,
        local_position=local_position,
        local_orientation=local_orientation,
    )
    return pose_dict(world_position, world_orientation)


def _build_target_poses(recipe_spec: dict, object_states: dict, workspace_offset: np.ndarray) -> dict:
    return {
        target_spec['name']: _resolve_target_pose(
            target_spec=target_spec,
            object_states=object_states,
            workspace_offset=workspace_offset,
        )
        for target_spec in recipe_spec['targets']
    }


def _build_annotation_target_metadata(recipe_spec: dict, target_poses: dict) -> dict:
    target_role_map = recipe_spec.get('annotation_target_roles', {})
    target_annotations = {}
    for target_name, pose in target_poses.items():
        target_annotation = copy.deepcopy(target_role_map.get(target_name, {}))
        target_annotation.setdefault('name', target_name)
        target_annotation['pose'] = copy.deepcopy(pose)
        target_annotations[target_name] = target_annotation
    return target_annotations


def _build_annotation_phase_metadata(recipe_spec: dict) -> list[dict]:
    note_map = {entry.get('name'): entry for entry in recipe_spec.get('annotation_phase_notes', []) if entry.get('name')}
    return [copy.deepcopy(note_map.get(phase_spec['name'], {'name': phase_spec['name']})) for phase_spec in recipe_spec.get('phases', [])]


def _normalize_phase_specs(phase_specs: Iterable[dict]) -> list[dict]:
    return [copy.deepcopy(phase_spec) for phase_spec in phase_specs]


def _normalize_success_criteria(success_criteria: Iterable[dict]) -> list[dict]:
    return [copy.deepcopy(criteria) for criteria in success_criteria]


def build_dual_franka_assembly_episode(
    recipe: str,
    seed: int,
    episode_idx: int = 0,
    spec_path: str | None = None,
    scene_profile: str | None = None,
    attach_runtime_cameras: bool = False,
) -> FactoryDualFrankaAssemblyTaskCfg:
    recipe_spec = load_task_recipe(spec_path or recipe, scene_profile=scene_profile)
    rng = random.Random(seed)
    workspace_offset = np.asarray(recipe_spec.get('workspace_offset', [0.0, 0.0, 0.0]), dtype=float)
    robots, robot_names = _build_robot_cfgs(recipe_spec)
    robot_cfg_map = {robot_cfg.name: robot_cfg for robot_cfg in robots}

    camera_metadata = []
    for camera_spec in recipe_spec.get('camera_specs', recipe_spec.get('cameras', [])):
        normalized_camera_spec = _normalize_camera_spec(
            camera_spec,
            force_runtime_sensor=attach_runtime_cameras,
        )
        owner_name = normalized_camera_spec['owner']
        if owner_name not in robot_cfg_map:
            raise KeyError(
                f"Camera {normalized_camera_spec['name']!r} references unknown robot owner {owner_name!r}."
            )
        if normalized_camera_spec.get('attach_runtime_sensor', False):
            robot_cfg_map[owner_name].sensors.append(_build_camera_cfg(normalized_camera_spec))
        camera_metadata.append(normalized_camera_spec)

    objects, object_states, tracked_object_names, object_metadata = _sample_objects(
        recipe_spec,
        rng,
        workspace_offset=workspace_offset,
    )
    target_poses = _build_target_poses(recipe_spec, object_states, workspace_offset=workspace_offset)
    annotation_target_metadata = _build_annotation_target_metadata(recipe_spec=recipe_spec, target_poses=target_poses)
    annotation_phase_metadata = _build_annotation_phase_metadata(recipe_spec=recipe_spec)

    return FactoryDualFrankaAssemblyTaskCfg(
        prompt=recipe_spec['prompt'],
        task_description=recipe_spec.get('task_description') or recipe_spec['prompt'],
        recipe=recipe_spec['task_name'],
        seed=seed,
        episode_idx=episode_idx,
        max_steps=int(recipe_spec.get('max_steps', 1800)),
        phase_timeout_steps=None
        if recipe_spec.get('phase_timeout_steps') is None
        else int(recipe_spec.get('phase_timeout_steps')),
        phase_timeout_action=str(recipe_spec.get('phase_timeout_action', 'fail')),
        phase_timeout_recovery_phase=recipe_spec.get('phase_timeout_recovery_phase'),
        object_state_sanity_enabled=bool(recipe_spec.get('object_state_sanity_enabled', True)),
        object_state_sanity_action=str(recipe_spec.get('object_state_sanity_action', 'fail')),
        object_state_sanity_max_position_norm=recipe_spec.get('object_state_sanity_max_position_norm', 50.0),
        object_state_sanity_max_reference_distance=recipe_spec.get('object_state_sanity_max_reference_distance', 10.0),
        object_state_sanity_max_linear_speed=recipe_spec.get('object_state_sanity_max_linear_speed', 50.0),
        object_state_sanity_max_angular_speed=recipe_spec.get('object_state_sanity_max_angular_speed', 500.0),
        object_state_sanity_recovery_attempts=int(
            3
            if recipe_spec.get('object_state_sanity_recovery_attempts') is None
            else recipe_spec.get('object_state_sanity_recovery_attempts')
        ),
        scene_asset_path=recipe_spec['scene_asset_path'],
        scene_asset_fallback_path=recipe_spec.get('scene_asset_fallback_path'),
        scene_scale=tuple(float(value) for value in recipe_spec.get('scene_scale', [1.0, 1.0, 1.0])),
        scene_position=tuple(float(value) for value in recipe_spec.get('scene_position', [0.0, 0.0, 0.0])),
        scene_orientation=tuple(float(value) for value in recipe_spec.get('scene_orientation', [1.0, 0.0, 0.0, 0.0])),
        robots=robots,
        objects=objects,
        robot_names=robot_names,
        tracked_object_names=tracked_object_names,
        phase_specs=_normalize_phase_specs(recipe_spec['phases']),
        target_poses=target_poses,
        success_criteria=_normalize_success_criteria(recipe_spec['success']),
        scene_profile=recipe_spec.get('scene_profile') or '',
        spec_path=recipe_spec.get('spec_path') or '',
        scene_profile_path=recipe_spec.get('scene_profile_path'),
        annotation_name=recipe_spec.get('annotation_name', ''),
        annotation_path=recipe_spec.get('annotation_path'),
        annotation_title=recipe_spec.get('annotation_title', ''),
        annotation_summary=recipe_spec.get('annotation_summary', ''),
        annotation_description=recipe_spec.get('annotation_description', ''),
        annotation_metadata=copy.deepcopy(recipe_spec.get('annotation_metadata', {})),
        annotation_object_roles=copy.deepcopy(recipe_spec.get('annotation_object_roles', {})),
        annotation_target_roles=copy.deepcopy(recipe_spec.get('annotation_target_roles', {})),
        annotation_phase_notes=copy.deepcopy(recipe_spec.get('annotation_phase_notes', [])),
        annotation_tags=copy.deepcopy(recipe_spec.get('annotation_tags', [])),
        target_annotations=annotation_target_metadata,
        phase_annotations=annotation_phase_metadata,
        workspace_offset=workspace_offset.tolist(),
        benchmark_metadata=copy.deepcopy(recipe_spec.get('metadata', {})),
        task_metadata=copy.deepcopy(recipe_spec.get('task_metadata', {})),
        scene_profile_metadata=copy.deepcopy(recipe_spec.get('scene_profile_metadata', {})),
        scene_lights=copy.deepcopy(recipe_spec.get('scene_lights', [])),
        asset_references=copy.deepcopy(recipe_spec.get('asset_references', [])),
        source_benchmark=str(recipe_spec.get('source_benchmark', recipe_spec.get('benchmark_family', 'factory_dual_franka_assembly'))),
        source_config_path=recipe_spec.get('source_config_path'),
        camera_metadata=copy.deepcopy(camera_metadata),
        robot_metadata=copy.deepcopy(recipe_spec.get('robots', [])),
        object_metadata=object_metadata,
    )


def build_dual_franka_assembly_batch(
    recipe: str,
    seeds: Iterable[int],
    scene_profile: str | None = None,
    attach_runtime_cameras: bool = False,
):
    return [
        build_dual_franka_assembly_episode(
            recipe=recipe,
            seed=seed,
            episode_idx=index,
            scene_profile=scene_profile,
            attach_runtime_cameras=attach_runtime_cameras,
        )
        for index, seed in enumerate(seeds)
    ]
