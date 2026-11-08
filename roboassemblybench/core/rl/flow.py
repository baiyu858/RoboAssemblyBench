"""Resume a recorded physical scene and capture a later insertion entry.

State writes occur only during reset. The continuation uses the ordinary
recipe policy and reconstructs contact forces with real physics steps.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .snapshot import json_value

FORMAT = 'roboassemblybench-physical-flow-v1'


def refresh_grasp_force_control(attachment, phases, resume_phase):
    """Use the current recipe's force settings; retain all captured grasp anchors."""
    result = copy.deepcopy(attachment)
    spec = result['attach_spec']
    for phase in phases:
        if phase['name'] == resume_phase:
            break
        for authored in phase.get('attach', []):
            if (
                authored.get('object') == spec['object']
                and authored.get('robot') == result['robot_name']
                and authored.get('hold_force_control', {}).get('enabled')
            ):
                spec['hold_force_control'] = copy.deepcopy(authored['hold_force_control'])
    return result


def restore_reference_targets(data, entry_step):
    """Preserve the active phase's frozen receiver frame across a reset."""
    targets = copy.deepcopy(data.get('reference_targets', {}))
    for key, pose in list(targets.items()):
        if key.startswith(data['phase'] + ':'):
            name = key.rsplit(':', 1)[-1]
            targets[f"{data['phase']}:{entry_step}:{name}"] = copy.deepcopy(pose)
    return targets


def refresh_flow_targets(restored, authored, anchors):
    """Apply newly authored target offsets in the recorded assembly frame."""
    from .geometry import compose_pose, relative_pose

    result = copy.deepcopy(restored)
    for target, anchor in anchors.items():
        local = relative_pose(
            authored[anchor]['position'],
            authored[anchor]['orientation'],
            authored[target]['position'],
            authored[target]['orientation'],
        )
        pose = compose_pose(result[anchor]['position'], result[anchor]['orientation'], *local)
        result[target] = dict(zip(('position', 'orientation'), pose))
    return result


def read_flow_snapshot(path):
    data = json.loads(Path(path).read_text())
    if data.get('format') not in (FORMAT, 'debug-physical-scene-state-v1'):
        raise ValueError('Unsupported physical flow snapshot')
    grasps = data.get('physical_grasps') or [dict(object=n, state=s) for n, s in data['attachments'].items()]
    for entry in grasps:
        attachment = entry['state']
        if (
            attachment.get('mode') != 'pure_physical_grasp'
            or attachment.get('joint_path')
            or attachment.get('collision_disabled')
            or attachment.get('kinematic_peel_carry')
            or not attachment.get('attach_spec', {}).get('continuous_physics')
        ):
            raise ValueError('Flow continuation requires real, collidable physical grasps')
    if any(state.get('locked_target') for state in data['objects'].values()):
        raise ValueError('Flow continuation cannot restore object locks')
    return data


def save_flow_snapshot(task, path, *, resume_phase=None):
    data = dict(
        format=FORMAT,
        reset_only=True,
        recipe=task.cfg.recipe,
        recipe_overrides=(task.cfg.task_metadata or {}).get('recipe_overrides', {}),
        phase=resume_phase or task.phase,
        source_step=int(task.step_counter),
        seed=task.cfg.seed,
        layout_seed=task.cfg.layout_seed,
        objects=task.get_tracked_object_states(),
        attachments=task._attachments,
        physical_grasps=[dict(object=name, state=state) for name, state in task._all_physical_grasps()],
        targets=task.target_poses,
        hold_commands=getattr(task, '_ur5e_contact_hold_commands', {}),
        last_arm_commands=getattr(task, '_ur5e_plumbers_last_arm_command_q', {}),
        rebased_anchors=sorted(getattr(task, '_rebased_anchors', set())),
        setup_pose_anchors=getattr(task, '_physical_setup_pose_anchors', {}),
        reference_targets=getattr(task, '_physical_reference_targets', {}),
        robots={
            name: dict(
                dof_names=list(robot.articulation.dof_names),
                joint_positions=robot.articulation.get_joint_positions(),
                joint_velocities=robot.articulation.get_joint_velocities(),
            )
            for name, robot in task.robots.items()
        },
    )
    path = Path(path)
    path.write_text(json.dumps(data, default=json_value, indent=2))
    return data


def build_flow_env(  # noqa: C901
    snapshot_path, *, recipe=None, overrides=None, video=False, diagnostics_dir=None, target_refresh_anchors=None
):
    from roboassemblybench.core.scene_profiles import deep_merge
    from toolkits.factory_dual_franka_assembly.generate_demos import _build_env
    from toolkits.factory_dual_franka_assembly.scene_builder import (
        build_dual_franka_assembly_episode,
    )

    data = read_flow_snapshot(snapshot_path)
    grasps = data.get('physical_grasps') or [dict(object=n, state=s) for n, s in data['attachments'].items()]
    cfg = build_dual_franka_assembly_episode(
        recipe or data['recipe'],
        seed=data.get('seed', 4906),
        layout_seed=data.get('layout_seed', 4906),
        attach_runtime_cameras=video,
        control_fps=240,
        recipe_overrides=deep_merge(data.get('recipe_overrides', {}), overrides or {}),
    )
    cfg.max_steps = 10**9
    from toolkits.factory_dual_franka_assembly.manipulation_stabilization import (
        thresholds_through,
    )
    from toolkits.factory_dual_franka_assembly.object_friction import friction_through

    # A continuation skips earlier phase actions. Restore their numerical
    # policy before physics starts instead of re-freezing already active parts.
    stabilization = thresholds_through(cfg.phase_specs, data['phase'])
    friction = friction_through(cfg.phase_specs, data['phase'])
    for obj in cfg.objects:
        if obj.name in friction:
            if not obj.rigid_body:
                raise ValueError(f'Friction policy references a static object: {obj.name}')
            obj.static_friction = friction[obj.name]['static_friction']
            obj.dynamic_friction = friction[obj.name]['dynamic_friction']
        if obj.name in stabilization:
            if not obj.rigid_body:
                raise ValueError(f'Stabilization policy references a static object: {obj.name}')
            obj.stabilization_threshold = stabilization[obj.name]
        if obj.name in data['objects']:
            state = data['objects'][obj.name]
            obj.position, obj.orientation = tuple(state['position']), tuple(state['orientation'])
    for robot in cfg.robots:
        state = data['robots'][robot.name]
        robot.initial_joint_positions = dict(zip(state['dof_names'], state['joint_positions']))
        if not video:
            robot.sensors = []
    if not video:
        cfg.camera_metadata = []
    env = _build_env([cfg], headless=True, rendering_fps=240, rendering_interval=15 if video else 999999)
    try:
        env.reset()
        task = next(iter(env.runner.current_tasks.values()))
        full_phases = copy.deepcopy(task.phase_specs)
        names = [p['name'] for p in full_phases]
        if data['phase'] not in names:
            raise ValueError(f"Cannot resume missing phase {data['phase']}")
        # Hold an empty arm at the recorded joint state. A released object's
        # cached grasp command is stale and must never pull that arm back down.
        actions = {}
        for name, robot in task.robots.items():
            active = any(entry['state']['robot_name'] == name for entry in grasps)
            saved = data['hold_commands'].get(name) if active else None
            if active and saved is None:
                raise ValueError(f'{name}: held object has no recorded joint target')
            if saved:
                actions[name] = dict(
                    arm_joint_controller=[saved['arm_q']], gripper_controller=[saved['gripper_openness']]
                )
            else:
                state = data['robots'][name]
                q = state['joint_positions'][state['dof_names'].index(robot.config.gripper_dof_name)]
                op, cl = robot.config.gripper_open_position, robot.config.gripper_closed_position
                arm_names = robot.controllers['arm_joint_controller'].config.joint_names
                arm = [state['joint_positions'][state['dof_names'].index(n)] for n in arm_names]
                actions[name] = dict(
                    arm_joint_controller=[arm], gripper_controller=[float(np.clip((q - cl) / (op - cl), 0.0, 1.0))]
                )
            robot.apply_action(actions[name])
        env.warm_up(steps=16, render=video, physics=True)
        task.phase_specs = [
            dict(
                name='physical_flow_reset',
                robot_targets={},
                timeout_steps=10**9,
                advance=dict(type='local_skill_complete', robot='__reset__', skill='__reset__'),
            )
        ]
        task.phase_index, task.phase, task._phase_initialized = 0, 'physical_flow_reset', True
        task._attachments = {}
        task._beam_shared_grasps = {}
        task._locked_targets.clear()
        task._frozen_lock_poses.clear()
        task._object_pose_history.clear()
        done = task.is_done

        def retain():
            done()
            return False

        task.is_done = retain
        from isaacsim.core.simulation_manager import SimulationManager

        for name, robot in task.robots.items():
            state = data['robots'][name]
            if list(robot.articulation.dof_names) != state['dof_names']:
                raise ValueError(f'{name}: snapshot DOF order differs from asset')
            robot.articulation.set_joint_positions(np.asarray(state['joint_positions']))
            robot.articulation.set_joint_velocities(np.zeros(len(state['joint_positions'])))
        for name, state in data['objects'].items():
            obj = task._resolve_object(name)
            if state.get('linear_velocity') is not None:
                obj.unwrap().initialize(physics_sim_view=SimulationManager.get_physics_sim_view())
            obj.set_pose(np.asarray(state['position']), np.asarray(state['orientation']))
            task._zero_object_velocity(name)
        for entry in grasps:
            attachment = entry['state']
            task._gripper_contact_metrics(
                entry['object'], attachment['robot_name'], attach_spec=attachment['attach_spec']
            )
        for _ in range(32):
            env.step([actions])
        task.failed = task.success = False
        task.terminal_reason = None
        task._attachments = {}
        task._beam_shared_grasps = {}
        for entry in copy.deepcopy(grasps):
            attachment = refresh_grasp_force_control(entry['state'], full_phases, data['phase'])
            for key in ('position', 'orientation'):
                attachment[key] = np.asarray(attachment[key])
            task._store_physical_grasp(entry['object'], attachment)
        active_robots = {entry['state']['robot_name'] for entry in grasps}
        task._ur5e_contact_hold_commands = {
            name: copy.deepcopy(saved) for name, saved in data['hold_commands'].items() if name in active_robots
        }
        task._ur5e_plumbers_last_arm_command_q = {
            name: np.asarray(action['arm_joint_controller'][0]).copy() for name, action in actions.items()
        }
        for saved in task._ur5e_contact_hold_commands.values():
            saved['arm_q'] = np.asarray(saved['arm_q'])
            saved['pose'] = {key: np.asarray(value) for key, value in saved['pose'].items()}
            for key in (
                'loaded_force_state',
                'rotation_command_state',
                'rotation_command_pending',
                'rotation_published_state',
            ):
                saved.pop(key, None)
        # Older local-RL recordings updated arm_q but left the Cartesian
        # hold pose at the insertion entry. Reconstruct this command metadata
        # from FK, without modifying any physical object or grasp reference.
        from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import (
            UR5eAssemblyAtomicSkillAdapter,
        )

        atomic = UR5eAssemblyAtomicSkillAdapter({})
        for name, saved in task._ur5e_contact_hold_commands.items():
            pose = atomic._grasp_command_fk_pose(task=task, robot_name=name, spec={}, command_q=saved['arm_q'])
            if pose is None:
                raise ValueError(f'{name}: restored command FK unavailable')
            saved['pose'] = {key: np.asarray(pose[key]).copy() for key in ('position', 'orientation')}
        task.target_poses = refresh_flow_targets(
            {name: {key: np.asarray(value) for key, value in pose.items()} for name, pose in data['targets'].items()},
            task.target_poses,
            target_refresh_anchors or {},
        )
        task._rebased_anchors = set(data.get('rebased_anchors', []))
        task._physical_setup_pose_anchors = copy.deepcopy(data.get('setup_pose_anchors', {}))
        task._physical_reference_targets = copy.deepcopy(data.get('reference_targets', {}))
        task.phase_specs = task.cfg.phase_specs = full_phases
        task._set_phase(names.index(data['phase']), reason='physical-flow-reset', transition_type='resume')
        task._physical_reference_targets = restore_reference_targets(data, task.phase_entry_step)
        for obj, attachment in task._all_physical_grasps():
            if not task._physical_hold_valid(obj, attachment, validation_stage='flow_reset_qualification'):
                raise RuntimeError(f'{obj}: real-contact flow reset failed')
        return env, task
    except BaseException:
        import traceback

        detail = traceback.format_exc()
        print(detail, flush=True)
        if diagnostics_dir is not None:
            (Path(diagnostics_dir) / 'flow_reset_error.txt').write_text(detail)
        env.close()
        raise
