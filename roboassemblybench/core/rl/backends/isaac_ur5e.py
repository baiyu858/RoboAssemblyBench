"""Isaac Sim task/UR5e bridge; identities come from InsertionTask."""
import numpy as np


class IsaacUR5eBridge:
    def __init__(self, task):
        from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import (
            UR5eAssemblyAtomicSkillAdapter,
        )

        self.task = task
        self.atomic = UR5eAssemblyAtomicSkillAdapter({})

    @property
    def step(self):
        return int(self.task.step_counter)

    @property
    def physics_dt(self):
        return self.task._contact_physics_dt()

    def initialize(self, binding):
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdPhysics

        from toolkits.factory_dual_franka_assembly.contact_reports import (
            create_prepared_contact_view,
        )

        self.binding = binding
        t = self.task
        for robot, _ in binding.participants:
            if 'ur5e' not in str(t.robots[robot].config.type).lower():
                raise ValueError(f'{robot}: IsaacUR5eBridge requires UR5e; supply another bridge for this robot')
        for name in binding.objects:
            if name in t._locked_targets or not t._object_collision_enabled.get(name, True):
                raise ValueError(f'{name}: insertion requires unlocked collidable objects')
            attachment = t._attachments.get(name)
            if attachment and attachment.get('mode') != 'pure_physical_grasp':
                raise ValueError(f'{name}: nonphysical attachment in insertion scene')
        if binding.socket_is_static:
            physical = list(Usd.PrimRange(t._object_prims[binding.socket_object], Usd.TraverseInstanceProxies()))
            if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in physical) or any(
                p.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False
                for p in physical
            ):
                raise ValueError('socket_is_static requires an authored static collider')
        dynamic = [binding.moving_object]
        if not binding.socket_is_static:
            dynamic.append(binding.socket_object)
        if binding.support_object:
            dynamic.append(binding.support_object)
        for name in dict.fromkeys(dynamic):
            t._resolve_object(name).unwrap().initialize(physics_sim_view=SimulationManager.get_physics_sim_view())
        moving_path = t._resolve_object(binding.moving_object).unwrap().prim_path
        filters = [t._resolve_object(name).unwrap().prim_path for name in binding.contact_objects]
        if any(
            a == b or a.startswith(b + '/') or b.startswith(a + '/')
            for i, a in enumerate(filters)
            for b in filters[i + 1 :]
        ):
            raise ValueError('Overlapping contact filters would double-count insertion forces')
        self.pair_paths = []
        for index, (name, path) in enumerate(zip(binding.contact_objects, filters)):
            key = (moving_path, path)
            t._contact_probes[key] = create_prepared_contact_view(
                t._object_prims[binding.moving_object],
                t._object_prims[name],
                name=f'insertion_rl_contact_{binding.stage_id}_{index}',
                disable_stablization=False,
                max_contact_count=512,
            )
            self.pair_paths.append(key)
        self._initialize_control_point(binding)

    def _initialize_control_point(self, binding):
        from roboassemblybench.core.rl.geometry import quat_conjugate, quat_rotate

        self.control_point_in_tool = None
        if binding.control_point_in_object is not None:
            p, q = self.physical_tcp_pose(binding.insertion_robot)
            op, oq = self.pose(binding.moving_object)
            point = op + quat_rotate(oq, binding.control_point_in_object)
            self.control_point_in_tool = quat_rotate(quat_conjugate(q), point - p).copy()

    def pose(self, name):
        return tuple(np.asarray(x, dtype=float) for x in self.task._resolve_object(name).get_pose())

    def target(self, name):
        return tuple(np.asarray(self.task.target_poses[name][k], dtype=float) for k in ('position', 'orientation'))

    def velocity(self, name):
        if name == self.binding.socket_object and self.binding.socket_is_static:
            return np.zeros(3), np.zeros(3)
        raw = self.task._object_velocity_metrics(name)
        if not raw.get('valid'):
            raise ValueError(f'{name}: native velocity unavailable')
        return raw['linear_velocity'], raw['angular_velocity']

    def tcp_pose(self, robot):
        from roboassemblybench.core.rl.geometry import quat_rotate

        p, q = self.physical_tcp_pose(robot)
        offset = getattr(self, 'control_point_in_tool', None)
        if robot == self.binding.insertion_robot and offset is not None:
            p = p + quat_rotate(q, offset)
        return p, q

    def physical_tcp_pose(self, robot):
        return self.task._get_robot_task_pose(robot)

    def control_pose(self, robot):
        return self.task._get_robot_control_pose(robot)

    def initial_arm_target(self, robot):
        saved = getattr(self.task, '_ur5e_contact_hold_commands', {}).get(robot)
        if saved is None:
            raise ValueError(f'{robot}: missing physical grasp command at handover')
        return np.asarray(saved['arm_q'], dtype=float).copy()

    def commanded_control_pose(self, robot):
        """FK metadata of the last published joint target, including holding bias."""
        pose = self.task._ur5e_contact_hold_commands[robot]['pose']
        return tuple(np.asarray(pose[k], dtype=float).copy() for k in ('position', 'orientation'))

    def contact_observation(self):
        t = self.task
        result = dict(
            valid=True,
            details_valid=True,
            details_at_capacity=False,
            friction_valid=True,
            force_world=np.zeros(3),
            contacts=[],
            friction_anchors=[],
        )
        for counterpart, pair in zip(self.binding.contact_objects, self.pair_paths, strict=True):
            observation = t._pair_contact_observation(*pair)
            if not observation.get('friction_valid'):
                probe = t._get_contact_probe(*pair)
                values = [t._tensor_to_numpy(x) for x in probe.get_friction_data(dt=self.physics_dt)]
                if any(x is None for x in values):
                    raise ValueError('Insertion friction buffers unavailable')
                forces, points, counts, starts = values
                forces, points = forces.reshape(-1, 3), points.reshape(-1, 3)
                anchors = []
                for count, start in zip(counts.reshape(-1), starts.reshape(-1)):
                    count, start = int(count), int(start)
                    if start < 0 or count < 0 or start + count > min(len(forces), len(points)):
                        raise ValueError('Insertion friction buffer range invalid')
                    anchors.extend(
                        dict(force_world=forces[i].tolist(), point_world=points[i].tolist())
                        for i in range(start, start + count)
                    )
                observation.update(friction_valid=True, friction_anchors=anchors)
            if (
                not observation.get('valid')
                or not observation.get('details_valid')
                or observation.get('details_at_capacity')
            ):
                raise ValueError(f'Insertion contact buffers invalid: {observation}')
            result['force_world'] += np.asarray(observation['force_world'])
            result['contacts'].extend(dict(c, contact_object=counterpart) for c in observation.get('contacts', []))
            result['friction_anchors'].extend(
                dict(c, contact_object=counterpart) for c in observation.get('friction_anchors', [])
            )
        return result

    def grasp_state(self, robot, obj):
        t = self.task
        attachment = t._attachments.get(obj)
        saved = getattr(t, '_ur5e_contact_hold_commands', {}).get(robot)
        if (
            attachment is None
            or attachment.get('mode') != 'pure_physical_grasp'
            or attachment.get('robot_name') != robot
            or saved is None
        ):
            raise ValueError(f'{obj}: physical grasp lost')
        control = self.atomic._update_loaded_hold_force(task=t, robot_name=robot, saved=saved, attachment=attachment)
        if control.get('failure') == 'held_object_not_grasped':
            from toolkits.factory_dual_franka_assembly.contact_recovery import (
                retain_contact_recovery,
            )

            if retain_contact_recovery(t, obj, attachment):
                # The controller can be the first reader of a new force
                # sample. Register the same bounded reclose as state_sync,
                # without marking the missing contact valid.
                control = self.atomic._update_loaded_hold_force(
                    task=t, robot_name=robot, saved=saved, attachment=attachment
                )
        force_config = attachment['attach_spec'].get('hold_force_control', {})
        control = dict(control, low_force=force_config.get('low', 6.0), unload_force=force_config.get('unload', 18.0))
        hold = attachment.get('last_hold_validation') or {}
        grasp = dict(
            valid=bool(hold.get('valid')),
            object=obj,
            reasons=hold.get('reasons'),
            recovering_contact=bool(attachment.get('contact_recovery')),
            position_slip=float(hold.get('position_slip', np.inf)),
            orientation_slip=float(hold.get('orientation_slip', np.inf)),
        )
        if attachment['attach_spec'].get('hold_orientation_slip_limit_override'):
            grasp['orientation_slip_limit'] = attachment['attach_spec']['hold_orientation_slip_tolerance']
        return control, grasp

    def solve_ik(self, robot, reference_q, target_pose, config):
        t = self.task
        ik = t.robots[robot].controllers['arm_ik_controller']
        base = ik.get_ik_base_world_pose()
        ik._kinematics_solver.set_robot_base_pose(robot_position=base[0] / ik._robot_scale, robot_orientation=base[1])
        reference_pose = self.atomic._grasp_command_fk_pose(task=t, robot_name=robot, spec={}, command_q=reference_q)
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            proposal = self.atomic._cartesian_pose_fraction(
                current_pose=reference_pose, target_pose=target_pose, ratio=fraction
            )
            q = self.atomic._local_grasp_ik(
                task=t,
                robot_name=robot,
                target_pose=proposal,
                reference_q=reference_q,
                joint_radius=config.joint_step,
                spec={'ik_position_tolerance': 1e-6, 'ik_orientation_tolerance': 1e-5},
            )
            if q is not None:
                return np.asarray(q), fraction
        return None, 0.0

    def joint_action(self, robot, q, control, config):
        measured = self.atomic._current_arm_q(self.task, robot)
        if (
            measured is None
            or not np.all(np.isfinite(q))
            or np.max(abs(np.asarray(q) - measured)) > config.joint_tracking_limit
        ):
            raise ValueError(f'{robot}: joint_tracking_limit')
        saved = self.task._ur5e_contact_hold_commands[robot]
        if not np.array_equal(np.asarray(saved['arm_q']), np.asarray(q)):
            pose = self.atomic._grasp_command_fk_pose(task=self.task, robot_name=robot, spec={}, command_q=q)
            if pose is None:
                raise ValueError(f'{robot}: published insertion command has no FK pose')
            saved['pose'] = {key: np.asarray(pose[key]).copy() for key in ('position', 'orientation')}
        saved['arm_q'] = np.asarray(q).copy()
        self.atomic._remember_arm_command(self.task, robot, q)
        return dict(
            arm_joint_controller=[np.asarray(q).tolist()], gripper_controller=[float(control['command_openness'])]
        )
