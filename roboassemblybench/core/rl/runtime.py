"""Object/arm-independent contact insertion; simulator access uses a bridge."""
from __future__ import annotations

import numpy as np

from .axis_alignment import BoreAlignment
from .control import (
    OBSERVATION_SIZE,
    BoundedAdmittance,
    InsertionConfig,
    InsertionGeometry,
    arbitrate_delta,
    bounded_relief_step,
    clip_norm,
    geometric_guide,
    pair_wrench,
    rotation_quaternion,
    rotation_vector,
    safety_failure,
    seated,
    seated_load_transfer_ready,
    validate_torque_hard_limit,
    validate_torque_soft_limit,
    vector,
    virtual_reference_limits,
)
from .geometry import compose_pose, quat_conjugate, quat_multiply, relative_pose
from .task import InsertionTask


class InsertionController:
    def __init__(
        self,
        task=None,
        *,
        binding: InsertionTask,
        config=None,
        bridge=None,
        torque_soft_limit=None,
        torque_hard_limit=None,
        axis_alignment=None,
        completion_stable_steps=None,
    ):
        if completion_stable_steps is not None and (
            type(completion_stable_steps) is not int or completion_stable_steps < 1
        ):
            raise ValueError('completion_stable_steps must be a positive integer')
        if bridge is None:
            from .backends.isaac_ur5e import IsaacUR5eBridge

            bridge = IsaacUR5eBridge(task)
        self.binding, self.bridge = binding, bridge
        self.config = config or InsertionConfig()
        self.completion_stable_steps = (
            self.config.seated_stable_steps if completion_stable_steps is None else completion_stable_steps
        )
        self.torque_soft_limit = validate_torque_soft_limit(torque_soft_limit, self.config)
        self.torque_hard_limit = validate_torque_hard_limit(torque_hard_limit, self.config, self.torque_soft_limit)
        if not np.isclose(bridge.physics_dt, 1.0 / self.config.physics_hz, rtol=1e-6, atol=0.0):
            raise ValueError(f'Insertion RL requires {self.config.physics_hz} Hz physics')
        bridge.initialize(binding)
        self.axis_alignment = None if axis_alignment is None else BoreAlignment(axis_alignment, binding)
        self.geometry = InsertionGeometry(
            base_pose=bridge.pose(binding.socket_object),
            entry_pose=bridge.target(binding.entry_target),
            goal_pose=bridge.target(binding.goal_target),
            insertion_axis=binding.insertion_axis,
            position_reference_point=binding.control_point_in_object,
        )
        self.support_anchor = (
            None
            if not binding.support_object
            else tuple(np.asarray(x).copy() for x in bridge.pose(binding.support_object))
        )
        self.admittance = BoundedAdmittance(self.config)
        self.action, self.previous_action = np.zeros(6), np.zeros(6)
        self.nominal_pose = tuple(np.asarray(x).copy() for x in bridge.tcp_pose(binding.insertion_robot))
        self.control_in_tcp = relative_pose(*self.nominal_pose, *bridge.control_pose(binding.insertion_robot))
        self.tcp_in_control = relative_pose(*bridge.control_pose(binding.insertion_robot), *self.nominal_pose)
        self.last_q = {robot: bridge.initial_arm_target(robot) for robot, _ in binding.participants}
        self.last_sample_step = self.last_tick_step = None
        self.stable_steps = self.ik_blocked_steps = 0
        self.failure = None
        self.local_seated = False
        self.last_sample = self.last_actions = None
        self.arbitration, self.guidance = 'initializing', True
        self.alignment_only = False

    def set_action(self, action):
        self.previous_action = self.action.copy()
        self.action = np.clip(vector(action, 6), -1.0, 1.0)

    def sample(self):
        b, c, stage = self.bridge, self.config, self.binding
        if self.last_sample_step == b.step:
            return self.last_sample
        moving, socket = b.pose(stage.moving_object), b.pose(stage.socket_object)
        tcp = b.tcp_pose(stage.insertion_robot)
        measurement = self.geometry.measure(moving, socket)
        if self.support_anchor is not None:
            p, q = b.pose(stage.support_object)
            measurement.update(
                support_position_drift=float(np.linalg.norm(p - self.support_anchor[0])),
                support_rotation_drift=float(
                    np.linalg.norm(rotation_vector(quat_multiply(q, quat_conjugate(self.support_anchor[1]))))
                ),
            )
        basis = self.geometry.world_basis(socket)
        alignment = (
            None
            if self.axis_alignment is None
            else self.axis_alignment.measure(moving, {name: b.pose(name) for name in stage.objects}, tcp[0], basis)
        )
        contact = b.contact_observation()
        wrench = pair_wrench(contact, tcp[0])
        physical_tcp = getattr(b, 'physical_tcp_pose', b.tcp_pose)(stage.insertion_robot)
        safety_wrench = pair_wrench(contact, physical_tcp[0])
        states = [b.grasp_state(robot, obj) for robot, obj in stage.participants]
        controls, grasps = [x[0] for x in states], [x[1] for x in states]
        wrench_task = np.r_[basis.T @ wrench[:3], basis.T @ wrench[3:]]
        failure = safety_failure(measurement, grasps, safety_wrench, c, torque_hard_limit=self.torque_hard_limit)
        failure = failure or next((x['failure'] for x in controls if x.get('failure')), None)
        self.failure = self.failure or failure
        release_ready = (
            seated_load_transfer_ready(controls, grasps)
            if stage.allow_seated_load_transfer
            else all(x.get('motion_ready') for x in controls)
        )
        wrist_torque = float(np.linalg.norm(safety_wrench[3:]))
        torque_limited = self.torque_soft_limit is not None and wrist_torque >= self.torque_soft_limit
        aligned = (
            seated(measurement, c)
            and not self.failure
            and release_ready
            and not torque_limited
            and (alignment is None or alignment['ready'])
        )
        self.stable_steps = self.stable_steps + 1 if aligned else 0
        self.local_seated = self.stable_steps >= self.completion_stable_steps
        v, w = (vector(x, 3) for x in b.velocity(stage.moving_object))
        sv, sw = (vector(x, 3) for x in b.velocity(stage.socket_object))
        rel_v, rel_w = v - sv - np.cross(sw, moving[0] - socket[0]), w - sw
        lead_p = basis.T @ (self.nominal_pose[0] - tcp[0])
        lead_r = basis.T @ rotation_vector(quat_multiply(self.nominal_pose[1], quat_conjugate(tcp[1])))
        offset = np.r_[basis.T @ self.admittance.offset[:3], basis.T @ self.admittance.offset[3:]]
        # Insert arm first, optional support second. Unused network slots are
        # zero padding, never invented physical grasp measurements.
        forces, slips, ready, limiting = np.zeros(4), np.zeros(4), np.zeros(2), np.zeros(2)
        for i, (ctrl, grasp) in enumerate(states):
            forces[2 * i : 2 * i + 2] = ctrl.get('forces', [0.0, 0.0])
            slips[2 * i : 2 * i + 2] = [
                grasp['position_slip'] / c.grasp_position_limit,
                grasp['orientation_slip'] / c.grasp_rotation_limit,
            ]
            ready[i] = float(ctrl.get('motion_ready', False))
            limiting[i] = float(ctrl.get('mode') in ('force_limited', 'force_relief', 'unload'))
        obs = np.r_[
            measurement['error'] / 0.05,
            measurement['rotation_error'] / 0.2,
            basis.T @ rel_v / 0.05,
            basis.T @ rel_w / 0.5,
            measurement['base_delta'] / c.base_position_limit,
            measurement['base_angle'] / c.base_rotation_limit,
            wrench_task[:3] / 20.0,
            wrench_task[3:],
            forces / 20.0,
            slips,
            ready,
            limiting,
            lead_p / c.position_lead,
            lead_r / c.rotation_lead,
            self.action,
            measurement['progress'],
            offset[:3] / c.compliance_position_limit,
            offset[3:] / c.compliance_rotation_limit,
        ]
        if obs.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(obs)):
            raise ValueError('Invalid physical insertion observation')
        self.last_sample = dict(
            observation=np.clip(obs, -10.0, 10.0).astype(np.float32),
            measurement=measurement,
            controls=controls,
            grasps=grasps,
            wrench_world=wrench,
            safety_wrench_world=safety_wrench,
            wrist_torque_norm=wrist_torque,
            torque_soft_limit=self.torque_soft_limit,
            torque_hard_limit=self.torque_hard_limit,
            torque_soft_active=torque_limited,
            wrench_task=wrench_task,
            contact=contact,
            failure=self.failure,
            local_seated=self.local_seated,
            axis_alignment=alignment,
        )
        self.last_sample_step = b.step
        return self.last_sample

    def tick(self):
        b, c, stage = self.bridge, self.config, self.binding
        robot = stage.insertion_robot
        if self.last_tick_step == b.step:
            return self.last_actions
        s = self.sample()
        if s['failure']:
            raise ValueError(s['failure'])
        basis = self.geometry.world_basis(b.pose(stage.socket_object))
        policy_delta = self.action * np.r_[np.full(3, c.translation_step), np.full(3, c.rotation_step)]
        if self.guidance:
            guide = geometric_guide(s['measurement'], c, alignment_deadband=stage.control_point_in_object is not None)
            if self.axis_alignment is not None:
                correction = self.axis_alignment.bounded_correction(s['axis_alignment'], 1.0 / c.physics_hz)
                guide[:2] = (basis.T @ correction[:3])[:2] * c.decimation
                guide[3:5] = (basis.T @ correction[3:])[:2] * c.decimation
                guide[2] = (
                    np.clip(s['measurement']['axial_remaining'], -c.guide_translation_step, c.guide_translation_step)
                    if s['axis_alignment']['ready'] and s['measurement']['angle_error'] <= c.seated_rotation_tolerance
                    else 0.0
                )
            if self.alignment_only:
                guide[2] = 0.0
            policy_delta += guide
            if self.axis_alignment is not None and not s['axis_alignment']['ready']:
                # Search resumes inside the shaft envelope. Outside it the
                # residual must not cancel the measured alignment correction.
                policy_delta[:2] = guide[:2]
                policy_delta[3:5] = guide[3:5]
        # The explicit seated handover mode is a dwell, not another insertion
        # stroke. Continue reading every force/slip guard and regulating the
        # jaws, but do not let residual exploration disturb an already seated
        # part while its stability counter is accumulating.
        seated_dwell = (
            stage.allow_seated_load_transfer
            and seated(s['measurement'], c)
            and not s['torque_soft_active']
            and (self.axis_alignment is None or s['axis_alignment']['ready'])
        )
        if seated_dwell:
            policy_delta[:] = 0.0
        if self.axis_alignment is not None:
            if not s['axis_alignment']['ready'] or s['measurement']['angle_error'] > c.seated_rotation_tolerance:
                policy_delta[2] = 0.0
        elif stage.control_point_in_object is not None and (
            s['measurement']['lateral_error'] > c.seated_lateral_tolerance
            or s['measurement']['angle_error'] > c.seated_rotation_tolerance / 4.0
        ):
            # Residual search may retreat or align; it must not bypass the
            # guide's alignment gate by commanding forward feed itself.
            policy_delta[2] = min(0.0, policy_delta[2])
        if any(g.get('recovering_contact') for g in s['grasps']):
            delta, self.arbitration = np.zeros(6), 'grasp_recovery_hold'
        else:
            delta, self.arbitration = arbitrate_delta(
                policy_delta,
                s['controls'],
                s['wrench_task'],
                c,
                torque_soft_limit=self.torque_soft_limit,
                safety_wrench=s['safety_wrench_world'],
            )
        if seated_dwell and self.arbitration != 'grasp_recovery_hold':
            self.arbitration = 'seated_dwell'
        offset = self.admittance.update(s['wrench_world'], 1.0 / c.physics_hz)
        current_p, current_q = (np.asarray(x) for x in b.tcp_pose(robot))
        vp, vr = virtual_reference_limits(c)
        nominal_p = current_p + clip_norm(self.nominal_pose[0] + basis @ delta[:3] / c.decimation - current_p, vp)
        nominal_q = quat_multiply(rotation_quaternion(basis @ delta[3:] / c.decimation), self.nominal_pose[1])
        lead_r = clip_norm(rotation_vector(quat_multiply(nominal_q, quat_conjugate(current_q))), vr)
        nominal_q = quat_multiply(rotation_quaternion(lead_r), current_q)
        hold = (
            seated_dwell
            or self.arbitration == 'grasp_recovery_hold'
            or (self.arbitration == 'retreat_only' and delta[2] >= -1e-12 and (basis.T @ offset[:3])[2] >= -1e-12)
            or (np.linalg.norm(delta) < 1e-12 and np.linalg.norm(offset) < 1e-10)
        )
        commands = dict(self.last_q)
        if not hold:
            target_p = current_p + clip_norm(nominal_p + offset[:3] - current_p, c.position_lead)
            target_q = quat_multiply(rotation_quaternion(offset[3:]), nominal_q)
            lead = clip_norm(rotation_vector(quat_multiply(target_q, quat_conjugate(current_q))), c.rotation_lead)
            target_q = quat_multiply(rotation_quaternion(lead), current_q)
            if self.arbitration in ('retreat_only', 'insertion_force_limit', 'insertion_torque_limit'):
                movement = basis.T @ (target_p - current_p)
                movement[2] = min(0.0, movement[2])
                if self.arbitration == 'retreat_only':
                    movement[2] = max(movement[2], -c.compliance_linear_speed / c.physics_hz)
                    movement[:2], target_q = 0.0, current_q.copy()
                target_p = current_p + basis @ movement
                # A clipped, zero backoff must preserve the previous joint
                # target, including its holding bias; do not solve measured FK.
                hold = self.arbitration == 'retreat_only' and movement[2] >= -1e-12
            if self.arbitration == 'compliant_relief':
                command_p, command_q = compose_pose(*b.commanded_control_pose(robot), *self.tcp_in_control)
                candidate = np.r_[
                    basis.T @ (target_p - command_p),
                    basis.T @ rotation_vector(quat_multiply(target_q, quat_conjugate(command_q))),
                ]
                correction = bounded_relief_step(candidate, s['wrench_task'], c)
                hold = np.linalg.norm(correction) < 1e-12
                target_p = command_p + basis @ correction[:3]
                target_q = quat_multiply(rotation_quaternion(basis @ correction[3:]), command_q)
            if self.axis_alignment is not None:
                command_p, command_q = compose_pose(*b.commanded_control_pose(robot), *self.tcp_in_control)
                if not s['axis_alignment']['ready'] and self.arbitration == 'policy':
                    correction = self.axis_alignment.bounded_correction(s['axis_alignment'], 1.0 / c.physics_hz)
                    target_p = command_p + correction[:3]
                    target_q = quat_multiply(rotation_quaternion(correction[3:]), command_q)
                    nominal_p = target_p - offset[:3]
                    nominal_q = quat_multiply(rotation_quaternion(-offset[3:]), target_q)
                if not s['axis_alignment']['ready']:
                    movement = basis.T @ (target_p - command_p)
                    movement[2] = min(0.0, movement[2])
                    target_p = command_p + basis @ movement
                (target_p, target_q), limited = self.axis_alignment.constrain_target(
                    target=(target_p, target_q),
                    current=(command_p, command_q),
                    moving=b.pose(stage.moving_object),
                    poses={name: b.pose(name) for name in stage.objects},
                    basis=basis,
                    sample=s['axis_alignment'],
                    dt=1.0 / c.physics_hz,
                    relief_start=(command_p, command_q) if self.arbitration == 'compliant_relief' else None,
                )
                s['axis_target_limited'] = limited
                s['axis_command_reference'] = (command_p, command_q)
                s['axis_control_pose'] = (current_p, current_q)
                s['axis_target_pose'] = (target_p, target_q)
                if limited:
                    # Back-calculate only the virtual command spring. Never
                    # accumulate a rejected off-axis reference behind the gate.
                    nominal_p = target_p - offset[:3]
                    nominal_q = quat_multiply(rotation_quaternion(-offset[3:]), target_q)
                if self.arbitration == 'compliant_relief':
                    hold = (
                        np.linalg.norm(target_p - command_p) < 1e-12
                        and np.linalg.norm(rotation_vector(quat_multiply(target_q, quat_conjugate(command_q)))) < 1e-12
                    )
                # Alignment must retain the existing target lead envelope.
                if (
                    np.linalg.norm(target_p - current_p) > c.position_lead + 1e-9
                    or np.linalg.norm(rotation_vector(quat_multiply(target_q, quat_conjugate(current_q))))
                    > c.rotation_lead + 1e-9
                ):
                    hold = True
                    s['axis_command_lead_blocked'] = True
        if not hold:
            p, q = compose_pose(target_p, target_q, *self.control_in_tcp)
            command, fraction = b.solve_ik(robot, self.last_q[robot], dict(position=p, orientation=q), c)
            if command is None:
                self.ik_blocked_steps += 1
                self.last_sample['ik_failure'] = dict(
                    target_position=p.tolist(), target_orientation=q.tolist(), reference_q=self.last_q[robot].tolist()
                )
                if self.ik_blocked_steps >= 48:
                    raise ValueError('local_insertion_ik_failed')
            else:
                self.ik_blocked_steps = 0
                self.last_sample['ik_accepted_fraction'] = fraction
                dq = (command - self.last_q[robot] + np.pi) % (2 * np.pi) - np.pi
                if np.max(abs(dq)) > c.joint_step:
                    dq *= c.joint_step / np.max(abs(dq))
                commands[robot] = self.last_q[robot] + dq
                self.nominal_pose = (
                    self.nominal_pose[0] + fraction * (nominal_p - self.nominal_pose[0]),
                    quat_multiply(
                        rotation_quaternion(
                            fraction * rotation_vector(quat_multiply(nominal_q, quat_conjugate(self.nominal_pose[1])))
                        ),
                        self.nominal_pose[1],
                    ),
                )
        actions = {}
        for (name, _), ctrl in zip(stage.participants, s['controls']):
            actions[name] = b.joint_action(name, commands[name], ctrl, c)
            self.last_q[name] = np.asarray(commands[name]).copy()
        self.last_tick_step, self.last_actions = b.step, actions
        return actions
