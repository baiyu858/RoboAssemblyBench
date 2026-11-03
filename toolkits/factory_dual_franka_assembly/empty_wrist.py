"""Remove wrist winding through real joint motion at an empty clearance pose."""
import math

import numpy as np


class EmptyWristUnwindAdapter:
    def __init__(self, spec):
        from .plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter
        self.atomic = UR5eAssemblyAtomicSkillAdapter({})
        self.states = {}

    def act(self, *, task, robot_name, phase_spec, skill_spec, **kwargs):
        spec = skill_spec
        if any(state['robot_name'] == robot_name for _, state in task._all_physical_grasps()):
            return self.atomic._failure_or_hold(task, robot_name, spec, 'unwind_requires_empty_gripper')
        q = self.atomic._current_arm_q(task, robot_name)
        if q is None or not np.all(np.isfinite(q)):
            return self.atomic._failure_or_hold(task, robot_name, spec, 'unwind_joint_state_unavailable')
        key = (id(task), task.phase_index, task.phase_entry_step, robot_name)
        if key not in self.states:
            target = q.copy()
            winding = spec.get('winding', 'centered')
            if winding not in ('centered', 'negative'):
                raise ValueError('Unwind winding must be centered or negative')
            target[-1] = ((q[-1] + math.pi) % (2 * math.pi) - math.pi if winding == 'centered'
                          else (q[-1] % (2 * math.pi) - 2 * math.pi))
            if abs(target[-1] + 2 * math.pi) < 1e-6:
                target[-1] = 0.
            native = task.robots[robot_name].articulation.unwrap()
            properties = native.dof_properties
            limits = np.column_stack((properties['lower'], properties['upper']))
            names = list(native.dof_names)
            arm_names = task.robots[robot_name].controllers['arm_joint_controller'].config.joint_names
            limits = limits[[names.index(name) for name in arm_names]]
            if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
                return self.atomic._failure_or_hold(task, robot_name, spec, 'unwind_target_outside_asset_limits')
            self.states[key] = dict(command=q.copy(), target=target, limits=limits, velocity=0., stable=0, step=None)
        state = self.states[key]
        if state['step'] == task.step_counter:
            return state['action']
        ready = task._gripper_normalized_openness(robot_name) is not None and task._gripper_normalized_openness(robot_name) >= .95
        support = phase_spec.get('plumbers_base_support')
        if support:
            from .plumbers_support import base_support_status
            status = base_support_status(task, support)
            if status.get('failure'):
                return self.atomic._failure_or_hold(task, robot_name, spec, status['failure'])
            ready = ready and status['ready']
        dt = task._contact_physics_dt()
        command = state['command'].copy()
        error = state['target'][-1] - command[-1]
        speed, acceleration, lead = .5, 1., .03
        if ready:
            desired = np.sign(error) * min(speed, math.sqrt(2 * acceleration * abs(error)))
            velocity = state['velocity'] + np.clip(desired - state['velocity'], -acceleration * dt, acceleration * dt)
            delta = float(np.clip(velocity * dt, -abs(error), abs(error)))
            candidate = command[-1] + delta
            if abs(candidate - q[-1]) <= lead:
                command[-1] = candidate
                state['velocity'] = velocity
            else:
                state['velocity'] = 0.
        else:
            state['velocity'] = 0.
        if np.any(command < state['limits'][:, 0] - 1e-6) or np.any(command > state['limits'][:, 1] + 1e-6):
            return self.atomic._failure_or_hold(task, robot_name, spec, 'unwind_command_outside_asset_limits')
        if np.max(abs(command - q)) > .10:
            return self.atomic._failure_or_hold(task, robot_name, spec, 'unwind_tracking_limit')
        state['stable'] = state['stable'] + 1 if ready and np.max(abs(q - state['target'])) < .003 else 0
        if state['stable'] >= 24:
            self.atomic._mark_complete(task=task, robot_name=robot_name, skill_name=spec['name'],
                                       detail=dict(target_q=state['target'].tolist(), measured_q=q.tolist(),
                                                   real_joint_motion=True, empty_gripper=True))
        action = dict(arm_joint_controller=[command.tolist()], gripper_controller=[1.])
        state.update(command=command, step=task.step_counter, action=action)
        self.atomic._remember_arm_command(task, robot_name, command)
        return action
