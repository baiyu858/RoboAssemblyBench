"""7-DoF EE delta-action -> Franka joint commands via RMPFlow IK.

Usage per VLA prediction:
    ctrl.set_target(action7)        # latch target once
    for _ in range(sub_steps):
        ctrl.step()                 # RMPFlow drives toward target
        world.step()
"""
import numpy as np

try:
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController,
    )
except ImportError:
    from omni.isaac.franka.controllers import RMPFlowController

try:
    from isaacsim.core.prims import SingleXFormPrim as _XFormPrim
except ImportError:
    try:
        from isaacsim.core.prims import XFormPrim as _XFormPrim
    except ImportError:
        from omni.isaac.core.prims import XFormPrim as _XFormPrim


class FrankaEEController:
    """OpenVLA Bridge action: [dx,dy,dz, droll,dpitch,dyaw, gripper] in meters/rad."""

    def __init__(
        self,
        franka,
        ee_prim_path: str = '/World/Franka/panda_hand',
        pos_scale: float = 1.0,
        max_step: float = 0.05,
        precheck=None,
    ):
        self.franka = franka
        self.pos_scale = pos_scale
        self.max_step = max_step  # safety clamp per VLA step (meters)
        self.precheck = precheck  # optional TrajectoryPrechecker
        self.rmpflow = RMPFlowController(
            name='franka_rmpflow',
            robot_articulation=franka,
        )
        self.ee_prim = _XFormPrim(ee_prim_path)
        self._target_pos = None
        self._target_orn = None
        self._gripper_open = True
        self.last_precheck = None

    def reset(self):
        self.rmpflow.reset()
        self._target_pos = None
        self._target_orn = None
        self.last_precheck = None

    def set_target(self, action7, step: int = 0):
        """Latch a new EE target from one VLA delta action.

        If a TrajectoryPrechecker is configured, the whole path from current
        EE pose to the new target is validated (IK reachability + collision)
        before latching. If infeasible, the target is rejected and the robot
        keeps its position.

        Returns True if the target was accepted, False if rejected.
        """
        a = np.asarray(action7, dtype=np.float32)
        delta = a[:3] * self.pos_scale
        # Safety: clamp per-step displacement
        n = float(np.linalg.norm(delta))
        if n > self.max_step:
            delta = delta * (self.max_step / n)

        cur_pos, cur_orn = self.ee_prim.get_world_pose()
        cur_pos = np.asarray(cur_pos, dtype=np.float32)
        cur_orn = np.asarray(cur_orn, dtype=np.float32)
        target_pos = cur_pos + delta

        if self.precheck is not None:
            report = self.precheck.check_trajectory(target_pos, cur_orn, step=step)
            self.last_precheck = report
            if not report.feasible:
                print(f'[controller] precheck rejected: {report}')
                return False

        self._target_pos = target_pos
        self._target_orn = cur_orn  # keep current orientation
        # LIBERO convention: gripper ~0 = open, ~1 = close (opposite of Bridge)
        self._gripper_open = float(a[6]) < 0.5
        return True

    def step(self):
        """One physics-step IK update toward the latched target."""
        if self._target_pos is None:
            return
        art_action = self.rmpflow.forward(
            target_end_effector_position=self._target_pos,
            target_end_effector_orientation=self._target_orn,
        )
        self.franka.apply_action(art_action)
        if self._gripper_open:
            self.franka.gripper.open()
        else:
            self.franka.gripper.close()
