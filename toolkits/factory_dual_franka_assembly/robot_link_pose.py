"""Read existing robot links without constructing or authoring rigid prims."""
import numpy as np

from internutopia.core.util.pose_mixin import PoseMixin
from toolkits.factory_dual_franka_assembly.physics_read_cache import read_physics_frame


def _simulation_view():
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except ImportError:
        from omni.isaac.core.simulation_manager import SimulationManager
    return SimulationManager.get_physics_sim_view()


def _usd_world_pose(path):
    try:
        from isaacsim.core.utils.xforms import get_world_pose
    except ImportError:
        from omni.isaac.core.utils.xforms import get_world_pose

    return get_world_pose(path)


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value, dtype=float)


class _LinkView:
    """Measurement handle only: no initialization or state-setting methods."""

    def __init__(self, prim_path):
        self.prim_path = self.name = prim_path
        self._simulation = self._view = None

    def _current_view(self):
        simulation = _simulation_view()
        if simulation is None:
            self._simulation = self._view = None
            return None
        if simulation is not self._simulation or self._view is None or not self._view.check():
            view = simulation.create_rigid_body_view(self.prim_path)
            if view.count != 1 or list(view.prim_paths) != [self.prim_path] or not view.check():
                raise RuntimeError(f'No unique live rigid body for robot link: {self.prim_path}')
            self._simulation, self._view = simulation, view
        return self._view

    def get_world_pose(self):
        view = self._current_view()
        if view is None:
            # Before PhysX starts, read the authored pose without changing USD.
            return tuple(_array(value).copy() for value in _usd_world_pose(self.prim_path))

        def read():
            pose = _array(view.get_transforms())
            if pose.shape != (1, 7) or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[0, 3:]) < 1e-12:
                raise RuntimeError(f'Invalid physical robot-link pose: {self.prim_path}')
            # Tensor poses use xyzw; the task's pose interface uses wxyz.
            return pose[0, :3].copy(), pose[0, [6, 3, 4, 5]].copy()

        return read_physics_frame(self, ('robot_link_pose', self.prim_path, id(view)), read)

    def get_velocities(self):
        view = self._current_view()
        if view is None:
            raise RuntimeError(f'Robot-link velocity requested before physics initialization: {self.prim_path}')

        def read():
            velocity = _array(view.get_velocities())
            if velocity.shape != (1, 6) or not np.all(np.isfinite(velocity)):
                raise RuntimeError(f'Invalid physical robot-link velocity: {self.prim_path}')
            return velocity[0].copy()

        return read_physics_frame(self, ('robot_link_velocity', self.prim_path, id(view)), read)


class RobotLinkPose(PoseMixin):
    """Keep the task's world/env pose convention using a read-only tensor view.

    Existing articulation links already have their physics APIs and reporting
    configured. Creating SingleRigidPrim here would rewrite transforms and
    sleep policy on a live body, disturbing other held parts in the scene.
    Keep this reader in the task cache, never in the robot's writable body map.
    """

    def __init__(self, prim_path):
        self._link = _LinkView(prim_path)
        self.name = prim_path
        super().__init__()

    def unwrap(self):
        return self._link

    def get_linear_velocity(self):
        return self._link.get_velocities()[:3]

    def get_angular_velocity(self):
        return self._link.get_velocities()[3:]
