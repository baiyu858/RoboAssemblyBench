import os
from typing import Optional

import numpy as np

from internutopia.core.robot.articulation import IArticulation
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.util.physics_status_util import (
    get_rigidbody_status,
    set_rigidbody_status,
)


class IsaacsimRigidBody(IRigidBody):
    """
    Isaacsim's implementation on `IRigidBody` class.
    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): shortname to be used as a key by Scene class.
                              Note: needs to be unique if the object is added to the Scene.
        usd_path (str, optional):  The file path containing the rigid body definition.
        position (Optional[np.ndarray], optional): position in the world frame of the prim. shape is (3, ).
                                                   Defaults to None, which means left unchanged.
        translation (Optional[np.ndarray], optional): translation in the local frame of the prim
                                                      (with respect to its parent prim). shape is (3, ).
                                                      Defaults to None, which means left unchanged.
        orientation (Optional[np.ndarray], optional): quaternion orientation in the world/ local frame of the prim
                                                      (depends if translation or position is specified).
                                                      quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                      Defaults to None, which means left unchanged.
        scale (Optional[np.ndarray], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        mass (Optional[float], optional): mass in kg. Defaults to None.
        linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
        angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.
        owner_articulation (Optional[IArticulation], optional): The articulation to which the rigid body belongs if rigid body represents a link. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        mass: Optional[float] = None,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
        owner_articulation: Optional[IArticulation] = None,
    ):
        from isaacsim.core.utils.prims import get_prim_at_path
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
        except ImportError:
            from omni.isaac.core.utils.stage import add_reference_to_stage

        self._is_link = False
        if owner_articulation is not None:
            self._is_link = True

        prim = get_prim_at_path(prim_path)
        if prim.IsValid():
            if usd_path is not None:
                raise ValueError(f"Prim {prim_path} already exist, 'usd_path' should be None.")
        else:
            if usd_path is None:
                raise ValueError(f"Prim {prim_path} not exist, 'usd_path' is required.")
            add_reference_to_stage(prim_path=prim_path, usd_path=os.path.abspath(usd_path))

        self._param = {
            'prim_path': prim_path,
            'name': name,
            'position': position,
            'translation': translation,
            'orientation': orientation,
            'scale': scale,
            'visible': visible,
            'mass': mass,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
        }

        self._rigid_prim = self._create_rigid_prim(**self._param)
        super().__init__()
        self._rigid_prim_view = self._rigid_prim._rigid_prim_view
        self.name = self._rigid_prim.name
        self.status = {}

    @staticmethod
    def _create_rigid_prim(**params):
        """Wrap a body without changing its configured sleep policy.

        SingleRigidPrim prepares contact reports by default, even for a pose
        lookup. In Isaac Sim 5.1 this authors sleepThreshold=0. Preserve the
        prior opinion (including an intentional zero), or restore the schema
        default when there was no opinion. Force-tracking probes are created
        separately and retain their contact-report preparation.
        """
        from isaacsim.core.utils.prims import get_prim_at_path
        try:
            from isaacsim.core.prims import SingleRigidPrim as RigidPrim
        except ImportError:
            from omni.isaac.core.prims import RigidPrim

        prim = get_prim_at_path(params['prim_path'])
        sleep_attr = prim.GetAttribute('physxRigidBody:sleepThreshold')
        authored = bool(sleep_attr and sleep_attr.HasAuthoredValueOpinion())
        threshold = sleep_attr.Get() if authored else None
        rigid_prim = RigidPrim(**params)
        if authored:
            rigid_prim.set_sleep_threshold(float(threshold))
        else:
            # Remove only the zero authored by the wrapper, preserving USD's
            # default rather than choosing a new threshold for the caller.
            sleep_attr = prim.GetAttribute('physxRigidBody:sleepThreshold')
            if sleep_attr:
                sleep_attr.Clear()
        return rigid_prim

    def set_linear_velocity(self, velocity: np.ndarray):
        """See `IRigidBody.set_linear_velocity` for documentation."""

        if self._is_link:
            raise TypeError('Cannot set linear velocity on a link')

        self._rigid_prim.set_linear_velocity(velocity=velocity)

    def get_linear_velocity(self) -> np.ndarray:
        """See `IRigidBody.get_linear_velocity` for documentation."""

        return self._rigid_prim.get_linear_velocity()

    def get_angular_velocity(self):
        """See `IRigidBody.get_angular_velocity` for documentation."""

        return self._rigid_prim.get_angular_velocity()

    def set_mass(self, mass: float) -> None:
        """See `IRigidBody.set_mass` for documentation."""

        self._rigid_prim.set_mass(mass=mass)

    def get_mass(self) -> float:
        """See `IRigidBody.get_mass` for documentation."""

        return self._rigid_prim.get_mass()

    def unwrap(self) -> any:
        return self._rigid_prim

    def save_status(self):
        self.status = get_rigidbody_status(self._rigid_prim)

    def restore_status(self):
        self._rigid_prim = self._create_rigid_prim(**self._param)
        self._rigid_prim_view = self._rigid_prim._rigid_prim_view
        if not self.status:
            self.save_status()
            return
        set_rigidbody_status(self._rigid_prim, self.status)

    def post_reset(self):
        return self._rigid_prim.post_reset()

    def is_valid(self) -> bool:
        return self._rigid_prim.is_valid()
