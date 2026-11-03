"""Access an existing static scene object without adding rigid-body schemas."""
import numpy as np

from internutopia.core.util.pose_mixin import PoseMixin


class StaticObjectPose(PoseMixin):
    def __init__(self, scene_object):
        self._object = scene_object
        self.name = scene_object.name
        super().__init__()

    def unwrap(self):
        return self._object

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)
