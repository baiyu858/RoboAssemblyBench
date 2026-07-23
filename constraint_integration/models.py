"""Robot collision models used by the isolated constraint bridge.

The existing constraint_detection demo hard-codes Franka link names.  The main
RoboAssemblyBench reproduction task uses UR5e + Robotiq, so this module keeps
robot geometry in data instead of inside the detector implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class CapsuleSpec:
    """A swept capsule approximation between two robot link frames."""

    parent: str
    child: str
    radius: float


@dataclass(frozen=True)
class RobotCollisionModel:
    """Minimal link/capsule model for proximity collision checks."""

    name: str
    link_names: Tuple[str, ...]
    capsules: Tuple[CapsuleSpec, ...]
    default_threshold: float = 0.03
    link_path_overrides: Dict[str, str | Tuple[str, ...]] | None = None

    def prim_paths_for_link(self, root_prim_path: str, link_name: str) -> Tuple[str, ...]:
        """Return candidate USD prim paths for a link, in preference order."""

        root = str(root_prim_path).rstrip("/")
        overrides = self.link_path_overrides or {}
        configured = overrides.get(link_name, link_name)
        relative_paths = (configured,) if isinstance(configured, str) else tuple(configured)
        return tuple(f"{root}/{str(path).strip('/')}" for path in relative_paths)

    def prim_path_for_link(self, root_prim_path: str, link_name: str) -> str:
        """Return the expected USD prim path for *link_name* under a robot root."""

        return self.prim_paths_for_link(root_prim_path, link_name)[0]

    @property
    def detector_capsules(self) -> list[tuple[str, str, float]]:
        """Format expected by constraint_detection.src.collision.CollisionDetector."""

        return [(item.parent, item.child, float(item.radius)) for item in self.capsules]


FRANKA_MODEL = RobotCollisionModel(
    name="franka_panda",
    link_names=(
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    ),
    capsules=(
        CapsuleSpec("panda_link1", "panda_link2", 0.060),
        CapsuleSpec("panda_link2", "panda_link3", 0.060),
        CapsuleSpec("panda_link3", "panda_link4", 0.055),
        CapsuleSpec("panda_link4", "panda_link5", 0.050),
        CapsuleSpec("panda_link5", "panda_link6", 0.050),
        CapsuleSpec("panda_link6", "panda_link7", 0.045),
        CapsuleSpec("panda_link7", "panda_hand", 0.045),
        CapsuleSpec("panda_hand", "panda_leftfinger", 0.022),
        CapsuleSpec("panda_hand", "panda_rightfinger", 0.022),
    ),
    default_threshold=0.03,
)


UR5E_ROBOTIQ_MODEL = RobotCollisionModel(
    name="ur5e_robotiq_2f85",
    link_names=(
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "Gripper/Robotiq_2F_85/base_link",
        "Gripper/Robotiq_2F_85/left_inner_finger",
        "Gripper/Robotiq_2F_85/right_inner_finger",
    ),
    capsules=(
        # Distal radii are narrower than the upper arm. Keeping the first-pass
        # conservative values here created false inter-arm overlaps during
        # close assembly work.
        CapsuleSpec("shoulder_link", "upper_arm_link", 0.070),
        CapsuleSpec("upper_arm_link", "forearm_link", 0.060),
        CapsuleSpec("forearm_link", "wrist_1_link", 0.055),
        CapsuleSpec("wrist_1_link", "wrist_2_link", 0.040),
        CapsuleSpec("wrist_2_link", "wrist_3_link", 0.035),
        CapsuleSpec("wrist_3_link", "Gripper/Robotiq_2F_85/base_link", 0.040),
        CapsuleSpec(
            "Gripper/Robotiq_2F_85/base_link",
            "Gripper/Robotiq_2F_85/left_inner_finger",
            0.020,
        ),
        CapsuleSpec(
            "Gripper/Robotiq_2F_85/base_link",
            "Gripper/Robotiq_2F_85/right_inner_finger",
            0.020,
        ),
    ),
    default_threshold=0.03,
    link_path_overrides={
        "Gripper/Robotiq_2F_85/base_link": (
            "Gripper/Robotiq_2F_85/base_link",
            "wrist_3_link/Gripper/Robotiq_2F_85/base_link",
            "Gripper/base_link",
            "wrist_3_link/Gripper/base_link",
        ),
        "Gripper/Robotiq_2F_85/left_inner_finger": (
            "Gripper/Robotiq_2F_85/left_inner_finger",
            "wrist_3_link/Gripper/Robotiq_2F_85/left_inner_finger",
        ),
        "Gripper/Robotiq_2F_85/right_inner_finger": (
            "Gripper/Robotiq_2F_85/right_inner_finger",
            "wrist_3_link/Gripper/Robotiq_2F_85/right_inner_finger",
        ),
    },
)


MODEL_REGISTRY = {
    FRANKA_MODEL.name: FRANKA_MODEL,
    UR5E_ROBOTIQ_MODEL.name: UR5E_ROBOTIQ_MODEL,
    "ur5e": UR5E_ROBOTIQ_MODEL,
    "UR5eRobot": UR5E_ROBOTIQ_MODEL,
    "franka": FRANKA_MODEL,
    "Franka": FRANKA_MODEL,
}


def get_robot_collision_model(name: str | None) -> RobotCollisionModel:
    """Resolve a robot collision model by name, defaulting to UR5e + Robotiq."""

    if not name:
        return UR5E_ROBOTIQ_MODEL
    return MODEL_REGISTRY.get(str(name), UR5E_ROBOTIQ_MODEL)


def available_models() -> Iterable[str]:
    return tuple(sorted(MODEL_REGISTRY))
