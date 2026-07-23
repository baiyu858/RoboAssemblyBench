"""Phase-aware classification for runtime collision candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_CONTACT_PHASE_TOKENS = (
    "descend",
    "grasp",
    "close_gripper",
    "insert",
    "release",
    "settle",
)

_END_EFFECTOR_TOKEN = "Gripper/Robotiq_2F_85/"


@dataclass(frozen=True)
class ContactContext:
    """Assembly state needed to decide whether contact is intentional."""

    phase: str | None
    robot: str | None
    object_name: str | None
    contact_intended: bool


@dataclass(frozen=True)
class ContactDecision:
    """Classification attached to one detector candidate."""

    classification: str
    reason: str
    phase: str | None
    active_robot: str | None
    active_object: str | None


class AssemblyContactPolicy:
    """Separate expected assembly contact, proximity, and collision."""

    def __init__(self, *, collision_threshold: float = 0.0, ignore_pairs: Iterable[Any] = ()):
        self.collision_threshold = float(collision_threshold)
        self.ignore_pairs = tuple(ignore_pairs)

    def classify(self, event, task) -> ContactDecision:
        context = self.context_from_task(task)
        if any(rule.matches(event.entity_a, event.entity_b) for rule in self.ignore_pairs):
            return self._decision("allowed_contact", "configured_ignore_pair", context)
        if self._is_expected_assembly_contact(event, context):
            return self._decision("allowed_contact", "phase_target_end_effector_contact", context)
        if float(event.distance) > self.collision_threshold:
            return self._decision("proximity", "positive_surface_clearance", context)
        return self._decision("collision", "capsule_surface_overlap", context)

    @staticmethod
    def context_from_task(task) -> ContactContext:
        phase = getattr(task, "phase", None)
        phase_spec = {}
        getter = getattr(task, "get_current_phase_spec", None)
        if callable(getter):
            phase_spec = getter() or {}
        if not isinstance(phase_spec, dict):
            phase_spec = {}

        local_skill = phase_spec.get("local_skill")
        if not isinstance(local_skill, dict):
            local_skill = {}
        robot = local_skill.get("robot")
        object_name = local_skill.get("object")

        attach = phase_spec.get("attach")
        attach_items = attach if isinstance(attach, list) else [attach]
        for item in attach_items:
            if isinstance(item, dict):
                robot = robot or item.get("robot")
                object_name = object_name or item.get("object")

        lock = phase_spec.get("lock")
        lock_items = lock if isinstance(lock, list) else [lock]
        for item in lock_items:
            if isinstance(item, dict) and item.get("object"):
                object_name = object_name or item["object"]
                break

        phase_text = str(phase or phase_spec.get("name") or "")
        if robot is None:
            if phase_text.startswith("left_"):
                robot = "franka_left"
            elif phase_text.startswith("right_"):
                robot = "franka_right"

        contact_intended = any(token in phase_text.lower() for token in _CONTACT_PHASE_TOKENS)
        return ContactContext(
            phase=phase_text or None,
            robot=None if robot is None else str(robot),
            object_name=None if object_name is None else str(object_name),
            contact_intended=contact_intended,
        )

    @staticmethod
    def _is_expected_assembly_contact(event, context: ContactContext) -> bool:
        if not context.contact_intended or not context.robot or not context.object_name:
            return False
        entities = (str(event.entity_a), str(event.entity_b))
        robot_entity = next((item for item in entities if item.startswith(f"{context.robot}/")), None)
        object_entity = next((item for item in entities if item == context.object_name), None)
        if robot_entity is None or object_entity is None:
            return False
        return _END_EFFECTOR_TOKEN in robot_entity

    @staticmethod
    def _decision(
        classification: str,
        reason: str,
        context: ContactContext,
    ) -> ContactDecision:
        return ContactDecision(
            classification=classification,
            reason=reason,
            phase=context.phase,
            active_robot=context.robot,
            active_object=context.object_name,
        )
