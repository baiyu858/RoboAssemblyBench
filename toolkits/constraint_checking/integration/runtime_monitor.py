"""Fail-open runtime collision monitoring for RoboAssemblyBench episodes."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .contact_policy import AssemblyContactPolicy, ContactDecision
from .models import RobotCollisionModel, get_robot_collision_model


def _load_collision_detector_cls():
    from toolkits.constraint_checking.detector.collision import CollisionDetector

    return CollisionDetector


def _load_xform_reader_cls():
    try:
        from isaacsim.core.prims import SingleXFormPrim
    except ImportError:
        from omni.isaac.core.prims import SingleXFormPrim

    return SingleXFormPrim


def _prim_path_exists(prim_path: str) -> bool:
    """Check a stage prim without letting an XForm wrapper create it."""

    try:
        from isaacsim.core.utils.prims import get_prim_at_path
    except ImportError:
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
        except ImportError:
            return True
    prim = get_prim_at_path(str(prim_path))
    return bool(prim and prim.IsValid())


def _load_static_scene_boxes(prim_path: str, object_name: str) -> list[dict]:
    """Read world AABBs for boundable leaves under one static scene object."""

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(str(prim_path))
    if not root or not root.IsValid():
        return []
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    boxes = []
    for index, prim in enumerate(Usd.PrimRange(root)):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = np.asarray(world_range.GetMin(), dtype=float)
        maximum = np.asarray(world_range.GetMax(), dtype=float)
        size = maximum - minimum
        if not np.all(np.isfinite(size)) or np.any(size <= 1e-5):
            continue
        boxes.append(
            {
                'name': f'static:{object_name}:{index}',
                'center': (minimum + maximum) * 0.5,
                'half_extents': np.maximum(size * 0.5, 1e-4),
            }
        )
    return boxes


@dataclass(frozen=True)
class PairFilter:
    """Symmetric substring filter for intentionally allowed contact pairs."""

    entity_a_contains: str
    entity_b_contains: str

    @classmethod
    def parse(cls, value: str) -> 'PairFilter':
        left, separator, right = str(value).partition(':')
        if not separator or not left.strip() or not right.strip():
            raise ValueError(f'Invalid constraint ignore pair {value!r}; expected A:B.')
        return cls(left.strip(), right.strip())

    def matches(self, entity_a: str, entity_b: str) -> bool:
        a = str(entity_a)
        b = str(entity_b)
        return (self.entity_a_contains in a and self.entity_b_contains in b) or (
            self.entity_a_contains in b and self.entity_b_contains in a
        )


@dataclass
class RuntimeConstraintConfig:
    """Configuration for passive runtime collision monitoring."""

    robot_model: str = 'ur5e_robotiq_2f85'
    threshold: Optional[float] = None
    check_stride: int = 8
    include_tracked_objects_as_boxes: bool = True
    include_static_scene_objects: bool = True
    include_ground: bool = False
    ground_z: float = 0.0
    ignore_pairs: List[PairFilter] = field(default_factory=list)
    collision_threshold: float = 0.0
    max_recorded_events: int = 5000

    def __post_init__(self) -> None:
        self.check_stride = max(int(self.check_stride), 1)
        if self.threshold is not None:
            self.threshold = float(self.threshold)
        self.collision_threshold = float(self.collision_threshold)
        self.max_recorded_events = max(int(self.max_recorded_events), 0)


class RuntimeConstraintMonitor:
    """Observe one episode without changing actions or task terminal state."""

    def __init__(self, config: RuntimeConstraintConfig | None = None):
        self.config = config or RuntimeConstraintConfig()
        self.robot_model: RobotCollisionModel = get_robot_collision_model(self.config.robot_model)
        threshold = self.config.threshold
        if threshold is None:
            threshold = self.robot_model.default_threshold

        self.detector = None
        self._registered = False
        self._robot_roots: Dict[str, str] = {}
        self._registered_links: Dict[str, List[str]] = {}
        self._missing_prims: List[dict] = []
        self._missing_object_geometry: set[str] = set()
        self._missing_static_geometry: set[str] = set()
        self._monitor_errors: List[dict] = []
        self._observations = 0
        self._checks = 0
        self._candidate_total = 0
        self._violation_total = 0
        self._violations_by_kind: Counter[str] = Counter()
        self._classifications: Counter[str] = Counter()
        self._minimum_distance: float | None = None
        self._candidate_minimum_distance: float | None = None
        self._events: List[dict] = []
        self._proximity_events: List[dict] = []
        self._allowed_contacts: List[dict] = []
        self._events_dropped = 0
        self._object_collision_sizes: Dict[str, np.ndarray] | None = None
        self._static_scene_boxes: List[dict] | None = None
        self._total_check_seconds = 0.0
        self._max_check_seconds = 0.0
        self._contact_policy = AssemblyContactPolicy(
            collision_threshold=self.config.collision_threshold,
            ignore_pairs=self.config.ignore_pairs,
        )

        try:
            detector_cls = _load_collision_detector_cls()
            self.detector = detector_cls(threshold=float(threshold))
            self.detector.franka_links = list(self.robot_model.link_names)
            self.detector.franka_capsules = self.robot_model.detector_capsules
        except Exception as exc:
            self._record_error('initialize', exc)

    def observe(self, task) -> dict:
        """Sample one task step and accumulate a serializable episode report."""

        step = int(getattr(task, 'step_counter', 0))
        self._observations += 1
        if step % self.config.check_stride != 0:
            return {'checked': False, 'step': step, 'reason': 'stride_skip', 'violations': []}

        self._checks += 1
        check_started = time.perf_counter()
        if self.detector is None:
            self._record_check_duration(check_started)
            return {
                'checked': True,
                'step': step,
                'violations': [],
                'monitor_error': self._monitor_errors[-1] if self._monitor_errors else None,
            }

        try:
            if not self._registered:
                self.register_task_robots(task)
            self.refresh_environment_from_task(task)
            candidates = self.detector.check_all(step=step)
            classified = [
                self._event_to_dict(event, self._contact_policy.classify(event, task)) for event in candidates
            ]
            collisions = [event for event in classified if event['classification'] == 'collision']
            self._accumulate(classified)
            return {
                'checked': True,
                'step': step,
                'summary': f'{len(collisions)} collisions from {len(classified)} candidates',
                'candidate_count': len(classified),
                'proximity_count': sum(item['classification'] == 'proximity' for item in classified),
                'allowed_contact_count': sum(item['classification'] == 'allowed_contact' for item in classified),
                'violations': collisions,
            }
        except Exception as exc:
            error = self._record_error('observe', exc, step=step)
            return {'checked': True, 'step': step, 'violations': [], 'monitor_error': error}
        finally:
            self._record_check_duration(check_started)

    def finalize(self) -> dict:
        """Return the accumulated report for this monitor's single episode."""

        return {
            'enabled': True,
            'robot_model': self.robot_model.name,
            'threshold': float(
                self.config.threshold if self.config.threshold is not None else self.robot_model.default_threshold
            ),
            'collision_threshold': float(self.config.collision_threshold),
            'check_stride': int(self.config.check_stride),
            'include_ground': bool(self.config.include_ground),
            'include_static_scene_objects': bool(self.config.include_static_scene_objects),
            'observed_steps': int(self._observations),
            'checks': int(self._checks),
            'total_check_seconds': float(self._total_check_seconds),
            'average_check_seconds': (float(self._total_check_seconds / self._checks) if self._checks else 0.0),
            'max_check_seconds': float(self._max_check_seconds),
            'candidate_total': int(self._candidate_total),
            'classifications': dict(sorted(self._classifications.items())),
            'violation_total': int(self._violation_total),
            'violations_by_kind': dict(sorted(self._violations_by_kind.items())),
            'minimum_distance': self._minimum_distance,
            'candidate_minimum_distance': self._candidate_minimum_distance,
            'events': list(self._events),
            'proximity_events': list(self._proximity_events),
            'allowed_contacts': list(self._allowed_contacts),
            'events_dropped': int(self._events_dropped),
            'registered_robots': dict(self._robot_roots),
            'registered_links': {name: list(links) for name, links in self._registered_links.items()},
            'missing_prims': list(self._missing_prims),
            'missing_object_geometry': sorted(self._missing_object_geometry),
            'registered_static_obstacles': (
                [] if self._static_scene_boxes is None else [item['name'] for item in self._static_scene_boxes]
            ),
            'missing_static_geometry': sorted(self._missing_static_geometry),
            'monitor_error': list(self._monitor_errors),
        }

    def register_task_robots(self, task) -> None:
        """Register all available robot links and retain missing-prim diagnostics."""

        xform_cls = _load_xform_reader_cls()
        robots = getattr(task, 'robots', {}) or {}
        for robot_name, robot in robots.items():
            robot_name = str(robot_name)
            root_prim_path = self._robot_root_prim_path(robot)
            if not root_prim_path:
                self._missing_prims.append(
                    {'robot': robot_name, 'link': None, 'candidate_paths': [], 'error': 'missing_robot_prim_path'}
                )
                continue

            link_xforms = {}
            for link_name in self.robot_model.link_names:
                candidate_paths = self.robot_model.prim_paths_for_link(root_prim_path, link_name)
                last_error = None
                for prim_path in candidate_paths:
                    if not _prim_path_exists(prim_path):
                        last_error = 'prim_unavailable'
                        continue
                    try:
                        try:
                            reader = xform_cls(prim_path=prim_path)
                        except TypeError:
                            reader = xform_cls(prim_path)
                        reader.get_world_pose()
                        link_xforms[link_name] = reader
                        break
                    except Exception as exc:
                        last_error = f'{type(exc).__name__}: {exc}'
                if link_name not in link_xforms:
                    self._missing_prims.append(
                        {
                            'robot': robot_name,
                            'link': link_name,
                            'candidate_paths': list(candidate_paths),
                            'error': last_error or 'prim_unavailable',
                        }
                    )

            self._robot_roots[robot_name] = str(root_prim_path)
            self._registered_links[robot_name] = sorted(link_xforms)
            if link_xforms:
                self.detector.add_agent(robot_name, link_xforms)
        self._registered = True

    def refresh_environment_from_task(self, task) -> None:
        """Refresh box obstacles from ungrasped tracked objects."""

        if hasattr(self.detector, '_boxes'):
            self.detector._boxes.clear()
        if hasattr(self.detector, '_ground_z'):
            self.detector._ground_z = None

        if self.config.include_ground:
            self.detector.add_ground(float(self.config.ground_z))
        self._refresh_static_scene_objects(task)
        if not self.config.include_tracked_objects_as_boxes:
            return

        get_states = getattr(task, 'get_tracked_object_states', None)
        if not callable(get_states):
            return

        for object_name, state in (get_states() or {}).items():
            if not isinstance(state, dict):
                continue
            if state.get('attached_to') is not None or state.get('grasped_by') is not None:
                continue
            if state.get('collision_enabled') is False:
                continue
            position = state.get('position')
            size = state.get('collision_size', state.get('size', state.get('dimensions')))
            if size is None:
                size = self._collision_size_from_task(task, str(object_name))
            if position is None or size is None:
                if size is None:
                    self._missing_object_geometry.add(str(object_name))
                continue
            half_extents = np.maximum(np.asarray(size, dtype=float) * 0.5, 1e-4)
            self.detector.add_box(
                str(object_name),
                center=np.asarray(position, dtype=float),
                half_extents=half_extents,
                orient=state.get('orientation'),
            )

    def _refresh_static_scene_objects(self, task) -> None:
        if not self.config.include_static_scene_objects:
            return
        if self._static_scene_boxes is None:
            self._static_scene_boxes = []
            tracked_names = set((getattr(task, 'get_tracked_object_states', lambda: {})() or {}))
            for object_name, scene_object in (getattr(task, 'objects', {}) or {}).items():
                object_name = str(object_name)
                if object_name in tracked_names:
                    continue
                config = getattr(scene_object, 'config', None)
                if config is None or getattr(config, 'collider', False) is False:
                    continue
                prim_path = getattr(config, 'prim_path', None)
                if not prim_path:
                    self._missing_static_geometry.add(object_name)
                    continue
                try:
                    boxes = _load_static_scene_boxes(str(prim_path), object_name)
                except Exception as exc:
                    self._record_error('static_scene_geometry', exc)
                    boxes = []
                if not boxes:
                    self._missing_static_geometry.add(object_name)
                self._static_scene_boxes.extend(boxes)

        for item in self._static_scene_boxes:
            self.detector.add_box(
                item['name'],
                center=item['center'],
                half_extents=item['half_extents'],
            )

    def _accumulate(self, events: List[dict]) -> None:
        self._candidate_total += len(events)
        collisions = [event for event in events if event['classification'] == 'collision']
        proximity = [event for event in events if event['classification'] == 'proximity']
        allowed = [event for event in events if event['classification'] == 'allowed_contact']
        self._violation_total += len(collisions)
        remaining = max(self.config.max_recorded_events - len(self._events), 0)
        if remaining:
            self._events.extend(collisions[:remaining])
        self._events_dropped += max(len(collisions) - remaining, 0)
        audit_limit = min(self.config.max_recorded_events, 500)
        proximity_remaining = max(audit_limit - len(self._proximity_events), 0)
        allowed_remaining = max(audit_limit - len(self._allowed_contacts), 0)
        self._proximity_events.extend(proximity[:proximity_remaining])
        self._allowed_contacts.extend(allowed[:allowed_remaining])
        for event in events:
            self._classifications[str(event['classification'])] += 1
            distance = float(event['distance'])
            if self._candidate_minimum_distance is None or distance < self._candidate_minimum_distance:
                self._candidate_minimum_distance = distance
        for event in collisions:
            self._violations_by_kind[str(event['kind'])] += 1
            distance = float(event['distance'])
            if self._minimum_distance is None or distance < self._minimum_distance:
                self._minimum_distance = distance

    def _collision_size_from_task(self, task, object_name: str) -> np.ndarray | None:
        if self._object_collision_sizes is None:
            self._object_collision_sizes = {}
            for phase_spec in getattr(task, 'phase_specs', []) or []:
                local_skill = phase_spec.get('local_skill') if isinstance(phase_spec, dict) else None
                if not isinstance(local_skill, dict):
                    continue
                phase_object = local_skill.get('object')
                if phase_object is None:
                    continue
                size = local_skill.get('contact_box_scale')
                if size is None and local_skill.get('contact_box_half_extents') is not None:
                    size = np.asarray(local_skill['contact_box_half_extents'], dtype=float) * 2.0
                if size is not None:
                    self._object_collision_sizes.setdefault(str(phase_object), np.asarray(size, dtype=float))
        return self._object_collision_sizes.get(str(object_name))

    def _record_error(self, stage: str, exc: Exception, *, step: int | None = None) -> dict:
        error = {
            'stage': str(stage),
            'step': None if step is None else int(step),
            'type': type(exc).__name__,
            'message': str(exc),
        }
        self._monitor_errors.append(error)
        return error

    def _record_check_duration(self, started: float) -> None:
        duration = max(time.perf_counter() - started, 0.0)
        self._total_check_seconds += duration
        self._max_check_seconds = max(self._max_check_seconds, duration)

    @staticmethod
    def _robot_root_prim_path(robot) -> str | None:
        config = getattr(robot, 'config', None)
        for owner in (config, robot):
            if owner is None:
                continue
            value = getattr(owner, 'prim_path', None)
            if value:
                return str(value)
        articulation = getattr(robot, 'articulation', None)
        value = getattr(articulation, 'prim_path', None)
        return None if not value else str(value)

    @staticmethod
    def _event_to_dict(event, decision: ContactDecision | None = None) -> dict:
        if decision is None:
            decision = ContactDecision(
                classification='candidate',
                reason='unclassified',
                phase=None,
                active_robot=None,
                active_object=None,
            )
        return {
            'step': int(event.step),
            'kind': str(event.kind),
            'entity_a': str(event.entity_a),
            'entity_b': str(event.entity_b),
            'distance': float(event.distance),
            'threshold': float(event.threshold),
            'classification': decision.classification,
            'classification_reason': decision.reason,
            'phase': decision.phase,
            'active_robot': decision.active_robot,
            'active_object': decision.active_object,
            'pos_a': None if event.pos_a is None else np.asarray(event.pos_a, dtype=float).tolist(),
            'pos_b': None if event.pos_b is None else np.asarray(event.pos_b, dtype=float).tolist(),
        }
