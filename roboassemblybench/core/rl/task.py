"""Object-independent insertion identity and phase handover contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class InsertionTask:
    task_family: str
    stage_id: str
    moving_object: str
    socket_object: str
    insertion_robot: str
    entry_target: str
    goal_target: str
    support_robot: str | None = None
    support_object: str | None = None
    socket_is_static: bool = False
    contact_objects: tuple[str, ...] = ()
    insertion_axis: tuple[float, float, float] | None = None  # socket-local frame
    entry_phase: str | None = None
    replace_phases: tuple[str, ...] = ()
    rl_phase: str | None = None
    settle_phase: str | None = None
    support_retreat_phase: str | None = None
    readiness_conditions: tuple[dict, ...] = ()
    # Fixed CAD insertion point (metres): geometric error and virtual TCP use
    # the same point so compliant rotation cannot create a false lateral error.
    control_point_in_object: tuple[float, float, float] | None = None
    allow_seated_load_transfer: bool = False

    def __post_init__(self):
        for field in ('task_family', 'stage_id', 'moving_object', 'socket_object',
                      'insertion_robot', 'entry_target', 'goal_target'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f'{field} must be a nonempty string')
        if self.moving_object == self.socket_object:
            raise ValueError('moving_object and socket_object must differ')
        for field in ('support_robot', 'support_object', 'entry_phase', 'rl_phase', 'settle_phase', 'support_retreat_phase'):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{field} must be a nonempty string when supplied')
        if bool(self.support_robot) != bool(self.support_object):
            raise ValueError('support_robot and support_object must be supplied together')
        if self.support_robot == self.insertion_robot:
            raise ValueError('Insertion and support must use different robots')
        if self.support_object == self.moving_object:
            raise ValueError('A moving object cannot also be the support object')
        if type(self.socket_is_static) is not bool:
            raise ValueError('socket_is_static must be a boolean')
        if type(self.allow_seated_load_transfer) is not bool:
            raise ValueError('allow_seated_load_transfer must be a boolean')
        if self.socket_is_static and self.socket_object == self.support_object:
            raise ValueError('A static socket cannot also be held by a robot')
        if isinstance(self.contact_objects, str) or isinstance(self.replace_phases, str):
            raise ValueError('contact_objects and replace_phases must be sequences of names, not strings')
        contacts = tuple(self.contact_objects) or (self.socket_object,)
        if (any(not isinstance(x, str) or not x for x in contacts)
                or len(set(contacts)) != len(contacts) or self.moving_object in contacts):
            raise ValueError('contact_objects must contain distinct other objects')
        object.__setattr__(self, 'contact_objects', contacts)
        object.__setattr__(self, 'replace_phases', tuple(self.replace_phases))
        if not isinstance(self.readiness_conditions, (list, tuple)) or any(
                not isinstance(x, dict) for x in self.readiness_conditions):
            raise ValueError('readiness_conditions must be a sequence of advance-condition mappings')
        object.__setattr__(self, 'readiness_conditions', tuple(dict(x) for x in self.readiness_conditions))
        if (any(not isinstance(x, str) or not x for x in self.replace_phases)
                or len(set(self.replace_phases)) != len(self.replace_phases)):
            raise ValueError('replace_phases must contain distinct phase names')
        if self.insertion_axis is not None:
            axis = tuple(float(x) for x in self.insertion_axis)
            if len(axis) != 3 or not all(math.isfinite(x) for x in axis) or sum(x * x for x in axis) < 1e-12:
                raise ValueError('insertion_axis must be a finite nonzero socket-local 3D vector')
            norm = math.sqrt(sum(x * x for x in axis))
            object.__setattr__(self, 'insertion_axis', tuple(x / norm for x in axis))
        if self.control_point_in_object is not None:
            point = tuple(float(x) for x in self.control_point_in_object)
            if len(point) != 3 or not all(math.isfinite(x) for x in point):
                raise ValueError('control_point_in_object must be a finite CAD-local three-vector')
            object.__setattr__(self, 'control_point_in_object', point)

    @property
    def participants(self):
        return ((self.insertion_robot, self.moving_object),) + (
            ((self.support_robot, self.support_object),) if self.support_robot else ())

    @property
    def objects(self):
        return tuple(dict.fromkeys((self.moving_object, self.socket_object, *self.contact_objects,
                                   *((self.support_object,) if self.support_object else ()))))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def policy_contract(self):
        """A policy is bound to bodies/roles/frames, not a runtime phase index."""
        data = self.to_dict()
        for field in ('stage_id', 'entry_phase', 'replace_phases', 'rl_phase',
                      'settle_phase', 'support_retreat_phase', 'readiness_conditions'):
            data.pop(field)
        # JSON-stable types for qualifications and Torch weights_only loading.
        data['contact_objects'] = list(data['contact_objects'])
        if data['insertion_axis'] is not None:
            data['insertion_axis'] = list(data['insertion_axis'])
        if self.control_point_in_object is None:
            data.pop('control_point_in_object')  # Preserve existing wrist-centred checkpoint contracts.
        else:
            data['control_point_in_object'] = list(self.control_point_in_object)
        if not self.allow_seated_load_transfer:
            data.pop('allow_seated_load_transfer')
        return data
