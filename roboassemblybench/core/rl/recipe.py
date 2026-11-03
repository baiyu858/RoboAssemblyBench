"""Compile any declared insertion window into a contact-RL local skill."""
from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import asdict
from pathlib import Path

from roboassemblybench.core.paths import BENCHMARK_ROOT
from .control import InsertionConfig, validate_torque_soft_limit, validate_torque_hard_limit
from .legacy import BACKEND as LEGACY_BACKEND
from .task import InsertionTask

BACKEND = 'native_contact_insertion'


def precision_insertion_settings(spec):
    entries = spec.get('precision_insertion', {})
    if not isinstance(entries, dict):
        raise ValueError('precision_insertion must be a mapping keyed by insertion stage/part id')
    active = {}
    allowed = {'controller', 'backend', 'checkpoint', 'checkpoint_sha256', 'entry_waypoint',
               'timeout_steps', 'binding', 'controller_config', 'retreat_control',
               'released_rotation_tolerance', 'released_axial_rotation', 'qualification_config',
               'torque_soft_limit', 'torque_hard_limit', 'axis_alignment'}
    for key, entry in entries.items():
        label = f'precision_insertion[{key!r}]'
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError(f'{label} must be a mapping with only {sorted(allowed)}')
        if entry.get('controller') not in ('planned', 'residual_rl'):
            raise ValueError(f'{label}.controller must be planned or residual_rl')
        if entry['controller'] == 'planned':
            continue
        if entry.get('backend', BACKEND) not in (BACKEND, LEGACY_BACKEND):
            raise ValueError(f'{label}: unsupported insertion RL backend')
        if spec.get('continuous_physics') is not True:
            raise ValueError(f'{label} requires continuous physical execution')
        checkpoint = entry.get('checkpoint')
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f'{label}.checkpoint must name an explicit policy file')
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = BENCHMARK_ROOT.parent / path
        waypoint, timeout = entry.get('entry_waypoint', 1), entry.get('timeout_steps', 6000)
        if type(waypoint) is not int or waypoint < 1:
            raise ValueError(f'{label}.entry_waypoint must be a positive integer')
        if type(timeout) is not int or timeout < 1:
            raise ValueError(f'{label}.timeout_steps must be a positive integer')
        digest = entry.get('checkpoint_sha256')
        if digest is not None and (not isinstance(digest, str) or re.fullmatch('[0-9a-fA-F]{64}', digest) is None):
            raise ValueError(f'{label}.checkpoint_sha256 must contain 64 hexadecimal characters')
        if str(key) in active:
            raise ValueError(f'Duplicate precision insertion stage {key}')
        config = entry.get('controller_config', {})
        if not isinstance(config, dict):
            raise ValueError('controller_config must be a mapping')
        config = asdict(InsertionConfig(**config))
        torque_soft = validate_torque_soft_limit(entry.get('torque_soft_limit'), InsertionConfig(**config))
        from .axis_alignment import alignment_settings
        alignment = alignment_settings(entry.get('axis_alignment'))
        torque_hard = entry.get('torque_hard_limit')
        validate_torque_hard_limit(torque_hard, InsertionConfig(**config), torque_soft)
        from .validate import QualificationConfig
        qualification = asdict(QualificationConfig(**entry.get('qualification_config', {})))
        released_rotation = entry.get('released_rotation_tolerance', config['seated_rotation_tolerance'])
        if (isinstance(released_rotation, bool) or not isinstance(released_rotation, (int, float))
                or not math.isfinite(released_rotation) or not 0 < released_rotation <= math.pi):
            raise ValueError(f'{label}.released_rotation_tolerance must be in (0, pi] radians')
        released_axial = entry.get('released_axial_rotation')
        if released_axial is not None:
            if not isinstance(released_axial, dict) or set(released_axial) != {'axis', 'twist_tolerance_rad'}:
                raise ValueError(f'{label}.released_axial_rotation requires axis and twist_tolerance_rad')
            spin = released_axial['twist_tolerance_rad']
            if (isinstance(spin, bool) or not isinstance(spin, (int, float))
                    or not math.isfinite(spin) or not 0 < spin <= math.pi):
                raise ValueError(f'{label}.released_axial_rotation.twist_tolerance_rad must be in (0, pi]')
            from toolkits.factory_dual_franka_assembly.planner_primitives import axis_rotation_error
            axis_rotation_error([1., 0., 0., 0.], [1., 0., 0., 0.], released_axial['axis'])
        if not isinstance(entry.get('binding', {}), dict) or not isinstance(entry.get('retreat_control', {}), dict):
            raise ValueError('binding and retreat_control must be mappings')
        active[str(key)] = dict(backend=BACKEND, checkpoint=str(path.resolve()),
                                checkpoint_sha256=digest.lower() if digest else None, entry_waypoint=waypoint,
                                timeout_steps=timeout, binding=copy.deepcopy(entry.get('binding', {})),
                                controller_config=config, retreat_control=copy.deepcopy(entry.get('retreat_control', {})),
                                torque_soft_limit=torque_soft,
                                torque_hard_limit=torque_hard, axis_alignment=alignment,
                                released_axial_rotation=copy.deepcopy(released_axial),
                                released_rotation_tolerance=float(released_rotation), qualification_config=qualification)
    return active


def infer_binding(payload, stage_id, entry_waypoint=1, overrides=None):
    """Fabrica naming is an authoring adapter, not a runtime constraint.

    Custom benchmarks supply an explicit binding, including replace_phases.
    """
    explicit = overrides or {}
    complete = {'task_family', 'moving_object', 'socket_object', 'insertion_robot',
                'entry_target', 'goal_target', 'replace_phases'} <= set(explicit)
    if complete:
        values = {}
    else:
        from .backends.fabrica import binding_defaults
        values = binding_defaults(payload, stage_id, entry_waypoint)
    values.update(overrides or {})
    values.setdefault('stage_id', str(stage_id))
    binding = InsertionTask.from_dict(values)
    targets = {t['name'] for t in payload.get('targets', [])}
    objects = {o['name'] for o in payload.get('objects', [])}
    robots = {r['name'] for r in payload.get('robots', [])}
    if (not {binding.entry_target, binding.goal_target} <= targets
            or not set(binding.objects) <= objects or not {r for r, _ in binding.participants} <= robots):
        raise ValueError(f'{stage_id}: binding references missing objects, robots or targets')
    return binding


def _tighten(entry, limits):
    for key, value in limits.items():
        previous = entry.get(key)
        entry[key] = value if previous is None else min(float(previous), value)
    entry.update(require_released=True, require_static=True)


def replace_insertion(phases, success, binding, settings):
    phases, success = copy.deepcopy(phases), copy.deepcopy(success)
    if not binding.replace_phases or not binding.rl_phase:
        raise ValueError('Binding requires replace_phases and rl_phase')
    names = [p['name'] for p in phases]
    if any(n not in names for n in binding.replace_phases):
        raise ValueError(f'{binding.stage_id}: insertion phase window missing')
    indices = [names.index(n) for n in binding.replace_phases]
    if indices != list(range(min(indices), max(indices) + 1)):
        raise ValueError('Insertion replacement must be a contiguous ordered phase window')
    if binding.settle_phase not in names:
        raise ValueError('Insertion must retain a post-insertion stability verification phase')
    first, last = min(indices), max(indices)
    for item in phases[first:last + 1]:
        if any(item.get(key) for key in ('attach', 'detach', 'lock', 'fixture_lock', 'freeze_current_pose')):
            raise ValueError('RL window must follow grasp creation and precede release/locking interactions')
    phase = phases[first]
    phase.update(name=binding.rl_phase, timeout_steps=settings['timeout_steps'])
    phase.pop('local_skill', None)
    phase.pop('skill', None)
    config = InsertionConfig(**settings.get('controller_config', {}))
    validate_torque_hard_limit(settings.get('torque_hard_limit'), config, settings.get('torque_soft_limit'))
    common = dict(name='contact_insert_rl', backend=BACKEND, checkpoint=settings['checkpoint'],
                  checkpoint_sha256=settings.get('checkpoint_sha256'), binding=binding.to_dict(),
                  controller_config=asdict(config), require_success=True, fallback='fail',
                  torque_soft_limit=validate_torque_soft_limit(settings.get('torque_soft_limit'), config),
                  torque_hard_limit=settings.get('torque_hard_limit'), axis_alignment=copy.deepcopy(settings.get('axis_alignment')),
                  requires_held_object=True, continuous_physics=True, observation_space=55, action_space=6)
    phase['local_skills'] = {r: dict(common, robot=r, held_object=o, object=o) for r, o in binding.participants}
    conditions = [dict(type='local_skill_complete', robot=r, skill='contact_insert_rl') for r, _ in binding.participants]
    conditions.extend(copy.deepcopy(binding.readiness_conditions))
    phase['advance'] = dict(type='all_of', conditions=conditions)
    phase.setdefault('gripper_commands', {}).update({r: 'contact_hold' for r, _ in binding.participants})
    phases = phases[:first] + [phase] + phases[last + 1:]
    limits = {name: dict(position_tolerance=config.base_position_limit,
              orientation_tolerance=config.base_rotation_limit, linear_velocity_threshold=.01, angular_velocity_threshold=.1)
              for name in (binding.socket_object, binding.support_object) if name}
    limits[binding.moving_object] = dict(position_tolerance=math.hypot(config.seated_axial_tolerance, config.seated_lateral_tolerance),
                                         orientation_tolerance=settings.get('released_rotation_tolerance', config.seated_rotation_tolerance),
                                         linear_velocity_threshold=.01, angular_velocity_threshold=.1)
    for entry in success:
        if entry.get('object') in limits:
            _tighten(entry, limits[entry['object']])
    for item in phases:
        if item['name'] == binding.settle_phase:
            for entry in item.get('advance', {}).get('objects', []):
                if isinstance(entry, dict) and entry.get('object') in limits:
                    retained_support = (item.get('physical_supported_verification')
                                        and entry.get('object') == binding.support_object
                                        and entry.get('require_released') is False)
                    _tighten(entry, limits[entry['object']])
                    if retained_support:
                        entry['require_released'] = False
        if item['name'] == binding.support_retreat_phase and settings.get('retreat_control'):
            item['local_skill'].update(settings['retreat_control'], object=binding.support_object, requires_held_object=False)
    axial = settings.get('released_axial_rotation')
    if axial is not None:
        # Only released-object checks after this insertion use swing/twist.
        # Held seating, earlier setup checks and all other parts keep their gates.
        released_entries = list(success)
        for item in phases[first + 1:]:
            if item.get('advance', {}).get('type') in ('physical_settle', 'plumbers_supported_settle'):
                released_entries.extend(item['advance'].get('objects', []))
        for entry in released_entries:
            if isinstance(entry, dict) and entry.get('object') == binding.moving_object:
                entry.update(orientation_axis=copy.deepcopy(axial['axis']),
                             twist_tolerance=float(axial['twist_tolerance_rad']))
    return phases, success


def compile_precision_insertions(payload, settings):
    if not settings:
        return payload
    # Resolve all windows before replacing any; multiple sequential stages do
    # not borrow the first stage's body identities or phase indices.
    bindings = {key: infer_binding(payload, key, s['entry_waypoint'], s['binding']) for key, s in settings.items()}
    used = set()
    for binding in bindings.values():
        if used.intersection(binding.replace_phases):
            raise ValueError('Precision insertion windows overlap')
        used.update(binding.replace_phases)
    resolved = {}
    for key, setting in settings.items():
        binding = bindings[key]
        payload['phases'], payload['success'] = replace_insertion(payload['phases'], payload['success'], binding, setting)
        config = InsertionConfig(**setting['controller_config'])
        resolved[key] = dict(setting, binding=binding.to_dict(), controller='residual_rl', phase=binding.rl_phase,
                             physics_hz=config.physics_hz, decimation=config.decimation, controller_version=config.version,
                             held_seating_is_benchmark_success=False, release_and_unloaded_verification_preserved=True,
                             intermediate_support_retained=next(p for p in payload['phases']
                                                                if p['name'] == binding.settle_phase).get('physical_supported_verification', False))
    payload['precision_insertion_resolved'] = resolved
    if 'fabrica_canonical_resolved' in payload:
        payload['fabrica_canonical_resolved']['precision_insertion'] = copy.deepcopy(resolved)
    return payload


def preflight_precision_policies(recipe_spec):
    for stage, setting in recipe_spec.get('precision_insertion_resolved', {}).items():
        path = Path(setting['checkpoint'])
        if not path.is_file():
            raise FileNotFoundError(f'Insertion {stage} checkpoint missing: {path}')
        if setting.get('checkpoint_sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != setting['checkpoint_sha256']:
            raise ValueError(f'Insertion {stage} checkpoint_sha256 mismatch: {path}')


def require_precision_frequency(recipe_spec, control_fps):
    for setting in recipe_spec.get('precision_insertion_resolved', {}).values():
        if int(control_fps) != setting['physics_hz']:
            raise ValueError(f'Precision RL requires {setting["physics_hz"]} Hz physics/control')
