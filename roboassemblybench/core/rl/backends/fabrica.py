"""Infer an insertion description from existing Fabrica authoring metadata."""
import re


def binding_defaults(payload, stage_id, entry_waypoint):
    spec = payload.get('fabrica_canonical') or {}
    resolved = payload.get('fabrica_canonical_resolved') or {}
    order = resolved.get('assembly_order', [])
    matches = [(i,s) for i,s in enumerate(order) if str(s['move_part']) == str(stage_id)]
    if not matches:
        return {}
    index, step = matches[0]
    assembly, part = resolved['assembly'], str(stage_id)
    prefix = f'assemble_{index:02d}_part_{part}'
    pattern = re.compile(re.escape(prefix)+r'_insert_(\d+)$')
    phases = [p for p in payload['phases'] if pattern.fullmatch(p['name'])
              and int(pattern.fullmatch(p['name'])[1]) >= entry_waypoint]
    if not phases:
        raise ValueError(f'{stage_id}: no planned insertion window at entry_waypoint {entry_waypoint}')
    primary = phases[0].get('local_skill') or {}
    support = phases[0].get('plumbers_base_support') or {}
    names = {p['name'] for p in payload['phases']}
    supported_settle = f'{prefix}_supported_settle'
    retreat = support.get('cycle', '') + '_retreat'
    return dict(task_family=f'fabrica/{assembly}', stage_id=part,
        moving_object=f'fabrica_{assembly}_{part}', socket_object=f'fabrica_{assembly}_{step["socket_part"]}',
        insertion_robot=primary.get('robot') or spec.get('assembly_robot'),
        entry_target=f'part_{part}_insert_{entry_waypoint-1:02d}', goal_target=f'part_{part}_assembled',
        entry_phase=phases[0]['name'], replace_phases=tuple(p['name'] for p in phases),
        rl_phase=f'{prefix}_physical_rl',
        settle_phase=supported_settle if supported_settle in names else f'{prefix}_unloaded_settle',
        support_robot=support.get('robot'), support_object=support.get('object'),
        support_retreat_phase=retreat if retreat in names else None,
        readiness_conditions=({'type':'plumbers_support_ready'},) if support else ())
