"""Open only enough to release a pin before leaving the other gripper."""
from __future__ import annotations

import copy
import math


def apply_pin_release_sequence(phases, *, spec):
    settings = spec.get('physical_pin_release', {})
    if not isinstance(settings, dict):
        raise ValueError('physical_pin_release must map pin ids to release settings')
    for part, raw in settings.items():
        if part not in ('1', '4') or not isinstance(raw, dict):
            raise ValueError('physical_pin_release requires pin ids 1 or 4 and settings mappings')
        config = dict(
            openness=0.34,
            minimum_openness=0.33,
            joint_rate=0.2,
            clear_force_steps=6,
            retreat_distance_m=0.06,
            retreat_speed_m_s=0.03,
        )
        if set(raw) - set(config):
            raise ValueError('Unknown physical_pin_release setting')
        config.update(raw)
        if (
            type(config['clear_force_steps']) is not int
            or not 2 <= config['clear_force_steps'] <= 30
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for v in config.values()
            )
            or not 0 < config['minimum_openness'] <= config['openness'] < 0.43
            or not 0.04 <= config['retreat_distance_m'] <= 0.15
        ):
            raise ValueError('Invalid physical_pin_release clearance, aperture or force qualification')
        obj = f'fabrica_plumbers_block_{part}'
        release = next((p for p in phases if p['name'].endswith(f'_part_{part}_release')), None)
        if release is None or not release.get('plumbers_base_support'):
            raise ValueError('physical_pin_release needs an existing supported release')
        prefix = release['name'].removesuffix('_release')
        retreat = next((p for p in phases if p['name'] == prefix + '_retreat'), None)
        if retreat is None:
            raise ValueError('physical_pin_release needs the physical retreat phase')
        robot = release['local_skill']['robot']
        release['local_skill'].update(release_openness=config['openness'], release_joint_rate=config['joint_rate'])
        release['physical_gripper_release'] = dict(
            robot=robot,
            object=obj,
            minimum_openness=config['minimum_openness'],
            clear_force_steps=config['clear_force_steps'],
        )

        def limited_gate(condition):
            if condition.get('type') == 'gripper_open' and condition.get('robot') == robot:
                condition['minimum_openness'] = config['minimum_openness']
            for child in condition.get('conditions', []):
                limited_gate(child)

        limited_gate(release['advance'])
        # This fixed target is captured once from the actual TCP and receiver
        # orientation. Keep the jaw aperture and roll throughout withdrawal.
        retreat['local_skill'].update(
            object=obj,
            requires_held_object=False,
            use_arm_ik_controller=False,
            require_success=True,
            fallback='fail',
            relative_to_current_tcp=True,
            offset=[0.0, 0.0, config['retreat_distance_m']],
            offset_frame='world',
            offset_reference_object=release['plumbers_base_support']['object'],
            lock_target_position=True,
            lock_target_orientation=True,
            gripper_command=config['openness'],
            free_space_trajectory=True,
            free_space_stage='approach',
            free_space_linear_speed=config['retreat_speed_m_s'],
            free_space_angular_speed=0.08,
            free_space_joint_speed=0.3,
            free_space_joint_acceleration=0.6,
            free_space_position_lookahead=0.003,
            free_space_orientation_lookahead=0.02,
            free_space_joint_lookahead=0.04,
            position_tolerance=0.0015,
            orientation_tolerance=0.02,
        )
        retreat['timeout_steps'] = max(retreat.get('timeout_steps', 0), 2400)
        retreat.setdefault('gripper_commands', {})[robot] = config['openness']
        opened = copy.deepcopy(release)
        opened['name'] = prefix + '_open_clear'
        opened.pop('detach', None)
        opened.pop('physical_gripper_release', None)
        opened['local_skill'].update(release_openness=1.0, release_joint_rate=0.6)
        opened['advance'] = dict(
            type='all_of',
            conditions=[
                dict(type='gripper_open', robot=robot, minimum_openness=0.95),
                dict(type='plumbers_support_ready'),
            ],
        )
        phases.insert(phases.index(retreat) + 1, opened)
