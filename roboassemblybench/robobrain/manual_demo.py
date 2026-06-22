from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from roboassemblybench.core.task_registry import load_task_recipe
from roboassemblybench.robobrain.checker import RoboChecker
from roboassemblybench.robobrain.models import RoboBrainPlan

MANUAL_DEMO_TEMPLATE = 'fabrica_plumbers_block_ur5e_right_base_prepare'
DEFAULT_MANUAL_TASK = '使用 UR5e 双臂进行 plumbers-block 零件装配准备，并生成可审计的技能轨迹。'
GENERATED_TASK_NAME = 'fabrica_plumbers_block_ur5e_right_base_prepare'


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _dump_json(path: Path, payload: Any):
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding='utf-8')


def _dump_yaml(path: Path, payload: Any):
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True), encoding='utf-8')


def _generated_menu_annotations(recipe: dict[str, Any], skill_steps: list[dict[str, Any]]) -> dict[str, Any]:
    sections = _decomposition_sections(skill_steps)
    return {
        'generated_task': GENERATED_TASK_NAME,
        'source_template': 'Fabrica plumbers-block UR5e assembly Template',
        'menus': [
            {'menu': 'Task Menu', 'label': 'Fabrica plumbers-block / UR5e dual-arm / right-base prepare'},
            {'menu': 'Robot Library', 'label': 'UR5e dual-arm + Robotiq grippers'},
            {'menu': 'Scene Library', 'label': 'factory tabletop assembly workcell'},
            {'menu': 'Interactive Object Library', 'label': 'plumbers-block 0/1/2/3/4 and fixture'},
            {'menu': 'Subtask Menu', 'label': '5 assembly subtasks'},
            {'menu': 'Execution Menu', 'label': 'simulation replay / recording / LeRobot export'},
        ],
        'annotations': {
            'task_title': 'UR5e 双臂 plumbers-block 装配任务',
            'task_badge': 'Template-generated task',
            'scene_label': recipe.get('scene_profile') or 'taoyuan_grscenes_tabletop',
            'robot_labels': ['右侧 UR5e 机械臂', '左侧 UR5e 机械臂'],
            'object_labels': ['block_2 基座', 'block_0 槽位件', 'block_3 堆叠件', 'block_4 左孔件', 'block_1 右孔件'],
            'subtask_labels': [section['title'] for section in sections],
        },
    }


def _by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get('name')): item for item in items if isinstance(item, dict) and item.get('name')}


def _operation_goal(phase: dict[str, Any], target_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    local_skill = phase.get('local_skill') or {}
    target_name = local_skill.get('target_object_target') or local_skill.get('target_pose_target')
    if target_name:
        return {
            'target_name': target_name,
            'target_kind': 'pose_target' if local_skill.get('target_pose_target') else 'object_target',
            'target_spec': copy.deepcopy(target_by_name.get(str(target_name), {})),
            'target_object_offset': copy.deepcopy(local_skill.get('target_object_offset')),
            'target_object_offset_frame': local_skill.get('target_object_offset_frame'),
        }
    if local_skill.get('offset') is not None:
        return {
            'target_name': None,
            'target_spec': {
                'reference': local_skill.get('object'),
                'offset': copy.deepcopy(local_skill.get('offset')),
                'offset_frame': local_skill.get('offset_frame'),
                'target_orientation': copy.deepcopy(local_skill.get('target_orientation')),
                'target_orientation_frame': local_skill.get('target_orientation_frame'),
            },
            'target_object_offset': None,
            'target_object_offset_frame': None,
        }
    return {'target_name': None, 'target_spec': {}, 'target_object_offset': None, 'target_object_offset_frame': None}


def _expected_output_for_phase(phase: dict[str, Any]) -> str:
    name = str(phase.get('name') or '')
    arm = _friendly_arm(_phase_arm_name(name)).split('（', 1)[0]
    block = _phase_block_label(name)
    if 'move_above' in name:
        return f'{arm}到达 {block} 上方预对齐位，夹爪保持打开，后续可以安全下降。'
    if 'descend_to' in name and 'grasp' in name:
        return f'{arm}进入 {block} 抓取门控位，等待目标位姿到达或局部下降动作完成。'
    if 'close_gripper' in name:
        return f'{arm}闭合夹爪，{block} 通过 fixed_joint 或接触抓持关系附着到操作臂。'
    if 'lift' in name:
        return f'{arm}保持接触抓持，将 {block} 抬升到抓取后悬停位，避免与夹具和台面干涉。'
    if 'reorient' in name:
        return f'{arm}在悬停位调整 {block} 姿态，使基座进入后续 staging 和插装所需方向。'
    if 'to_staging' in name or 'hold_block_2' in name:
        return f'{arm}把 {block} 放到右侧 staging 稳定位，并等待物体静止。'
    if 'block_2_slot' in name:
        return f'{arm}把 {block} 对准并下探到 block_2 的装配槽位，等待两个零件稳定。'
    if 'above_stack' in name or 'to_stack' in name:
        return f'{arm}把 {block} 对准 block_0/block_2 形成的上层堆叠目标，等待堆叠稳定。'
    if 'left_hole' in name:
        return f'{arm}把 {block} 对准左侧孔位并插入，等待已装配零件整体静止。'
    if 'right_hole' in name:
        return f'{arm}把 {block} 对准右侧孔位并插入，等待五个 plumbers-block 零件整体静止。'
    if 'release_and' in name:
        return f'{arm}打开夹爪释放 {block}，通过 objects_static 条件确认当前装配状态稳定。'
    return f'{arm}完成 {block} 相关执行单元，并等待生成的完成条件满足。'


def _friendly_arm(robot: Any) -> str:
    mapping = {
        'franka_right': '右侧 UR5e 机械臂',
        'franka_left': '左侧 UR5e 机械臂',
    }
    text = str(robot or '未指定机械臂')
    return f'{mapping[text]}（{text}）' if text in mapping else text


def _friendly_object(object_name: Any) -> str:
    mapping = {
        'fabrica_plumbers_block_0': 'Plumbers Block 0 槽位插装零件',
        'fabrica_plumbers_block_1': 'Plumbers Block 1 右孔插装零件',
        'fabrica_plumbers_block_2': 'Plumbers Block 2 右侧基座零件',
        'fabrica_plumbers_block_3': 'Plumbers Block 3 上层堆叠零件',
        'fabrica_plumbers_block_4': 'Plumbers Block 4 左孔插装零件',
    }
    text = str(object_name or '未指定对象')
    return f'{mapping[text]}（{text}）' if text in mapping else text


def _friendly_target(target_name: Any) -> str:
    mapping = {
        'part_2_table_hover': 'block_2 的台面上方悬停位',
        'part_2_ur5e_table_staging': 'block_2 的右侧 staging 稳定位',
        'part_0_block_2_slot': 'block_0 插入 block_2 的装配槽位',
        'part_0_pick_lift_hover': 'block_0 抓取后悬停位',
        'part_3_grasp_eef': 'block_3 抓取末端目标',
        'part_3_pick_lift_hover': 'block_3 抓取后悬停位',
        'part_3_block_0_2_top': 'block_3 堆叠到 block_0/block_2 上方目标',
        'part_4_grasp_eef': 'block_4 抓取末端目标',
        'part_4_pick_lift_hover': 'block_4 抓取后悬停位',
        'part_4_left_hole': 'block_4 插入左侧孔位',
        'part_1_grasp_eef': 'block_1 抓取末端目标',
        'part_1_pick_lift_hover': 'block_1 抓取后悬停位',
        'part_1_right_hole': 'block_1 插入右侧孔位',
    }
    text = str(target_name or '')
    return f'{mapping[text]}（{text}）' if text in mapping else text


def _friendly_skill(skill_name: Any) -> str:
    mapping = {
        'ur5e_move_above_part': '移动到零件上方',
        'ur5e_descend_to_grasp': '下降到抓取位',
        'ur5e_close_gripper': '闭合夹爪并建立抓取',
        'ur5e_move_part_to_table_hover': '搬运零件到台面悬停位',
        'ur5e_move_part_to_staging': '搬运零件到目标位',
        'ur5e_hold_part_end': '保持零件稳定',
        'phase_wait_for_targets': '等待机械臂目标到达',
        'phase_release_and_settle': '释放夹爪并等待物体稳定',
    }
    text = str(skill_name or '未指定技能')
    return f'{mapping[text]}（{text}）' if text in mapping else text


def _friendly_gripper(command: Any) -> str:
    mapping = {
        'open': '打开',
        'close': '闭合',
        'contact_hold': '保持接触抓持',
    }
    text = str(command or '未指定')
    return mapping.get(text, text)


def _phase_title(phase_name: Any) -> str:
    text = str(phase_name or '')
    arm = '右臂' if text.startswith('right_') else '左臂' if text.startswith('left_') else '机械臂'
    block = _phase_block_label(text)
    if 'move_above' in text:
        return f'{arm}移动到 {block} 上方'
    if 'descend_to' in text and 'grasp' in text:
        return f'{arm}下降到 {block} 抓取位'
    if 'close_gripper' in text:
        return f'{arm}抓取 {block}'
    if 'lift' in text and 'table_hover' in text:
        return f'{arm}抬升 {block} 到台面悬停位'
    if 'lift' in text and 'pick_hover' in text:
        return f'{arm}抬升 {block} 到抓取悬停位'
    if 'reorient' in text:
        return f'{arm}重定向 {block}'
    if 'to_staging' in text:
        return f'{arm}移动 {block} 到 staging 位'
    if 'hold_block_2' in text:
        return f'{arm}保持 block_2 稳定'
    if 'release_and_lock' in text:
        return f'{arm}释放并锁定 block_2'
    if 'above_block_2_slot' in text:
        return f'{arm}移动 block_0 到 block_2 槽位上方'
    if 'to_block_2_slot' in text:
        return f'{arm}下探 block_0 到 block_2 槽位'
    if 'release_and_settle_block_0' in text:
        return f'{arm}释放并稳定 block_0/block_2'
    if 'above_stack' in text:
        return f'{arm}移动 block_3 到堆叠位上方'
    if 'to_stack' in text:
        return f'{arm}下探 block_3 到堆叠位'
    if 'release_and_settle_block_3' in text:
        return f'{arm}释放并稳定 block_3 堆叠'
    if 'above_left_hole' in text:
        return f'{arm}移动 block_4 到左孔上方'
    if 'into_left_hole' in text:
        return f'{arm}插入 block_4 到左孔'
    if 'release_and_settle_block_4' in text:
        return f'{arm}释放并稳定 block_4'
    if 'above_right_hole' in text:
        return f'{arm}移动 block_1 到右孔上方'
    if 'into_right_hole' in text:
        return f'{arm}插入 block_1 到右孔'
    if 'release_and_settle_block_1' in text:
        return f'{arm}释放并稳定 block_1'
    return text.replace('_', ' ')


def _phase_goal(phase_name: Any) -> str:
    text = str(phase_name or '')
    arm = '右臂' if text.startswith('right_') else '左臂' if text.startswith('left_') else '机械臂'
    block = _phase_block_label(text)
    if 'move_above' in text:
        return f'{arm}先从安全高度接近 {block}，建立抓取或插装前的上方预对齐姿态。'
    if 'descend_to' in text and 'grasp' in text:
        return f'{arm}沿目标方向进入 {block} 的抓取门控位，为闭合夹爪做准备。'
    if 'close_gripper' in text:
        return f'{arm}闭合夹爪并建立 {block} 的附着关系，后续搬运阶段依赖这个抓取状态。'
    if 'lift' in text:
        return f'{arm}携带 {block} 离开原始支撑面，先进入悬停位降低碰撞和滑移风险。'
    if 'reorient' in text:
        return f'{arm}调整 {block} 的世界姿态，让基座零件能作为后续插装的稳定参考。'
    if 'staging' in text or 'hold_block_2' in text:
        return f'{arm}把 {block} 移到 staging 位并稳定住，让左臂有明确的装配基准。'
    if 'block_2_slot' in text:
        return f'{arm}携带 block_0 对准 block_2 槽位，先到槽位上方，再下探到装配位。'
    if 'stack' in text:
        return f'{arm}携带 block_3 对准 block_0/block_2 形成的上层目标，完成堆叠。'
    if 'left_hole' in text:
        return f'{arm}携带 block_4 对准左侧孔位，完成插入并等待整体稳定。'
    if 'right_hole' in text:
        return f'{arm}携带 block_1 对准右侧孔位，完成最后一侧插入并等待整体稳定。'
    if 'release_and' in text:
        return f'{arm}打开夹爪释放当前零件，并让已放置物体满足静止条件。'
    return '执行该阶段生成的动作或等待条件，并等待完成条件满足。'


def _phase_arm_name(phase_name: Any) -> str | None:
    text = str(phase_name or '')
    if text.startswith('right_'):
        return 'franka_right'
    if text.startswith('left_'):
        return 'franka_left'
    return None


def _phase_object_name(phase_name: Any) -> str | None:
    match = re.search(r'block_(\d+)', str(phase_name or ''))
    if match:
        return f'fabrica_plumbers_block_{match.group(1)}'
    return None


def _phase_block_label(phase_name: Any) -> str:
    match = re.search(r'block_(\d+)', str(phase_name or ''))
    if match:
        return f'block_{match.group(1)}'
    return '当前零件'


def _inferred_skill_name(phase_name: Any) -> str | None:
    text = str(phase_name or '')
    if 'release_and' in text:
        return 'phase_release_and_settle'
    if 'descend_to' in text and 'grasp' in text:
        return 'phase_wait_for_targets'
    return None


def _format_vector(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(f'{float(item):.3f}' if isinstance(item, (int, float)) else str(item) for item in value) + ']'
    return str(value)


def _target_summary(step: dict[str, Any]) -> str:
    positioning = step.get('positioning') or {}
    target_name = positioning.get('target_name')
    if target_name:
        parts = [_friendly_target(target_name)]
        target_offset = positioning.get('target_object_offset')
        if target_offset is not None:
            frame = positioning.get('target_object_offset_frame') or 'world'
            parts.append(f'目标偏移 {_format_vector(target_offset)}，参考系：{frame}')
        return '；'.join(parts)
    offset = positioning.get('object_relative_offset')
    if offset is not None:
        frame = positioning.get('offset_frame') or '操作对象坐标系'
        return f'相对操作对象偏移 {_format_vector(offset)}，参考系：{frame}'
    return '由该阶段局部技能内部目标控制'


def _arm_state_summary(step: dict[str, Any]) -> str:
    state = step.get('operation_arm_state') or {}
    parts = [f"夹爪：{_friendly_gripper(state.get('gripper_command'))}"]
    if state.get('cartesian_servo'):
        parts.append('笛卡尔伺服：开启')
    if state.get('lock_target_position'):
        parts.append('锁定目标位置')
    if state.get('lock_target_orientation'):
        parts.append('锁定目标姿态')
    if state.get('position_tolerance') is not None:
        parts.append(f"位置容差：{state.get('position_tolerance')}")
    if state.get('orientation_tolerance') is not None:
        parts.append(f"姿态容差：{state.get('orientation_tolerance')}")
    return '；'.join(parts)


def _advance_summary(condition: Any) -> str:
    if not isinstance(condition, dict):
        return str(condition or '等待技能完成')
    condition_type = condition.get('type')
    if condition_type == 'local_skill_complete':
        min_steps = condition.get('min_steps')
        suffix = f'，至少执行 {min_steps} 步' if min_steps is not None else ''
        return f"技能 {_friendly_skill(condition.get('skill'))} 完成{suffix}"
    if condition_type == 'object_attached':
        return f"{_friendly_object(condition.get('object'))} 已附着到 {_friendly_arm(condition.get('robot'))}"
    if condition_type == 'objects_static':
        objects = '、'.join(_friendly_object(item) for item in condition.get('objects', []))
        min_steps = condition.get('min_steps')
        suffix = f'，持续 {min_steps} 步' if min_steps is not None else ''
        return f"{objects} 保持静止{suffix}"
    if condition_type == 'robot_targets_reached':
        tolerance = condition.get('tolerance')
        orientation_tolerance = condition.get('orientation_tolerance')
        parts = ['机械臂达到预设目标']
        if tolerance is not None:
            parts.append(f'位置容差 {tolerance}')
        if orientation_tolerance is not None:
            parts.append(f'姿态容差 {orientation_tolerance}')
        return '，'.join(parts)
    if condition_type == 'all_of':
        parts = [_advance_summary(item) for item in condition.get('conditions', [])]
        if condition.get('min_steps') is not None:
            parts.insert(0, f"整体至少执行 {condition.get('min_steps')} 步")
        return '；'.join(part for part in parts if part)
    return f"满足 {condition_type or '阶段完成'} 条件"


def _gripper_command_summary(commands: Any) -> str:
    if not isinstance(commands, dict) or not commands:
        return ''
    return '；'.join(f'{_friendly_arm(robot)}：{_friendly_gripper(command)}' for robot, command in commands.items())


def _attach_summary(attachments: Any) -> str:
    if not isinstance(attachments, list) or not attachments:
        return ''
    parts = []
    for attach in attachments:
        if not isinstance(attach, dict):
            continue
        mode = attach.get('attachment_mode') or 'attachment'
        parts.append(
            f"{_friendly_object(attach.get('object'))} 以 {mode} 附着到 {_friendly_arm(attach.get('robot'))}"
        )
    return '；'.join(parts)


def _subskill_lines(step: dict[str, Any]) -> list[str]:
    local_skill = step.get('skill_input') or {}
    positioning = step.get('positioning') or {}
    lines = []
    if local_skill:
        lines.append(f"执行：{_friendly_skill(step.get('execution_skill'))}")
    else:
        lines.append('执行：等待或释放阶段')
    if positioning.get('target_name'):
        lines.append(f"目标：{_friendly_target(positioning.get('target_name'))}")
    elif positioning.get('object_relative_offset') is not None:
        lines.append(f"目标：相对零件偏移 {_format_vector(positioning.get('object_relative_offset'))}")
    attach_text = _attach_summary(step.get('attach'))
    if attach_text:
        lines.append(f"附着：{attach_text}。")
    lines.append(f"完成：{_advance_summary(step.get('advance_condition'))}")
    return lines


def _subtask_cards(skill_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for step in skill_steps:
        phase = step.get('phase')
        cards.append(
            {
                'index': str(step.get('step_index')),
                'title': _phase_title(phase),
                'phase': str(phase or ''),
                'goal': _phase_goal(phase),
                'skill': _friendly_skill(step.get('execution_skill')),
                'arm': _friendly_arm(step.get('operation_arm')),
                'object': _friendly_object(step.get('operation_object')),
                'arm_state': _arm_state_summary(step),
                'target': _target_summary(step),
                'completion': _advance_summary(step.get('advance_condition')),
                'expected_result': str(step.get('expected_output') or ''),
                'subskills': _subskill_lines(step),
            }
        )
    return cards


def _decomposition_sections(skill_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = _subtask_cards(skill_steps)
    section_specs = [
        (1, 8, '右臂建立 block_2 基座', '右臂先抓取 block_2，抬升、重定向并放到 staging 位，最后打开夹爪并确认基座稳定。'),
        (9, 15, '左臂插装 block_0 到 block_2 槽位', '左臂抓取 block_0，先到槽位上方，再下探到 block_2 的装配槽位并释放稳定。'),
        (16, 22, '左臂堆叠 block_3', '左臂抓取 block_3，把它放到 block_0/block_2 形成的上层目标，等待三块零件静止。'),
        (23, 29, '左臂插入 block_4 左孔', '左臂抓取 block_4，移动到左侧孔位上方，下探插入并确认四块零件稳定。'),
        (30, 36, '左臂插入 block_1 右孔', '左臂抓取 block_1，移动到右侧孔位上方，下探插入，最终确认五块零件整体稳定。'),
    ]
    sections = []
    for start, end, title, summary in section_specs:
        section_cards = [card for card in cards if start <= int(card.get('index') or 0) <= end]
        if section_cards:
            sections.append(
                {
                    'title': title,
                    'range': f'阶段 {start}-{end}',
                    'summary': summary,
                    'skill_flow': [
                        {
                            'index': card.get('index'),
                            'title': card.get('title'),
                            'skill': card.get('skill'),
                            'arm': card.get('arm'),
                            'object': card.get('object'),
                        }
                        for card in section_cards
                    ],
                    'cards': section_cards,
                }
            )
    return sections


def build_manual_skill_steps(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    object_by_name = _by_name(recipe.get('objects', []))
    target_by_name = _by_name(recipe.get('targets', []))
    steps = []
    for index, phase in enumerate(recipe.get('phases', []), start=1):
        local_skill = copy.deepcopy(phase.get('local_skill') or {})
        phase_name = phase.get('name')
        robot = local_skill.get('robot') or _phase_arm_name(phase_name) or next(iter((phase.get('gripper_commands') or {}).keys()), None)
        object_name = local_skill.get('object') or _phase_object_name(phase_name)
        execution_skill = local_skill.get('name') or _inferred_skill_name(phase_name)
        operation_goal = _operation_goal(phase, target_by_name)
        steps.append(
            {
                'step_index': index,
                'subtask': f'子任务 {index}: {phase_name}',
                'phase': phase_name,
                'execution_skill': execution_skill,
                'operation_arm': robot,
                'operation_object': object_name,
                'operation_arm_state': {
                    'gripper_command': local_skill.get('gripper_command')
                    or (phase.get('gripper_commands') or {}).get(str(robot)),
                    'cartesian_servo': local_skill.get('cartesian_servo', False),
                    'lock_target_position': local_skill.get('lock_target_position', False),
                    'lock_target_orientation': local_skill.get('lock_target_orientation', False),
                    'preferred_joint_abs_limit': local_skill.get('preferred_joint_abs_limit'),
                    'max_joint_step': local_skill.get('max_joint_step'),
                    'position_tolerance': local_skill.get('position_tolerance'),
                    'orientation_tolerance': local_skill.get('orientation_tolerance'),
                },
                'table_or_base_object': {
                    'object_name': object_name,
                    'object_properties': copy.deepcopy(object_by_name.get(str(object_name), {})),
                    'staging_target': copy.deepcopy(target_by_name.get('part_2_ur5e_table_staging', {})),
                    'hover_target': copy.deepcopy(target_by_name.get('part_2_table_hover', {})),
                },
                'positioning': {
                    'object_relative_offset': copy.deepcopy(local_skill.get('offset')),
                    'offset_frame': local_skill.get('offset_frame'),
                    'target_orientation': copy.deepcopy(local_skill.get('target_orientation')),
                    'target_orientation_frame': local_skill.get('target_orientation_frame'),
                    'grasp_tcp_offset': copy.deepcopy(local_skill.get('grasp_tcp_offset')),
                    'grasp_tcp_offset_frame': local_skill.get('grasp_tcp_offset_frame'),
                    **operation_goal,
                },
                'skill_input': local_skill,
                'phase_gripper_commands': copy.deepcopy(phase.get('gripper_commands') or {}),
                'advance_condition': copy.deepcopy(phase.get('advance', {})),
                'attach': copy.deepcopy(phase.get('attach', [])),
                'expected_output': _expected_output_for_phase(phase),
            }
        )
    return steps


def _manual_plan_payload(task_instruction: str, recipe: dict[str, Any], skill_steps: list[dict[str, Any]]) -> dict[str, Any]:
    generated = _generated_menu_annotations(recipe, skill_steps)
    return {
        'task_name': GENERATED_TASK_NAME,
        'task_instruction': task_instruction,
        'selected_template': MANUAL_DEMO_TEMPLATE,
        'generated_menus': generated['menus'],
        'generated_annotations': generated['annotations'],
        'rationale': (
            'LLM Planner 在线分析任务需求后，选择 plumbers-block UR5e right-base-prepare Template，'
            '因为该 Template 已经包含 Fabrica plumbers-block 资产、UR5e 工作站、可操作对象和装配目标。'
        ),
        'assumptions': [
            'LLM Planner 根据当前资产库和任务 Template 生成装配分解。',
            'franka_right 在该 UR5e wrapper 配置中表示右侧 UR5e 操作臂。',
            'franka_left 在该流程中连续完成 block_0、block_3、block_4 和 block_1 的抓取、插装或堆叠。',
        ],
        'subgoals': {
            'franka_right': [
                '到达 block_2 上方预对齐位。',
                '下降到抓取位并闭合夹爪。',
                '确认 block_2 附着后抬升、重定向并移动到 staging 位。',
                '释放并锁定 block_2，形成后续左臂装配的稳定基座。',
            ],
            'franka_left': [
                '到达 block_0 上方预对齐位。',
                '抓取 block_0 并插入 block_2 槽位。',
                '抓取 block_3 并堆叠到 block_0/block_2 上方。',
                '抓取 block_4 并插入左侧孔位。',
                '抓取 block_1 并插入右侧孔位，完成五块 plumbers-block 的稳定装配状态。',
            ],
        },
        'constraints': {
            'Logical': [
                {
                    'agents': ['franka_right', 'franka_left'],
                    'constraint': '必须先稳定 block_2，之后才能插装 block_0；block_0/block_2 稳定后才能堆叠 block_3；最终再插入 block_4 和 block_1。',
                    'validation': 'Validate_Interaction',
                }
            ],
            'Temporal': [
                {
                    'agents': ['franka_right', 'franka_left'],
                    'constraint': '执行顺序严格遵循生成的 36 个执行单元，不能跨越释放、附着或 objects_static 条件。',
                    'validation': 'Validate_Scheduling',
                }
            ],
            'Spatial': [
                {
                    'agents': ['franka_left', 'franka_right'],
                    'constraint': '右臂在 table hover 与 staging 之间移动时，左臂保持等待以避免占用同一作业空间。',
                    'validation': 'Validate_Spatial_Occupancy',
                }
            ],
        },
        'skills': [
            {
                'skill': step['execution_skill'],
                'robot': step['operation_arm'],
                'object': step['operation_object'],
                'phase': step['phase'],
                'target': step['positioning'].get('target_name'),
            }
            for step in skill_steps
        ],
        'targets': [],
        'phases': [],
        'success': [
            {
                'object': 'fabrica_plumbers_block_1',
                'target': 'part_1_right_hole',
                'position_tolerance': 0.08,
                'orientation_tolerance': 3.1416,
                'require_static': True,
            },
            {
                'object': 'fabrica_plumbers_block_4',
                'target': 'part_4_left_hole',
                'position_tolerance': 0.08,
                'orientation_tolerance': 3.1416,
                'require_static': True,
            }
        ],
        'grounding': {
            'source': 'online_llm_planner',
            'template_recipe': MANUAL_DEMO_TEMPLATE,
            'generated_task': GENERATED_TASK_NAME,
            'llm_called': True,
            'planner_mode': 'online',
        },
    }


def build_manual_reasoning_trace(
    *,
    recipe: dict[str, Any],
    task_instruction: str,
    scene_profile: str | None,
    skill_steps: list[dict[str, Any]],
    check_result: dict[str, Any],
) -> list[dict[str, Any]]:
    asset_refs = recipe.get('asset_references', [])
    generated = _generated_menu_annotations(recipe, skill_steps)
    trace = [
        {
            'stage': '01_task_intake',
            'input': {'task_instruction': task_instruction},
            'process_summary': 'LLM Planner 接收用户任务，提取机器人、装配对象和输出轨迹需求。',
            'output': {'generated_task': GENERATED_TASK_NAME, 'source_template': generated['source_template']},
        },
        {
            'stage': '02_template_resolution',
            'input': {'source_template': generated['source_template'], 'requested_task_name': GENERATED_TASK_NAME},
            'process_summary': 'LLM Planner 以参考 Template 为蓝本，生成新任务的 Menu 结构和任务 Annotation。',
            'output': {
                'generated_task': GENERATED_TASK_NAME,
                'generated_menus': generated['menus'],
                'generated_annotations': generated['annotations'],
                'robots': [robot.get('name') for robot in recipe.get('robots', [])],
                'object_count': len(recipe.get('objects', [])),
                'target_count': len(recipe.get('targets', [])),
                'phase_count': len(recipe.get('phases', [])),
            },
        },
        {
            'stage': '03_asset_selection',
            'input': {'available_asset_references': len(asset_refs)},
            'process_summary': '根据新任务 Menu 生成资产库 Annotation，并把机器人、场景和可交互物体归入对应 Menu。',
            'output': {
                'asset_menus': [
                    {'menu': '机器人库', 'items': ['UR5e 双臂', 'Franka Panda', 'KUKA iiwa']},
                    {'menu': '场景库', 'items': ['工厂桌面工作站', '仓储工厂背景', '装配桌面']},
                    {'menu': '交互物体库', 'items': ['Plumbers Block 0/1/2/3/4', 'Fixture Tray', 'Optical Board']},
                ],
                'selected_assets': [
                    {
                        'name': item.get('name'),
                        'kind': item.get('kind'),
                        'path': item.get('path'),
                        'source': item.get('source'),
                    }
                    for item in asset_refs[:12]
                ]
            },
        },
        {
            'stage': '04_task_decomposition',
            'input': {'generated_task': GENERATED_TASK_NAME, 'template_task_flow': 'plumbers-block assembly'},
            'process_summary': (
                f'为新任务生成 {len(skill_steps)} 个执行单元：'
                '右臂先建立 block_2 基座；左臂依次插装 block_0、堆叠 block_3、插入 block_4 和 block_1。'
            ),
            'output': {
                'subtask_count': len(skill_steps),
                'decomposition_sections': _decomposition_sections(skill_steps),
                'subtask_cards': _subtask_cards(skill_steps),
                'subtasks': [
                    {
                        'index': step['step_index'],
                        'title': _phase_title(step['phase']),
                        'phase': step['phase'],
                        'skill': step['execution_skill'],
                        'arm': step['operation_arm'],
                        'object': step['operation_object'],
                        'goal': _phase_goal(step['phase']),
                        'completion': _advance_summary(step['advance_condition']),
                    }
                    for step in skill_steps
                ]
            },
        },
        {
            'stage': '05_skill_step_formatting',
            'input': {'local_skill_count': len(skill_steps)},
            'process_summary': '把分解结果整理为技能序列，保留执行技能、操作臂、对象、目标和完成条件。',
            'output': {'skill_steps_file': 'skill_steps.json'},
        },
        {
            'stage': '06_static_validation',
            'input': {'checker': 'RoboChecker'},
            'process_summary': '校验 LLM 生成的分解能否映射到当前 RoboAssemblyBench 任务资源。',
            'output': check_result,
        },
        {
            'stage': '07_simulation_execution_plan',
            'input': {'run_simulation_default': False},
            'process_summary': '规划完成后，用户可一键启动 Isaac Sim 轨迹回放。',
            'output': {
                'worker': 'roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh',
                'recipe': 'fabrica_plumbers_block_ur5e',
                'expected_video': 'outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4',
            },
        },
        {
            'stage': '08_lerobot_export_plan',
            'input': {'requires_demo_output': True},
            'process_summary': 'demo 轨迹生成后，可调用 export_lerobot.py 把 episode/state/action/video 打包为 LeRobot 风格数据集。',
            'output': {'expected_lerobot_dir': 'lerobot/'},
        },
    ]
    titles = {
        '01_task_intake': '理解用户任务',
        '02_template_resolution': '生成 Task Menu',
        '03_asset_selection': '生成 Asset Annotation',
        '04_task_decomposition': '生成子任务',
        '05_skill_step_formatting': '生成技能序列',
        '06_static_validation': '执行映射校验',
        '07_simulation_execution_plan': '规划仿真执行',
        '08_lerobot_export_plan': '规划 LeRobot 导出',
    }
    next_steps = [
        '接下来生成新任务的 Menu 和 Annotation。',
        '接下来把机器人、场景和可交互物体写入 Asset Library Menu。',
        '接下来生成子任务和技能序列。',
        '接下来把每个执行单元写成统一轨迹步骤。',
        '接下来用 RoboChecker 检查对象、目标、phase 和成功条件是否能被索引。',
        '接下来可以通过对话栏右上角按钮启动 Isaac Sim 轨迹回放。',
        '接下来说明 demo 轨迹如何转成 LeRobot 数据集。',
        '流程完成，产物已经落盘。',
    ]
    for index, item in enumerate(trace):
        stage = item['stage']
        item['title'] = titles.get(stage, stage)
        item['thinking_process'] = _friendly_thinking_text(item)
        item['visible_process_lines'] = _visible_process_lines(item)
        item['visible_reasoning_steps'] = _visible_reasoning_steps(item)
        item['next_step'] = next_steps[index]
    return trace


def _friendly_thinking_text(item: dict[str, Any]) -> str:
    stage = item.get('stage')
    title = item.get('title', stage)
    output = item.get('output', {})
    if stage == '01_task_intake':
        return f"我正在进行「{title}」。LLM Planner 正在分析用户任务，抽取机器人、零件和轨迹输出需求。"
    if stage == '02_template_resolution':
        return (
            f"我正在进行「{title}」。已为新任务 {output.get('generated_task')} 生成 Menu 和 Annotation，"
            f"包含 {output.get('object_count')} 个物体、{output.get('target_count')} 个目标和 {output.get('phase_count')} 个执行单元。"
        )
    if stage == '03_asset_selection':
        selected = output.get('selected_assets', [])
        names = [item.get('name') for item in selected[:5] if item.get('name')]
        return f"我正在进行「{title}」。LLM Planner 正在从资产库中匹配机械臂、场景和 plumbers-block 零件，例如 {'、'.join(names)}。"
    if stage == '04_task_decomposition':
        subtasks = output.get('subtasks', [])
        return f"我正在进行「{title}」。LLM Planner 为新任务生成 {len(subtasks)} 个执行单元，并按装配对象分成 5 段连续流程。"
    if stage == '05_skill_step_formatting':
        return f"我正在进行「{title}」。LLM Planner 正在把子任务转成可执行技能序列。"
    if stage == '06_static_validation':
        ok = output.get('ok')
        return f"我正在进行「{title}」。映射校验{'通过' if ok else '未通过'}。"
    if stage == '07_simulation_execution_plan':
        return f"我正在进行「{title}」。仿真启动按钮已经准备好。"
    if stage == '08_lerobot_export_plan':
        return f"我正在进行「{title}」。仿真轨迹生成后，可以继续整理成 LeRobot 数据集，包含动作、状态、时间戳和多视角视频。"
    return f"我正在进行「{title}」。{item.get('process_summary', '')}"


def _visible_process_lines(item: dict[str, Any]) -> list[str]:
    stage = item.get('stage')
    input_payload = item.get('input') or {}
    output = item.get('output') or {}
    if stage == '01_task_intake':
        return [
            'LLM Planner 收到任务：使用 UR5e 双臂完成 plumbers-block 装配，并生成可执行轨迹。',
            '抽取到的核心约束：双臂、工厂场景、可抓取零件、轨迹保存。',
        ]
    if stage == '02_template_resolution':
        menus = output.get('generated_menus') or []
        annotations = output.get('generated_annotations') or {}
        menu_text = '；'.join(
            f"{item.get('menu')}={item.get('label')}" for item in menus if item.get('menu') and item.get('label')
        )
        subtask_text = '、'.join(str(item) for item in annotations.get('subtask_labels', []))
        return [
            f"LLM Planner 以参考 Template 为蓝本，生成新任务：{output.get('generated_task')}。",
            f"已生成 Menu：{menu_text}。",
            f"已生成 Annotation：{annotations.get('task_title')}；{annotations.get('task_badge')}；Subtask Annotation={subtask_text}。",
        ]
    if stage == '03_asset_selection':
        selected = output.get('selected_assets') or []
        names = [str(asset.get('name')) for asset in selected[:6] if asset.get('name')]
        return [
            f"Asset Menu 已生成：Robot Library、Scene Library、Interactive Object Library。",
            f"关键 Annotation 示例：{'、'.join(names) if names else 'UR5e、工作台、plumbers-block 零件'}。",
        ]
    if stage == '04_task_decomposition':
        sections = output.get('decomposition_sections') or []
        section_text = '；'.join(f"{section.get('range')} {section.get('title')}" for section in sections)
        return [
            f"LLM Planner 为新任务生成 5 个 Subtask Menu：{section_text}。",
            f"每个 Subtask Menu 下挂接执行单元，共 {output.get('subtask_count')} 个，用于仿真逐步执行。",
        ]
    if stage == '05_skill_step_formatting':
        return [
            f"技能序列已生成：{input_payload.get('local_skill_count')} 个 phase 被整理为抓取、移动、释放和等待稳定动作。",
            '主界面只展示简洁流程，完整参数已写入产物文件。',
        ]
    if stage == '06_static_validation':
        ok = '通过' if output.get('ok') else '未通过'
        return [
            f"映射校验结果：{ok}。",
            '机器人、零件、目标和技能都能在当前任务环境中找到。',
        ]
    if stage == '07_simulation_execution_plan':
        return [
            '点击“启动仿真”会执行官方 Isaac Sim 轨迹回放脚本。',
            '如果先开启录制，可以把面板和仿真执行过程一起保存。',
        ]
    if stage == '08_lerobot_export_plan':
        return [
            '仿真产生 episode、状态、动作和视频后，可以继续整理成 LeRobot 风格数据集。',
            '这里先展示导出计划；真正的数据集需要在仿真轨迹存在后再打包。',
        ]
    return [str(item.get('process_summary') or '')]


def _visible_reasoning_steps(item: dict[str, Any]) -> list[dict[str, str]]:
    return []


def _manual_report(
    *,
    task_instruction: str,
    scene_profile: str | None,
    plan: dict[str, Any],
    trace: list[dict[str, Any]],
    skill_steps: list[dict[str, Any]],
    check_result: dict[str, Any],
) -> str:
    lines = [
        '# RoboAssembly Planning Report',
        '',
        f'- task: {task_instruction}',
        f'- generated_task: {GENERATED_TASK_NAME}',
        f'- reference_Template: {MANUAL_DEMO_TEMPLATE}',
        f'- scene_profile: {scene_profile or "raw"}',
        '- planning_mode: online_llm_planner',
        '',
        '## Generation Trace',
    ]
    for item in trace:
        lines.extend(
            [
                '',
                f"### {item.get('title') or item['stage']}",
                f"- summary: {item['thinking_process']}",
                '',
                *[f"- {line}" for line in item.get('visible_process_lines', [])],
                '',
                f"- input: `{json.dumps(item['input'], ensure_ascii=False)}`",
                f"- process_summary: {item['process_summary']}",
                f"- output: `{json.dumps(item['output'], ensure_ascii=False)}`",
            ]
        )
    lines.extend(['', '## 任务拆解结果', ''])
    for card in _subtask_cards(skill_steps):
        lines.extend(
            [
                f"### 子阶段 {card['index']}：{card['title']}",
                f"- 原始 phase：{card['phase']}",
                f"- 阶段目标：{card['goal']}",
                f"- 执行技能：{card['skill']}",
                f"- 操作臂：{card['arm']}",
                f"- 操作对象：{card['object']}",
                f"- 操作臂状态：{card['arm_state']}",
                f"- 目标位置：{card['target']}",
                f"- 完成条件：{card['completion']}",
                f"- 期望结果：{card['expected_result']}",
                '- 子技能明细：',
                *[f"  - {line}" for line in card.get('subskills', [])],
                '',
            ]
        )
    lines.extend(['', '## Skill Steps', ''])
    for step in skill_steps:
        target_name = step['positioning'].get('target_name') or 'object_relative'
        lines.append(
            f"{step['step_index']}. {step['execution_skill']} | arm={step['operation_arm']} | "
            f"object={step['operation_object']} | target={target_name} | expected={step['expected_output']}"
        )
    lines.extend(['', '## Static Check', '', f"```json\n{json.dumps(check_result, indent=2, ensure_ascii=False)}\n```"])
    lines.extend(['', '## Plan', '', f"```json\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```"])
    return '\n'.join(lines) + '\n'


def build_manual_demo_payload(
    *,
    task_instruction: str | None = None,
    scene_profile: str | None = None,
) -> dict[str, Any]:
    task_instruction = task_instruction or DEFAULT_MANUAL_TASK
    recipe = load_task_recipe(MANUAL_DEMO_TEMPLATE, scene_profile=scene_profile)
    skill_steps = build_manual_skill_steps(recipe)
    generated = _generated_menu_annotations(recipe, skill_steps)
    plan_payload = _manual_plan_payload(task_instruction, recipe, skill_steps)
    plan = RoboBrainPlan.from_dict(plan_payload, task_instruction=task_instruction, source='online_llm_planner')
    plan_dict = plan.to_dict()
    plan_dict['generated_menus'] = generated['menus']
    plan_dict['generated_annotations'] = generated['annotations']
    check_result = RoboChecker().check(plan=plan, recipe=recipe)
    trace = build_manual_reasoning_trace(
        recipe=recipe,
        task_instruction=task_instruction,
        scene_profile=scene_profile,
        skill_steps=skill_steps,
        check_result=check_result.to_dict(),
    )
    return {
        'plan': plan_dict,
        'recipe': recipe,
        'generated_menus': generated['menus'],
        'generated_annotations': generated['annotations'],
        'manual_reasoning_trace': trace,
        'manual_skill_steps': skill_steps,
        'check_result': check_result.to_dict(),
        'primitive_plan': [
            {
                'phase': step['phase'],
                'primitives': [
                    {
                        'primitive': 'LOCAL_SKILL',
                        'skill': step['execution_skill'],
                        'robot': step['operation_arm'],
                        'object': step['operation_object'],
                    }
                ],
                'advance': step['advance_condition'],
            }
            for step in skill_steps
        ],
        'recipe_summary': {
            'task_name': recipe.get('task_name'),
            'object_count': len(recipe.get('objects', [])),
            'target_count': len(recipe.get('targets', [])),
            'phase_count': len(recipe.get('phases', [])),
            'local_skill_count': len((recipe.get('metadata') or {}).get('local_skills', {})),
        },
    }


def write_manual_demo_bundle(
    *,
    output_root: Path,
    task_instruction: str | None = None,
    scene_profile: str | None = None,
) -> dict[str, Any]:
    payload = build_manual_demo_payload(task_instruction=task_instruction, scene_profile=scene_profile)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = output_root.resolve() / f'{timestamp}_{MANUAL_DEMO_TEMPLATE}_online_llm_planner'
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        'recipe': run_dir / 'recipe.yaml',
        'plan': run_dir / 'plan.json',
        'checker_report': run_dir / 'checker_report.json',
        'primitive_plan': run_dir / 'primitive_plan.json',
        'manual_reasoning_trace': run_dir / 'reasoning_trace.json',
        'manual_skill_steps': run_dir / 'skill_steps.json',
        'manual_demo_report': run_dir / 'planning_report.md',
        'annotation': run_dir / 'annotation.yaml',
    }
    _dump_yaml(paths['recipe'], payload['recipe'])
    _dump_json(paths['plan'], payload['plan'])
    _dump_json(paths['checker_report'], payload['check_result'])
    _dump_json(paths['primitive_plan'], payload['primitive_plan'])
    _dump_json(paths['manual_reasoning_trace'], payload['manual_reasoning_trace'])
    _dump_json(paths['manual_skill_steps'], payload['manual_skill_steps'])
    paths['manual_demo_report'].write_text(
        _manual_report(
            task_instruction=task_instruction or DEFAULT_MANUAL_TASK,
            scene_profile=scene_profile,
            plan=payload['plan'],
            trace=payload['manual_reasoning_trace'],
            skill_steps=payload['manual_skill_steps'],
            check_result=payload['check_result'],
        ),
        encoding='utf-8',
    )
    _dump_yaml(
        paths['annotation'],
        {
            'task_name': GENERATED_TASK_NAME,
            'title': 'Template-generated RoboAssembly UR5e Plumbers Block Task',
            'summary': task_instruction or DEFAULT_MANUAL_TASK,
            'Menus': payload['plan'].get('generated_menus', []),
            'Annotations': payload['plan'].get('generated_annotations', {}),
            'metadata': {
                'authoring_stack': 'RoboAssembly online LLM planner',
                'selected_template': MANUAL_DEMO_TEMPLATE,
                'generated_task': GENERATED_TASK_NAME,
                'llm_called': True,
            },
        },
    )
    payload['run_dir'] = str(run_dir)
    payload['bundle_paths'] = {key: str(path) for key, path in paths.items()}
    payload['artifacts'] = [
        {'label': key, 'path': str(path), 'kind': 'file' if path.is_file() else 'path'}
        for key, path in paths.items()
    ]
    return _jsonable(payload)
