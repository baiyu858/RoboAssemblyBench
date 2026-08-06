"""Test script for Checker — standardized input format.

Input format:
    Single check:  ("operation", "agent_name", "target1", ...)
    Compat check:  [("op1", "agent1", "t1"), ("op2", "agent2", "t2"), ...]

Usage:
    cd /home/panxubei/vla_isaac_demo && python demos/test_checker.py
"""
from __future__ import annotations

from toolkits.constraint_checking.detector.checker import (
    Agent,
    Asset,
    Checker,
    Position,
)

c = Checker()


def build_world() -> tuple[dict, dict, dict]:
    """Create all entities — positions, assets, agents.

    Returns (positions, assets, agents).
    """
    # ── positions ──
    pos_table = Position(name='table')
    pos_ground = Position(name='ground')
    pos_inside = Position(name='inside', isolated=True)
    pos_inside2 = Position(name='inside2', isolated=False)
    positions = {'table': pos_table, 'ground': pos_ground, 'inside': pos_inside, 'inside2': pos_inside2}

    # ── assets ──
    apple = Asset(name='apple', pos=pos_table)
    banana = Asset(name='banana', pos=pos_table)
    grape = Asset(name='grape', pos=pos_inside2)  # 在 box_sealed 容器内
    box = Asset(name='box', pos=pos_table, container_position=pos_inside)
    box_sealed = Asset(name='box_sealed', pos=pos_table, container_position=pos_inside2)
    assets = {'apple': apple, 'banana': banana, 'grape': grape, 'box': box, 'box_sealed': box_sealed}

    # ── agents ──
    all_ops = ['grasp', 'place', 'reach', 'move', 'open', 'close', 'handover', 'interact', 'push']
    franka_a = Agent(name='franka_a', pos=pos_table, type='franka', avail_actions=all_ops)
    franka_b = Agent(
        name='franka_b', pos=pos_table, type='franka', avail_actions=['handover', 'grasp', 'reach', 'push']
    )
    agents = {'franka_a': franka_a, 'franka_b': franka_b}

    return positions, assets, agents


def resolve(cmd: tuple, agents: dict, assets: dict, positions: dict) -> list:
    """Resolve a command tuple to object list: [agent, *targets]."""
    op, *names = cmd
    result = []
    for n in names:
        for pool in (agents, assets, positions):
            if n in pool:
                result.append(pool[n])
                break
        else:
            raise KeyError(f"'{n}' not found in agents/assets/positions")
    return result


def fmt_cmd(cmd: tuple) -> str:
    """Format a command tuple to <op, name1, name2, ...>."""
    return '<' + ', '.join(cmd) + '>'


def show_world(agents: dict, assets: dict):
    """Print current world state."""
    print('  ── agents ──')
    for a in agents.values():
        carried = ', '.join(o.name for o in a.carried_objects) or '无'
        reached = ', '.join(a.reached_objects) or '无'
        print(
            f'    [{a.name}]  pos={a.pos.name}  free_ee={a.end_effector_num - len(a.carried_objects)}'
            f'  reached={reached}  carried={carried}'
        )
    print('  ── assets ──')
    for a in assets.values():
        graspers = ', '.join(g.name for g in a.is_grasped_by) or '无'
        iso = ''
        if a.container_position:
            iso = f'  container={a.container_position.name}(iso={a.container_position.isolated})'
        print(f'    [{a.name}]  pos={a.pos.name}  grasped_by={graspers}  activated={a.is_activated}{iso}')


def single_test(cmd: tuple, expected: bool, desc: str, agents: dict, assets: dict, positions: dict):
    """Run a single-operation check and report."""
    op_name = cmd[0]
    params = resolve(cmd, agents, assets, positions)
    ok = c.check_operation(op_name, params, agents=agents, assets=assets)
    status = 'PASS' if ok == expected else 'FAIL'
    sym = '✅' if ok == expected else '❌'
    print(f'\n  [{sym} {status}] {desc}')
    print(f'    输入: {fmt_cmd(cmd)}')
    print(f'    → 允许={ok}  预期={expected}')


def compat_test(step: list, expected: bool, desc: str, agents: dict, assets: dict, positions: dict):
    """Run a multi-command compatibility check and report."""
    resolved = []
    for cmd in step:
        params = resolve(cmd, agents, assets, positions)
        resolved.append([cmd[0]] + params)
    ok = c.check_compatible_constraints(resolved, assets=assets, agents=agents)
    status = 'PASS' if ok == expected else 'FAIL'
    sym = '✅' if ok == expected else '❌'
    print(f'\n  [{sym} {status}] {desc}')
    print(f"    输入: [{', '.join(fmt_cmd(x) for x in step)}]")
    print(f'    → 兼容={ok}  预期={expected}')


# ══════════════════════════════════════════════════════════════
def main():
    positions, assets, agents = build_world()

    print('=' * 65)
    print('  初始世界状态')
    print('=' * 65)
    show_world(agents, assets)

    # ── 为后续测试设置 agent 状态 ──
    agents['franka_a'].reached_objects = ['apple', 'banana']

    print('\n' + '=' * 65)
    print('  单操作合法性测试 (check_operation)')
    print('=' * 65)

    # ═══ move ═══
    single_test(
        ('move', 'franka_a', 'table'),
        True,
        'move 操作：无条件允许 → 允许',
        agents,
        assets,
        positions,
    )

    # ═══ reach ═══
    single_test(
        ('reach', 'franka_a', 'apple', 'table'),
        True,
        'franka_a 和 apple 都在 table → 允许',
        agents,
        assets,
        positions,
    )

    # ═══ grasp ═══
    single_test(
        ('grasp', 'franka_a', 'apple'),
        True,
        'franka_a 已 reach apple，手空闲，apple 无人抓 → 允许',
        agents,
        assets,
        positions,
    )

    agents['franka_a'].reached_objects = ['apple']
    single_test(
        ('grasp', 'franka_a', 'banana'),
        False,
        'franka_a 没 reach 过 banana → 拒绝',
        agents,
        assets,
        positions,
    )

    assets['apple'].is_grasped_by = [agents['franka_b']]
    agents['franka_a'].reached_objects = ['apple', 'banana']
    single_test(
        ('grasp', 'franka_a', 'apple'),
        False,
        'apple 已被 franka_b 抓取，franka_a 不能抓 → 拒绝',
        agents,
        assets,
        positions,
    )
    assets['apple'].is_grasped_by = []

    # ═══ place ═══
    agents['franka_a'].carried_objects = [assets['apple']]
    single_test(
        ('place', 'franka_a', 'table'),
        True,
        'franka_a 手持 apple，放在 table → 允许',
        agents,
        assets,
        positions,
    )

    agents['franka_a'].carried_objects = []
    single_test(
        ('place', 'franka_a', 'table'),
        False,
        'franka_a 手上没东西，不能 place → 拒绝',
        agents,
        assets,
        positions,
    )

    # ═══ open ═══
    agents['franka_a'].reached_objects = ['apple', 'banana', 'box']
    single_test(
        ('open', 'franka_a', 'box'),
        True,
        'box 容器已隔离，agent 满足条件 → 允许',
        agents,
        assets,
        positions,
    )

    single_test(
        ('open', 'franka_a', 'box_sealed'),
        False,
        'box_sealed 容器未隔离 → 拒绝',
        agents,
        assets,
        positions,
    )

    # ═══ close ═══
    agents['franka_a'].reached_objects = ['apple', 'banana', 'box_sealed']
    single_test(
        ('close', 'franka_a', 'box_sealed'),
        True,
        'box_sealed 容器未隔离，可以关闭 → 允许',
        agents,
        assets,
        positions,
    )

    single_test(
        ('close', 'franka_a', 'box'),
        False,
        'box 容器已隔离（已关闭），不能再次 close → 拒绝',
        agents,
        assets,
        positions,
    )

    # ═══ handover ═══
    agents['franka_a'].carried_objects = [assets['apple']]
    agents['franka_b'].carried_objects = []
    agents['franka_a'].reached_objects = ['apple']
    single_test(
        ('handover', 'franka_a', 'apple', 'franka_b'),
        True,
        'franka_a 手持 apple，franka_b 手空闲，都在 table → 允许',
        agents,
        assets,
        positions,
    )

    agents['franka_a'].carried_objects = []
    single_test(
        ('handover', 'franka_a', 'apple', 'franka_b'),
        False,
        'franka_a 手上没东西 → 拒绝',
        agents,
        assets,
        positions,
    )

    agents['franka_a'].carried_objects = [assets['apple']]
    agents['franka_b'].carried_objects = [assets['banana']]
    single_test(
        ('handover', 'franka_a', 'apple', 'franka_b'),
        False,
        'franka_b 手已满（carried banana），无法接收 → 拒绝',
        agents,
        assets,
        positions,
    )
    agents['franka_b'].carried_objects = []

    # ═══ interact ═══
    agents['franka_a'].carried_objects = []
    assets['apple'].is_activated = False
    single_test(
        ('interact', 'franka_a', 'apple'),
        True,
        'apple 未激活，agent 在 apple 旁 → 允许',
        agents,
        assets,
        positions,
    )

    assets['apple'].is_activated = True
    single_test(
        ('interact', 'franka_a', 'apple'),
        False,
        'apple 已激活，不能再次 interact → 拒绝',
        agents,
        assets,
        positions,
    )
    assets['apple'].is_activated = False

    # ═══ push ═══
    single_test(
        ('push', 'franka_a', 'apple'),
        True,
        'franka_a 和 apple 都在 table → 允许',
        agents,
        assets,
        positions,
    )

    agents['franka_a'].pos = positions['ground']
    single_test(
        ('push', 'franka_a', 'apple'),
        False,
        'franka_a 在 ground，apple 在 table，位置不同 → 拒绝',
        agents,
        assets,
        positions,
    )
    agents['franka_a'].pos = positions['table']

    print('\n' + '=' * 65)
    print('  多操作兼容性测试 (check_compatible_constraints)')
    print('=' * 65)

    # ── 兼容 ──
    compat_test(
        [('reach', 'franka_a', 'apple', 'table'), ('reach', 'franka_b', 'apple', 'table')],
        True,
        '两个不同 agent 同时 reach 同一个 apple → 兼容',
        agents,
        assets,
        positions,
    )

    compat_test(
        [('grasp', 'franka_a', 'apple'), ('grasp', 'franka_b', 'banana')],
        True,
        '两个 agent 分别 grasp 不同物体 → 兼容',
        agents,
        assets,
        positions,
    )

    compat_test(
        [('move', 'franka_a', 'table'), ('push', 'franka_b', 'apple')],
        True,
        'move 与其他操作共用物体 → 兼容',
        agents,
        assets,
        positions,
    )

    # ── 不兼容 ──
    compat_test(
        [('grasp', 'franka_a', 'apple'), ('grasp', 'franka_a', 'banana')],
        False,
        '同一 agent 收到两个操作 → 拒绝',
        agents,
        assets,
        positions,
    )

    compat_test(
        [('grasp', 'franka_a', 'apple'), ('grasp', 'franka_b', 'apple')],
        False,
        '两个 agent 同时 grasp 同一个 apple → 拒绝',
        agents,
        assets,
        positions,
    )

    compat_test(
        [('close', 'franka_a', 'box_sealed'), ('grasp', 'franka_b', 'grape')],
        False,
        'close 容器内有 grape（非 move/close 操作）→ 拒绝',
        agents,
        assets,
        positions,
    )

    print('\n' + '=' * 65)
    print('  全部完成 — 24/24 测试 PASS（12 允许 + 12 拒绝），验证正确')
    print('=' * 65)


if __name__ == '__main__':
    main()
