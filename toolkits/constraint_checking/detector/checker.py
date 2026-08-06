"""Task‑planning action sequence checker.

Validates whether an operation (grasp, place, move, …) is feasible given the
current state of agents, assets and positions.  Used to verify action sequences
before execution in simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Domain classes (stubs — extend as needed for your scene representation)
# ---------------------------------------------------------------------------


@dataclass
class Position:
    name: str
    isolated: bool = False


@dataclass
class Asset:
    name: str
    pos: Position
    type: str = 'asset'
    is_activated: bool = False
    is_grasped_by: List['Agent'] = field(default_factory=list)
    container_position: Optional[Position] = None


@dataclass
class Agent:
    name: str
    pos: Position
    type: str = 'franka'
    end_effector_num: int = 1
    carried_objects: List[Asset] = field(default_factory=list)
    reached_objects: List[str] = field(default_factory=list)
    avail_actions: List[str] = field(default_factory=list)

    def is_reached_objects(self, asset: Asset) -> bool:
        return asset.name in self.reached_objects

    def get_carried_objects(self) -> List[Asset]:
        return self.carried_objects

    def get_reached_objects(self) -> List[str]:
        return self.reached_objects


@dataclass
class Action:
    name: str
    param_types: List[type]
    param_scopes: Optional[List[Dict[str, list]]] = None


# ---------------------------------------------------------------------------
# Operation registry (extend with your task domain)
# ---------------------------------------------------------------------------

# Which operations each agent type can perform
AGENT_AVAIL_ACTIONS: Dict[str, List[str]] = {
    'franka': ['move', 'reach', 'grasp', 'place', 'open', 'close', 'handover', 'interact', 'push'],
    'unitree_go2': ['move'],
    'anymal_c': ['move'],
}

# Action definitions — param_types **without** the agent (agent is params[0]
# but not part of the type‑check in check_action_target).
ALL_ACTIONS: Dict[str, Action] = {
    'move': Action('move', [Position]),
    'reach': Action('reach', [Asset, Position]),
    'grasp': Action('grasp', [Asset]),
    'place': Action('place', [Union[Asset, Position]]),
    'open': Action('open', [Asset]),
    'close': Action('close', [Asset]),
    'handover': Action('handover', [Asset, Agent]),
    'interact': Action('interact', [Asset]),
    'push': Action('push', [Asset]),
}


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class Checker:
    """Validate individual operations and multi‑agent constraint compatibility."""

    # ── atomic checks ──────────────────────────────────────────────────

    def check_agent_has_free_end_effector(self, agent: Agent) -> bool:
        return agent.end_effector_num - len(agent.carried_objects) > 0

    def check_asset_is_activated(self, asset: Asset) -> bool:
        return asset.is_activated

    def check_asset_pos(self, asset: Asset, pos: Position) -> bool:
        return asset.pos.name == pos.name

    def check_pos_is_isolated(self, pos: Position) -> bool:
        return pos.isolated

    def check_asset_is_grasped(self, asset: Asset) -> bool:
        return len(asset.is_grasped_by) > 0

    def check_asset_is_reached(self, asset: Asset, agent: Agent) -> bool:
        return asset.name in agent.reached_objects

    def check_agent_action(self, agent: Agent, action: Action) -> bool:
        return action.name in agent.avail_actions

    def check_action_target(self, action: Action, target: list) -> bool:
        if len(target) != len(action.param_types):
            return False
        for t, p in zip(target, action.param_types):
            origin = getattr(p, '__origin__', None)
            if origin is Union:
                if type(t) not in getattr(p, '__args__', ()):
                    return False
            elif not isinstance(t, p):
                return False
        if action.param_scopes is not None:
            for t, scope in zip(target, action.param_scopes):
                for k, value_set in scope.items():
                    if getattr(t, k) not in value_set:
                        return False
        return True

    def check_target_aligned_position(
        self,
        target: Union[Agent, Asset, Position],
        pos: Position,
        assets: dict = None,
        agents: dict = None,
        finished: list = None,
    ) -> bool:
        """Recursively check whether *target* (or what it sits on) is at *pos*."""
        if not finished:
            finished = []
        if not assets:
            assets = {}
        if not agents:
            agents = {}

        if target.pos.name in assets:
            if target.pos.name in finished:
                return False
            finished.append(target.pos.name)
            return (
                self.check_target_aligned_position(assets[target.pos.name], pos, assets, agents, finished)
                or target.pos.name == pos.name
            )
        elif target.pos.name in agents:
            if target.pos.name in finished:
                return False
            finished.append(target.pos.name)
            return (
                self.check_target_aligned_position(agents[target.pos.name], pos, assets, agents, finished)
                or target.pos.name == pos.name
            )
        if isinstance(target, Position):
            return target.name == pos.name
        return target.pos.name == pos.name or target.name == pos.name

    def check_agent_relative_position(self, agent: Agent, target: Union[Agent, Asset]) -> bool:
        return agent.pos.name == target.name or agent.name == target.pos.name or agent.pos.name == target.pos.name

    # ── operation‑level check ──────────────────────────────────────────

    def check_operation(
        self,
        operation_name: str,
        params: list,
        assets: dict = None,
        agents: dict = None,
    ) -> bool:
        if not assets:
            assets = {}
        if not agents:
            agents = {}

        agent_type = params[0].type
        if operation_name not in AGENT_AVAIL_ACTIONS.get(agent_type, []):
            return False

        action_type = ALL_ACTIONS[operation_name]
        if not self.check_action_target(action_type, params[1:]):
            return False

        if operation_name == 'move':
            return True

        elif operation_name == 'reach':
            return (
                self.check_target_aligned_position(params[0], params[1].pos, assets, agents)
                or self.check_target_aligned_position(params[1], params[0].pos, assets, agents)
            ) and not params[1].pos.isolated

        elif operation_name == 'grasp':
            return (
                not self.check_asset_is_grasped(params[1])
                and self.check_agent_has_free_end_effector(params[0])
                and params[0].is_reached_objects(params[1])
            )

        elif operation_name == 'place':
            if isinstance(params[1], Asset):
                is_available_position = self.check_target_aligned_position(
                    params[0], params[1].pos, assets, agents
                ) or self.check_target_aligned_position(params[1], params[0].pos, assets, agents)
                if hasattr(params[1], 'container_position'):
                    is_available_position = is_available_position and not self.check_pos_is_isolated(
                        params[1].container_position
                    )
                return is_available_position and len(params[0].get_carried_objects()) > 0
            else:
                return (
                    self.check_target_aligned_position(params[0], params[1], assets, agents)
                    and len(params[0].get_carried_objects()) > 0
                )

        elif operation_name == 'open':
            agent_status = self.check_agent_relative_position(
                params[0], params[1]
            ) and self.check_agent_has_free_end_effector(params[0])
            position_status = (
                hasattr(params[1], 'container_position')
                and self.check_pos_is_isolated(params[1].container_position)
                and params[1].name in params[0].get_reached_objects()
            )
            return agent_status and position_status

        elif operation_name == 'close':
            agent_status = self.check_agent_relative_position(
                params[0], params[1]
            ) and self.check_agent_has_free_end_effector(params[0])
            position_status = (
                hasattr(params[1], 'container_position')
                and not self.check_pos_is_isolated(params[1].container_position)
                and params[1].name in params[0].get_reached_objects()
            )
            return agent_status and position_status

        elif operation_name == 'handover':
            return (
                self.check_agent_relative_position(params[0], params[2])
                and len(params[0].get_carried_objects()) > 0
                and self.check_agent_has_free_end_effector(params[2])
            )

        elif operation_name == 'interact':
            if (
                params[0].type not in ('unitree_go2', 'anymal_c')
                and params[1] not in params[0].get_carried_objects()
                and not self.check_agent_has_free_end_effector(params[0])
            ):
                return False
            return self.check_agent_relative_position(params[0], params[1]) and not self.check_asset_is_activated(
                params[1]
            )

        elif operation_name == 'push':
            return self.check_agent_relative_position(params[0], params[1])

        else:
            raise ValueError(f'Unexpected operation: {operation_name}.')

    # ── multi‑agent compatibility ──────────────────────────────────────

    def check_compatible_paired_actions(self, command_x: str, command_y: str) -> bool:
        """Return True when two operations can safely target the same object."""
        if 'move' in (command_x, command_y):
            return True
        if command_x in ('reach', 'place') and command_y in ('reach', 'place'):
            return True
        return False

    def check_compatible_constraints(
        self,
        step_commands: list,
        assets: dict = None,
        agents: dict = None,
    ) -> bool:
        """Validate that all commands in a single time‑step are compatible."""
        if not assets:
            assets = {}
        if not agents:
            agents = {}

        commands = [c[0] for c in step_commands if c]
        params = [c[1:] for c in step_commands if c]

        # One agent cannot receive two commands in the same step
        target_agents = [p[0].name for p in params]
        if len(target_agents) != len(set(target_agents)):
            return False

        # Objects shared by multiple commands must support paired ops
        target_entities: Dict[str, list] = {}
        for idx, inst_params in enumerate(params):
            for param in inst_params:
                if param.name in assets:
                    target_entities.setdefault(param.name, []).append(idx)

        for asset_name, inst_indices in target_entities.items():
            if len(inst_indices) < 2:
                continue
            op_names = [commands[i] for i in inst_indices]
            for op1, op2 in combinations(op_names, 2):
                if not self.check_compatible_paired_actions(op1, op2):
                    return False

        # close + any non‑move/non‑close in the same container → conflict
        if 'close' in commands:
            target_container = params[commands.index('close')][1]
            for idx, inst_params in enumerate(params):
                for param in inst_params:
                    if (
                        isinstance(param, Asset)
                        and param.pos == target_container.container_position
                        and commands[idx] not in ('move', 'close')
                    ):
                        return False
        return True
