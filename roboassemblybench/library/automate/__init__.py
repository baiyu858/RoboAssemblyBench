# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

ASSEMBLY_TASK_ID = "RoboAssemblyBench-AutoMate-Assembly-Direct-v0"
DISASSEMBLY_TASK_ID = "RoboAssemblyBench-AutoMate-Disassembly-Direct-v0"

##
# Register Gym environments.
##

gym.register(
    id=ASSEMBLY_TASK_ID,
    entry_point=f"{__name__}.assembly_env:AssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assembly_env:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id=DISASSEMBLY_TASK_ID,
    entry_point=f"{__name__}.disassembly_env:DisassemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.disassembly_env:DisassemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
