# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
M4 locomotion environment.
"""

import gymnasium as gym

from . import agents, m4_position_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-M4-Position-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": m4_position_env_cfg.M4PositionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.M4PositionPPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
