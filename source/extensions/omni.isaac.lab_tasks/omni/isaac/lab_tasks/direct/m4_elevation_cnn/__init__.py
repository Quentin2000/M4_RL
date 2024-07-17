# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
M4 locomotion environment.
"""

import gymnasium as gym

from . import agents
from .m4_elevation_cnn_env_cfg import M4ElevationCnnEnv, M4ElevationCnnEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-M4-Elevation-Cnn-v0",
    entry_point="omni.isaac.lab_tasks.direct.m4_elevation_cnn:M4ElevationCnnEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": M4ElevationCnnEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.M4ElevationCnnPPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
