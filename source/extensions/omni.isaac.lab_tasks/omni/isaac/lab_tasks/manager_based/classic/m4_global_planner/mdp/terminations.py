# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reversed_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: RigidObject = env.scene[robot_cfg.name]
    
    curr_pos_z_w = asset.data.body_state_w[:, 0, 2] # Accessing base_link z in world frame

    # print("Current Z: ", curr_pos_z_w)
    # print("Elevation: ", curr_pos_z_w < 0.20)

    # rewarded if the object is lifted above the threshold
    return curr_pos_z_w < 0.20 # Robot base_link is always above 0.28 so anything below that is reversed

def reached_goal(env: ManagerBasedRLEnv, threshold: float, command_name: str) -> torch.Tensor:
    
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    des_pos_b = command[:, :4] # Includes x, y, z, heading
    error = torch.norm(des_pos_b, dim=1)

    # print("Error: ", error < threshold)

    return error < threshold
    
def lin_speed_limit_reached(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    return torch.abs(asset.data.root_lin_vel_b[:, 0]) > threshold

def ang_speed_limit_reached(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    return torch.abs(asset.data.root_ang_vel_b[:, 2]) > threshold

def elevation_target_out_of_range(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]



    return torch.abs(asset.data.root_ang_vel_b[:, 2]) > threshold