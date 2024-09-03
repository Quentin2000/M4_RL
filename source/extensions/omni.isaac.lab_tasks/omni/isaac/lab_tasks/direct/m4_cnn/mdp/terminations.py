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


def reversed_robot(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[robot_cfg.name]

    return asset.data.projected_gravity_b[:, 2] > 0.0

def reached_goal(env: ManagerBasedRLEnv, threshold: float, command_name: str) -> torch.Tensor:
    
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    des_pos_b = command[:, :4] # Includes x, y, z, heading
    error = torch.norm(des_pos_b, dim=1)

    # print("Error: ", error < threshold)

    return error < threshold
    
def lin_speed_limit_reached_term(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    return torch.abs(asset.data.root_lin_vel_b[:, 0]) > threshold

def lin_speed_limit_reached_full_term(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    return torch.norm(asset.data.root_lin_vel_b, dim=1) > threshold

def ang_speed_limit_reached_full_term(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    return torch.norm(asset.data.root_ang_vel_b, dim=1) > threshold

def ang_speed_limit_reached_term(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    return torch.abs(asset.data.root_ang_vel_b[:, 2]) > threshold

def elevation_target_out_of_range(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]



    return torch.abs(asset.data.root_ang_vel_b[:, 2]) > threshold

def local_planner_action_forward_termination(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    raw_actions_xy_b = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, :2] # in robot frame

    # print("local target: ", local_planner_target)
    # print("current pos: ",  current_pos)

    reward = abs(torch.atan2(raw_actions_xy_b[:, 1], raw_actions_xy_b[:, 0]))  > threshold
    # print("Reward: ", reward)

    return reward

def local_planner_action_proximity_termination(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    raw_actions_xy_b = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, :2] # in robot frame

    # print("local target: ", local_planner_target)
    # print("current pos: ",  current_pos)

    reward = torch.norm(raw_actions_xy_b, dim=1) > threshold
    # print("Reward: ", reward)

    return reward

def out_of_bounds(env: ManagerBasedRLEnv, x: float, y: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    # print("Pos: ", asset.data.root_pos_w[:, :2])

    result = ((asset.data.root_pos_w[:, 0] > x/2) | (asset.data.root_pos_w[:, 0] < -x/2) | (asset.data.root_pos_w[:, 1] > y/2) | (asset.data.root_pos_w[:, 1] < -y/2))

    return result