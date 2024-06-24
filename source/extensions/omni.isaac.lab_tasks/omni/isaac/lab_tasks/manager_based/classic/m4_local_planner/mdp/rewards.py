# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat, combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# Rewards for Anymal like robots

def apply_actions(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]

    # print("Actions: ", env.action_manager.get_term("pre_trained_policy_action").processed_actions)

    max_height = 0.3475
    # min_height = 0.30
    
    hip_joint_indices, _ = asset.find_joints(["front_left_hip_joint", "front_right_hip_joint", "rear_left_hip_joint", "rear_right_hip_joint"], preserve_order = True)

    height_command = env.command_manager.get_command(command_name)[:, 2]

    # print("Height Command: ", height_command)
    print("Pose: ", asset.data.root_pos_w)

    target = torch.arccos(height_command/max_height)
    target = target.unsqueeze(1).repeat(1, 4)

    asset.set_joint_position_target(target=target, joint_ids=hip_joint_indices)

    return 0.0

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def position_command_error_m4(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    des_pos_b = command[:, :2]
    distance = torch.square(torch.norm(des_pos_b, dim=1))
    return distance / std

def distance_from_geodesic(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    print("Pose: ", asset.data.root_pos_w)
    print("Pose2: ", asset.data.init_state.pos)

    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]

    # Consider only the x and y coordinates for 2D distance
    origin = torch.tensor([0.0, 0.0], device=des_pos_w.device)
    des_pos_w_2d = des_pos_w[:, :2]
    curr_pos_w_2d = curr_pos_w[:, :2]

    # Vector of straight line between world origin and target (world frame)
    line_vec = des_pos_w_2d - origin
    line_vec_norm = line_vec / torch.norm(line_vec, dim=1, keepdim=True)

    # Vector of current position from world origin (world frame)
    point_vec = curr_pos_w_2d - origin
    
    projection = (point_vec * line_vec_norm).sum(dim=1, keepdim=True) * line_vec_norm
    
    perpendicular_vec = point_vec - projection
    
    distance = torch.norm(perpendicular_vec, dim=1)
    
    return distance





def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def heading_command_error_m4(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]

    reward = torch.square(heading_b)
    return reward