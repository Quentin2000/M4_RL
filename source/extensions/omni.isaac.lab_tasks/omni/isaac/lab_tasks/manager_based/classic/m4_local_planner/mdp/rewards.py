# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat, combine_frame_transforms, quat_error_magnitude, quat_mul, wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# Rewards for Anymal like robots

def apply_actions(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    height_command = env.command_manager.get_command(command_name)[:, 2]

    max_height = 0.32
    # rotation_center_offset_from_max_height = 0.02
    dist_center_rotation_to_leg = 0.10
    diag = math.sqrt(dist_center_rotation_to_leg*dist_center_rotation_to_leg + max_height*max_height)

    # print("Command: ", env.command_manager.get_command(command_name))

    diag_init_ang = math.acos(max_height/diag)
    # print("Diag_init_angle: ", diag_init_ang)

    target = torch.arccos((height_command)/(diag)) - diag_init_ang
    
    target = target.unsqueeze(1).repeat(1, 4)

    # print("Height command: ", height_command)
    # print("Current height: ", asset.data.body_state_w[:, 0, 2])
    # print("Current height root: ", asset.data.root_pos_w[:, 2])

    hip_joint_indices, _ = asset.find_joints(["front_left_hip_joint", "front_right_hip_joint", "rear_left_hip_joint", "rear_right_hip_joint"], preserve_order = True)
    # print("Hip indices", hip_joint_indices)
    # print("Target: ", target)
    # print("Pos: ", asset.data.joint_pos[:,hip_joint_indices])
    asset.set_joint_position_target(target=target, joint_ids=hip_joint_indices)

    return 0.0

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def position_command_error_m4(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)

    distance = command[:, :2]

    # target_vec = command[:, :3] - asset.data.root_pos_w[:, :3]
    # des_pos_b = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), target_vec)
    # distance = des_pos_b[:, :2]

    distance = torch.square(torch.norm(distance, dim=1))

    # print("Distance: ", distance)

    return distance / std

def heading_command_error_m4(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    
    # heading_b = wrap_to_pi(command[:, 3] - asset.data.heading_w)
    heading_b = command[:, 3]
    # print("Heading: ", heading_b)

    reward = torch.square(heading_b)

    return reward / std

# def lin_speed_limit_reached(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
#     asset: RigidObject = env.scene[robot_cfg.name]

#     mask = torch.abs(asset.data.root_lin_vel_b[:, 0]) > threshold

#     reward = mask.float * torch.square(abs(asset.data.root_lin_vel_b[:, 0]) - threshold)

#     return reward

# def ang_speed_limit_reached(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
#     asset: RigidObject = env.scene[robot_cfg.name]

#     mask = torch.abs(asset.data.root_ang_vel_b[:, 2]) > threshold

#     reward = mask.float * torch.square(abs(asset.data.root_ang_vel_b[:, 2]) - threshold)

#     return reward

def ang_speed_limit_reached_log(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    distance_to_max_speed = threshold - torch.abs(asset.data.root_ang_vel_b[:, 2])
    
    relu = torch.nn.ReLU()
    distance_to_max_speed_positive = relu(distance_to_max_speed)
    
    epsilon = 1e-10

    reward = torch.log(distance_to_max_speed_positive + epsilon)

    return reward

def lin_speed_limit_reached_log(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[robot_cfg.name]

    distance_to_max_speed = threshold - torch.abs(asset.data.root_lin_vel_b[:, 0])

    relu = torch.nn.ReLU()
    distance_to_max_speed_positive = relu(distance_to_max_speed)
    
    epsilon = 1e-10
    reward = torch.log(distance_to_max_speed_positive + epsilon)

    return reward

def distance_from_geodesic(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    # print("Env : ", env.scene.env_origins)
    # print("Pose: ", asset.data.root_pos_w)
    # print("Pose2: ", asset.data.default_root_state)

    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, 0, :3] # Accessing base_link state (world frame)

    # Consider only the x and y coordinates for 2D distance
    origin_w_2d = env.scene.env_origins[:, :2] # In world frame
    des_pos_w_2d = des_pos_w[:, :2] # In world frame
    curr_pos_w_2d = curr_pos_w[:, :2] # In world frame

    # Vector of straight line between world origin and target (world frame)
    line_vec = des_pos_w_2d - origin_w_2d
    line_vec_norm = line_vec / torch.norm(line_vec, dim=1, keepdim=True)

    # Vector of current position from world origin (world frame)
    point_vec = curr_pos_w_2d - origin_w_2d
    
    projection = (point_vec * line_vec_norm).sum(dim=1, keepdim=True) * line_vec_norm
    
    perpendicular_vec = point_vec - projection
    
    distance = torch.norm(perpendicular_vec, dim=1)
    
    return distance

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()