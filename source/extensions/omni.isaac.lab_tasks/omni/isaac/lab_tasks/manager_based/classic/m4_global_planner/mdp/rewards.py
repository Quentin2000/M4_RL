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
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat, combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# Rewards for Anymal like robots

def apply_actions(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    # height_command = env.command_manager.get_command(command_name)[:, 2]
    height_command = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, 2]
    lp_actions_raw = env.action_manager.get_term("pre_trained_policy_action_2").raw_actions
    lp_actions_processed = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions
    # print("LP Actions Raw: ", lp_actions_raw)
    # print("LP Actions Processed: ", lp_actions_processed)
    # print("Height command 1:", height_command)

    height_command = torch.clamp(height_command, min=0.25, max=0.32)

    # print("Height command 2:", height_command)

    max_height = 0.32
    # rotation_center_offset_from_max_height = 0.02
    dist_center_rotation_to_leg = 0.10
    diag = math.sqrt(dist_center_rotation_to_leg*dist_center_rotation_to_leg + max_height*max_height)

    # print("Diag: ", diag)

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


def energy_consumption(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # The more we open the hips of the robot, the more power it draws
    # Normal mode:      30W   (elevation at 0.3475 m    -> hip servos at 0 degrees)
    # Crouching mode:   45W   (elevation down to 0.28 m -> hip servos at 36.3 degrees)
    # We assume a linear interpolation between the power consumption and the robot elevation: 
    
    lin_coef = (30-45)/(0.3475-0.28) # [W/m]
    offset_coef = 30 - lin_coef * 0.3475 # [W]

    asset: RigidObject = env.scene[robot_cfg.name]

    current_elevation = asset.data.body_state_w[:, 0, 2] # Accessing base_link state (world frame) 

    return lin_coef * current_elevation + offset_coef - 30 # "- 30" to have 0 penalty when in normal mode


# def energy_consumption_timed(
#     env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:

#     asset: RigidObject = env.scene[robot_cfg.name]

#     # The more we open the hips of the robot, the more power it draws
#     # Normal mode:      30W   (elevation at 0.3475 m    -> hip servos at 0 degrees)
#     # Crouching mode:   45W   (elevation down to 0.28 m -> hip servos at 36.3 degrees)
#     # We assume a linear interpolation between the power consumption and the robot elevation: 
    
#     lin_coef = (30-45)/(0.3475-0.28) # [W/m]
#     offset_coef = 30 - lin_coef * 0.3475 # [W]

#     current_elevation = asset.data.body_state_w[:, 0, 2] # Accessing base_link state (world frame) 

#     current_power_consumption = lin_coef * current_elevation + offset_coef

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

def position_command_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    des_pos_b = command[:, :2]
    reward = 1-torch.exp(torch.norm(des_pos_b, dim=1)/std)
    return reward

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

def heading_command_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    # print("Command: ", command)
    heading_b = command[:, 3]
    reward = 1-torch.exp(abs(heading_b)/std)
    return reward

# def lin_speed_limit_reached(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
#     asset: RigidObject = env.scene[robot_cfg.name]

#     mask = torch.abs(asset.data.root_lin_vel_b[:, 0]) > threshold

#     reward = mask.float

#     return reward

# def ang_speed_limit_reached(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
#     asset: RigidObject = env.scene[robot_cfg.name]

#     mask = torch.abs(asset.data.root_ang_vel_b[:, 2]) > threshold

#     reward = mask.float

#     return reward

def distance_from_geodesic(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    # print("Env : ", env.scene.env_origins)
    # print("Pose: ", asset.data.root_pos_w)
    # print("Pose2: ", asset.data.default_root_state)

    command = env.command_manager.get_command(command_name)
    # Retrieving the desired and current positions
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

def local_planner_action_proximity_log(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    local_planner_target = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, :2] # in robot frame

    # print("local target: ", local_planner_target)
    # print("current pos: ",  current_pos)

    action_norm = torch.norm(local_planner_target, dim=1)

    reward = torch.where(action_norm <= threshold, 0, 1-torch.exp(action_norm - threshold))

    # print("Error: ", action_norm - threshold)
    # print("Reward: ", reward)

    # relu = torch.nn.ReLU()
    # distance_to_max_dist_positive = relu(distance_to_max_dist)
    
    # epsilon = 1e-10
    # reward = torch.log(distance_to_max_dist_positive + epsilon)

    return reward

def move_forward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    reward = 1 - torch.exp(-asset.data.root_lin_vel_b[:, 0] / std)

    return reward

def local_planner_action_forward_log(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    local_planner_target = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, :2] # in robot frame

    # print("local target: ", local_planner_target)
    # print("current pos: ",  current_pos)

    

    distance_to_max_angle = threshold - torch.atan2(local_planner_target[:, 1], local_planner_target[:, 0])

    relu = torch.nn.ReLU()
    distance_to_max_angle_positive = relu(distance_to_max_angle)
    
    epsilon = 1e-10
    reward = torch.log(distance_to_max_angle_positive + epsilon)

    return reward

def local_planner_action_forward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    raw_actions_xy_b = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, :2] # in robot frame

    # print("local target: ", local_planner_target)
    # print("current pos: ",  current_pos)

    reward = torch.square(torch.atan2(raw_actions_xy_b[:, 1], raw_actions_xy_b[:, 0]))

    # print("Forward: ", torch.atan2(raw_actions_xy_b[:, 1], raw_actions_xy_b[:, 0]))
    # print("Forward: ",  abs(torch.atan2(raw_actions_xy_b[:, 1], raw_actions_xy_b[:, 0])) < math.pi/4)
    # print("Reward: ", reward)

    return reward

def local_planner_action_proximity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    raw_actions_xy_b = env.action_manager.get_term("pre_trained_policy_action_2").processed_actions[:, :2] # in robot frame

    # print("local target: ", local_planner_target)
    # print("current pos: ",  current_pos)

    reward = torch.square(torch.norm(raw_actions_xy_b, dim=1))
    # print("Prox: ", torch.norm(raw_actions_xy_b, dim=1))
    # print("Prox: ", torch.norm(raw_actions_xy_b, dim=1) < 0.5)
    
    # print("Reward: ", reward)

    return reward