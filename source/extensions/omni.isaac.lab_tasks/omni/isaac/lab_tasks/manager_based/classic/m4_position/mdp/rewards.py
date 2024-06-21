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
    # print("Joint names", asset.joint_names)
    wheel_joint_indices, _ = asset.find_joints(["front_left_wheel_joint", "front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"], preserve_order = True)
    # hip_joint_indices, _ = asset.find_joints(["front_left_hip_joint", "front_right_hip_joint", "rear_left_hip_joint", "rear_right_hip_joint"], preserve_order = True)

    wheels_velocity_actions = env.action_manager.get_term("joint_vel").processed_actions
    # elevation_actions  = env.action_manager.get_term("elevation").processed_actions

    asset.write_joint_state_to_sim(position=asset.data.joint_pos[:, wheel_joint_indices], velocity=wheels_velocity_actions, joint_ids=wheel_joint_indices)
    # asset.set_joint_velocity_target(target = wheels_velocity_actions, joint_ids = wheel_joint_indices)

    # print("elevation_actions: ", elevation_actions)

    # asset.write_joint_state_to_sim(position=elevation_actions, velocity=asset.data.joint_vel[:, hip_joint_indices], joint_ids=hip_joint_indices)

    # asset.set_joint_position_target(target = elevation_actions, joint_ids = hip_joint_indices)

    return 0.0


def diff_wheels(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joint_indices, _ = asset.find_joints(["front_left_wheel_joint", "front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"], preserve_order = True)
    FL_wheel_speed = asset.data.joint_vel[:, wheel_joint_indices[0]]
    FR_wheel_speed = asset.data.joint_vel[:, wheel_joint_indices[1]]
    RL_wheel_speed = asset.data.joint_vel[:, wheel_joint_indices[2]]
    RR_wheel_speed = asset.data.joint_vel[:, wheel_joint_indices[3]]

    reward = (torch.square(RL_wheel_speed - FL_wheel_speed) + torch.square(RR_wheel_speed - FR_wheel_speed))

    # print("Reward0: ", reward)

    return reward


def position_command_error_tanh_m4(env: ManagerBasedRLEnv, std: float, command_name_xy: str, command_name_z: str)  -> torch.Tensor:
    """Reward position tracking with tanh kernel."""

    command_xy = env.command_manager.get_command(command_name_xy)
    # command_z = env.command_manager.get_command(command_name_z)
    
    # print("XY Command : ", command_xy)
    # print("Z Command : ", command_z)

    des_pos_b_xy = command_xy[:, :2]
    # des_pos_b_z = command_z[:, 0].unsqueeze(1)
    # des_pos_b = torch.cat((des_pos_b_xy, des_pos_b_z), dim=1)

    # print("Elevation: ", command_xy[:, 2])

    distance = torch.norm(des_pos_b_xy, dim=1)

    # print("Distance: ", distance)

    return torch.tanh(distance / std)

def elevation_command_error_tanh_m4(env: ManagerBasedRLEnv, std: float, elevation_command: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))  -> torch.Tensor:
    """Reward position tracking with tanh kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(elevation_command)
    # print("Command: ", command)
    # obtain the desired and current positions
    des_pos_base_link_b = command[:, 2]
    # des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_base_link_w = asset.data.body_state_w[:, 0, 2]  # Base_link elevation
    # print("asset_cfg.body_ids: ", asset_cfg.body_ids)
    # print("Current elevation: ", curr_pos_base_link_w) 
    # print("Desired elevation: ", des_pos_base_link_b)
    return torch.tanh((curr_pos_base_link_w - des_pos_base_link_b) / std)


def heading_command_error_abs_m4(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]

    # print("Heading: ", heading_b)

    return heading_b.abs()


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_ang_vel_b[:, 2])

    # print("Reward1: ", reward)

    return reward

def joint_acc_l2_m4(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_acc), dim=1)
    # print("Acc: ", asset.data.joint_acc)
    # print("Reward2: ", reward)

    return reward


def reverse_movement(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    reward = abs(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]) > abs(env.command_manager.get_command(command_name)[:, 0])

    reward[reward==True] = 1.0
    reward[reward==False] = 0.0

    return reward

def diff_wheels_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    RL_wheel_torque = asset.data.applied_torque[:, 0]
    RR_wheel_torque = asset.data.applied_torque[:, 1]
    FL_wheel_torque = asset.data.applied_torque[:, 2]
    FR_wheel_torque = asset.data.applied_torque[:, 3]

    reward = - (torch.norm(RL_wheel_torque-FL_wheel_torque) + torch.norm(RR_wheel_torque-FR_wheel_torque))
    return reward

def all_wheels_moving(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the velocities of the wheels
    wheel_speeds = asset.data.joint_vel[:, asset_cfg.joint_ids][:, :4]  # Assuming the first 4 joint IDs correspond to the wheels

    # Calculate the differences in velocities between each pair of wheels
    RL_wheel_speed = wheel_speeds[:, 0]  # Rear Left wheel
    RR_wheel_speed = wheel_speeds[:, 1]  # Rear Right wheel
    FL_wheel_speed = wheel_speeds[:, 2]  # Front Left wheel
    FR_wheel_speed = wheel_speeds[:, 3]  # Front Right wheel

    # Calculate the norms of the differences between each pair
    diff_rear = abs(RL_wheel_speed) - abs(RR_wheel_speed)
    diff_front = abs(FL_wheel_speed) - abs(FR_wheel_speed)

    # Reward function that penalizes differences in wheel speeds
    reward = (abs(diff_rear) + abs(diff_front))

    return reward

def balanced_wheels(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the velocities of the wheels
    wheel_speeds = asset.data.joint_vel[:, asset_cfg.joint_ids][:, :4]  # Assuming the first 4 joint IDs correspond to the wheels

    # Calculate the differences in velocities between each pair of wheels
    FL_wheel_speed = wheel_speeds[:, 0]  # Front Left wheel
    FR_wheel_speed = wheel_speeds[:, 1]  # Front Right wheel
    RL_wheel_speed = wheel_speeds[:, 2]  # Rear Left wheel
    RR_wheel_speed = wheel_speeds[:, 3]  # Rear Right wheel
    
    # Calculate the norms of the differences between each pair
    diff_rear = torch.square(abs(RL_wheel_speed) - abs(RR_wheel_speed))
    diff_front = torch.square(abs(FL_wheel_speed) - abs(FR_wheel_speed))

    # Reward function that penalizes differences in wheel speeds
    reward = diff_rear + diff_front

    return reward


def track_lin_vel_xy_m4(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # env.action_manager.get_term("joint_vel").process_actions
    # env.action_manager.get_term("joint_vel").apply_actions
    

    # print("asset.data.root_lin_vel_b: ", asset.data.root_lin_vel_b)
    # print("env.command_manager.get_command(command_name): ", env.command_manager.get_command(command_name))

    # joint_pos = asset.data.joint_pos
    # print("joint_vel: ", joint_vel)
    # asset.write_joint_state_to_sim(joint_pos, env.action_manager.get_term("joint_vel").processed_actions)
    # print("Robot joint vel: ", asset.data.joint_vel)
    # non_zero_command = env.command_manager.get_command(command_name)
    # non_zero_command[non_zero_command == 0.0] = 1.0
    # print("non_zero_command: ", non_zero_command)

    # compute the error
    reward = torch.sum(torch.square((env.command_manager.get_command(command_name) - asset.data.root_lin_vel_b)), dim=1)
    # print("Reward3: ", reward)

    return reward

def track_lin_vel_xy_exp_m4(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    return reward

def track_ang_vel_z_m4(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return ang_vel_error

def non_zero_speed(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.01
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    # print("Command: ", env.command_manager.get_command(command_name))
    # print("Action: ", env.action_manager.action)
    # print("Robot: ", asset.data.root_lin_vel_b[:, :2])

    # joint_pos, joint_vel = asset.data.joint_pos, asset.data.default_joint_vel
    # print("joint_vel: ", joint_vel)
    # asset.write_joint_state_to_sim(joint_pos, joint_vel)

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    # print("Norm command: ", command_norm)
    action_norm = torch.norm(env.action_manager.action, dim=1)
    # print("Norm Action: ", action_norm)
    robot_norm_velocity = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    # print("Norm Robot Velocity: ", robot_norm_velocity)
    indices_of_commands_above_threshold = torch.nonzero(command_norm > threshold, as_tuple=False).unique()
    # print("indices_of_commands_above_threshold ;", indices_of_commands_above_threshold)

    # matching_action_values = torch.norm(env.action_manager.action[indices_of_commands_above_threshold], dim=1)
    # print("matching_action_values ;", matching_action_values)

    wheel_radius = 0.1
    # max_lin_command = 1.0

    indices_of_actions_below_command = torch.nonzero(action_norm < 0.9 * command_norm / wheel_radius, as_tuple=False).unique()
    reward = torch.zeros_like(action_norm)
    reward[indices_of_actions_below_command] = 1.0
    # print("indices_of_actions_below_command: ", indices_of_actions_below_command)
    # print("Reward: ", reward)

    return reward

def all_wheels_moving_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the velocities of the wheels
    wheel_torques = asset.data.applied_torque[:, asset_cfg.joint_ids][:, :4]  # Assuming the first 4 joint IDs correspond to the wheels

    # Calculate the differences in velocities between each pair of wheels
    RL_wheel_torque = wheel_torques[:, 0]  # Rear Left wheel
    RR_wheel_torque = wheel_torques[:, 1]  # Rear Right wheel
    FL_wheel_torque = wheel_torques[:, 2]  # Front Left wheel
    FR_wheel_torque = wheel_torques[:, 3]  # Front Right wheel

    # Calculate the norms of the differences between each pair
    diff_rear = torch.norm(abs(RL_wheel_torque) - abs(RR_wheel_torque))
    diff_front = torch.norm(abs(FL_wheel_torque) - abs(FR_wheel_torque))

    # Reward function that penalizes differences in wheel speeds
    reward = - (diff_rear + diff_front)

    return reward

def all_hips_movement(
    env: ManagerBasedRLEnv, max_joint_pos: float, min_joint_pos: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # print(env.action_manager.get_term(action_cfg))
    # print(env.action_manager.action.shape) # [Num_envs,12], 12 because we the actions are configured to send positions to 4 legs and 4 hips and velocities to 4 wheels
    # print("Action :", env.action_manager.action[0])
    # asset: Articulation = env.scene[asset_cfg.name]
    # print(asset_cfg.joint_ids)
    # print("Position :", asset.data.joint_pos[0, asset_cfg.joint_ids][8:])
    # print("Speed ;", asset.data.joint_vel[0, asset_cfg.joint_ids][8:])
    # print("Torque :", asset.data.applied_torque[0, asset_cfg.joint_ids][8:])

    # print("Position :", asset.data.joint_pos[0, asset_cfg.joint_ids])
    # print("Speed ;", asset.data.joint_vel[0, asset_cfg.joint_ids])
    # print("Torque :", asset.data.applied_torque[0, asset_cfg.joint_ids])

    asset: Articulation = env.scene[asset_cfg.name]
    RL_hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][0]
    RR_hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][1]
    FL_hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][2]
    FR_hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][3]

    differences = [torch.norm(RL_hip_pos-RR_hip_pos), torch.norm(RL_hip_pos-FL_hip_pos), torch.norm(RL_hip_pos-FR_hip_pos), torch.norm(RR_hip_pos-FL_hip_pos), torch.norm(RR_hip_pos-FR_hip_pos), torch.norm(FL_hip_pos-FR_hip_pos)]

    reward = -torch.sum(differences, dim=1)
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

# Rewards for arm like robots

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


# def position_command_error_tanh(
#     env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward tracking of the position using the tanh kernel.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame) and maps it with a tanh kernel.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
#     distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
#     return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

