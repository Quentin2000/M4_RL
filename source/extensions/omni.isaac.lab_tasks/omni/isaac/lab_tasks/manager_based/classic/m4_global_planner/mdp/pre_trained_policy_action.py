# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, yaw_quat, combine_frame_transforms, quat_error_magnitude, quat_mul
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class PreTrainedPolicyAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.command = env.command_manager

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.pose_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.pose_commands.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll2_policy": cfg.low_level_observations}, env)

        self._counter = 0
        self._counter2 = 0
        self.episode_length = env.max_episode_length

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        # target_vec = self.raw_actions_w[:, :3] - self.robot.data.root_pos_w[:, :3]
        # des_pos_b = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        # des_pos_b[:, 2] = self.raw_actions_w[:, 2]
        # self._raw_actions[:, :3] =  des_pos_b
        # self._raw_actions[:, 3] = self.raw_actions_w[:, 3] - self.robot.data.heading_w
        # print("Robot position", self.robot.data.root_pos_w[:, :3])
        # print("Raw actions updated", self._raw_actions)
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # print("Raw actions", self._raw_actions)
        # print("Processed actions", self.raw_actions)
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # raw_actions_pos_w, _ = combine_frame_transforms(self.robot.data.root_state_w[:, :3], self.robot.data.root_state_w[:, 3:7], actions[:, :3])
        # self.raw_actions_w = raw_actions_pos_w
        # self.raw_actions_w[:, 2] = actions[:, 2]
        # raw_actions_rot_w = self.robot.data.heading_w + actions[:, 3]
        # self.raw_actions_w =  torch.cat((self.raw_actions_w[:, :3], raw_actions_rot_w.unsqueeze(1)), dim=1) 
        # print("World actions: ", self.raw_actions_w)

        # print("Raw actions: ", actions)

        # Constraining actions to 0.5 m around the robot
        # norm_actions_xy = torch.norm(actions[:, :2], dim=1)
        # capped_target_xy = torch.where(norm_actions_xy.unsqueeze(1) <= 0.5, actions[:, :2], actions[:, :2] / norm_actions_xy.unsqueeze(1) * 0.5)

        # outward_heading = torch.atan2(actions[:, 1], actions[:, 0])

        # self._raw_actions[:] = torch.cat((capped_target_xy, actions[:, 2].unsqueeze(1), outward_heading.unsqueeze(1)), dim=1)
        self._raw_actions[:] = actions
        # print("Processed actions: ", self._raw_actions)
        


    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll2_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "arrow_goal_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
                marker_cfg.prim_path = "/Visuals/Command/pose_goal"
                self.arrow_goal_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
                marker_cfg.prim_path = "/Visuals/Command/pose"
                self.arrow_pose_visualizer = VisualizationMarkers(marker_cfg)

                # marker_cfg = CUBOID_MARKER_CFG.copy()
                # marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # marker_cfg.prim_path = "/Visuals/Command/front_position"
                # self.robot_front_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.arrow_goal_visualizer.set_visibility(True)
            self.arrow_pose_visualizer.set_visibility(True)
            # self.robot_front_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.arrow_goal_visualizer.set_visibility(False)
                self.arrow_pose_visualizer.set_visibility(False)
                # self.robot_front_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        # Get the existing translations and orientations if they exist
        if hasattr(self, "existing_translations"):
            existing_translations = self.existing_translations
            existing_orientations = self.existing_orientations
            existing_translations_robot = self.existing_translations_robot
            existing_orientations_robot = self.existing_orientations_robot
        else:
            existing_translations = torch.empty((0, 3), device=self.device)
            existing_orientations = torch.empty((0, 4), device=self.device)
            existing_translations_robot = torch.empty((0, 3), device=self.device)
            existing_orientations_robot = torch.empty((0, 4), device=self.device)
        
        # Deleting the markers after episode termination
        if (self._counter2 % self.episode_length == 0) or (torch.norm(self.command.get_command("pose_command"), dim=1) < 0.6755):
            existing_translations = torch.empty((0, 3), device=self.device)
            existing_orientations = torch.empty((0, 4), device=self.device)
            existing_translations_robot = torch.empty((0, 3), device=self.device)
            existing_orientations_robot = torch.empty((0, 4), device=self.device)
            self._counter2 = 0
        self._counter2 += 1
        # print("Command norm: ", torch.norm(self.command.get_command("pose_command"), dim=1))

        # Translations actions in world frame 
        # XY
        raw_actions_xy_b = self.raw_actions[:, :2] # Action is given as offset with respect to the robot base in the world frame
        robot_xy_w = self.robot.data.root_pos_w[:, :2] # Robot base position in world frame
        new_translations_xy_w = raw_actions_xy_b + robot_xy_w # Action in world frame for display
        # Z
        raw_actions_z_from_env_origin = torch.clamp(self.raw_actions[:, 2], min=0.25, max=0.32)
        # Putting it together
        new_translations_w = torch.cat((new_translations_xy_w, raw_actions_z_from_env_origin.unsqueeze(1)), dim=1)

        # Orientations actions in world frame
        new_rotations_xy_w = self.robot.data.heading_w + self.raw_actions[:, 3] # Action given as offset from current robot heading (needs to be in world frame for display)
        # Quaternion
        new_orientations_w = quat_from_euler_xyz(
            torch.zeros_like(self.raw_actions[:, 3]),
            torch.zeros_like(self.raw_actions[:, 3]),
            new_rotations_xy_w,
        )
        
        # Update the stored translations and orientations
        self.existing_translations = torch.cat((existing_translations, new_translations_w), dim=0)
        self.existing_orientations = torch.cat((existing_orientations, new_orientations_w), dim=0)
        self.existing_translations_robot = torch.cat((existing_translations_robot, self.robot.data.root_pos_w), dim=0)
        self.existing_orientations_robot = torch.cat((existing_orientations_robot, self.robot.data.root_quat_w), dim=0)
        
        # print("Raw actions Marker: ", self.raw_actions)

        # Update the visualization
        self.arrow_goal_visualizer.visualize(
            translations=self.existing_translations,
            orientations=self.existing_orientations,
        )
        self.arrow_pose_visualizer.visualize(
            translations=self.existing_translations_robot,
            orientations=self.existing_orientations_robot,
        )
        

        # # Robot front marker
        # x = self.robot.data.root_pos_w[:, 0]
        # y = self.robot.data.root_pos_w[:, 1]
        # heading = self.robot.data.heading_w
        # # Distance in front of the robot
        # distance = 0.3
        # # Computing the offset in x and y directions
        # delta_x = distance * torch.cos(heading)
        # delta_y = distance * torch.sin(heading)
        # # Computing the new position in the world frame
        # marker_x = x + delta_x
        # marker_y = y + delta_y

        # marker_pos_w = torch.stack((marker_x, marker_y), dim=1)
        # marker_pos_w = torch.cat((marker_pos_w, self.robot.data.root_pos_w[:, 2].unsqueeze(1)), dim=1)

        # self.robot_front_visualizer.visualize(marker_pos_w)



    # def _debug_vis_callback(self, event):
    #     # update the box marker

    #     raw_actions_xy_from_env_origin = self.raw_actions[:, :2]
    #     print("Actions: ", raw_actions_xy_from_env_origin)
    #     # env_origin_xy = self.robot.data.root_pos_w[:, :2]
    #     env_origin_xy = self._env.scene.env_origins[:, :2]
    #     # print("Origins: ", env_origin_xy)

    #     new_translations_xy_w = raw_actions_xy_from_env_origin + env_origin_xy

    #     raw_actions_z_from_env_origin = torch.clamp(self.raw_actions[:, 2], min=0.25, max=0.32) 

    #     new_translations_w = torch.cat((new_translations_xy_w, raw_actions_z_from_env_origin.unsqueeze(1)), dim=1)

    #     self.arrow_goal_visualizer.visualize(
    #         translations=new_translations_w,
    #         orientations=quat_from_euler_xyz(
    #             torch.zeros_like(self.raw_actions[:, 3]),
    #             torch.zeros_like(self.raw_actions[:, 3]),
    #             self.raw_actions[:, 3],
    #         ),
    #     )



@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4*10
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
