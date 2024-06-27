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
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils.math import quat_from_euler_xyz
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
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
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
            # set their visibility to true
            self.arrow_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.arrow_goal_visualizer.set_visibility(False)


    # def _debug_vis_callback(self, event):
    #     # Get the existing translations and orientations if they exist
    #     if hasattr(self, "existing_translations"):
    #         existing_translations = self.existing_translations
    #         existing_orientations = self.existing_orientations
    #     else:
    #         existing_translations = torch.empty((0, 3), device=self.device)
    #         existing_orientations = torch.empty((0, 4), device=self.device)
        
    #     # New translations and orientations
    #     raw_actions_xy_from_env_origin = self.raw_actions[:, :2]
    #     env_origin_xy = self.robot.data.root_pos_w[:, :2]

    #     new_translations_xy_w = raw_actions_xy_from_env_origin + env_origin_xy

    #     new_translations_w = torch.cat((new_translations_xy_w, self.raw_actions[:, 2].unsqueeze(1)), dim=1)

    #     # new_translations_w = self.raw_actions[:, :3]
    #     new_orientations = quat_from_euler_xyz(
    #         torch.zeros_like(self.raw_actions[:, 3]),
    #         torch.zeros_like(self.raw_actions[:, 3]),
    #         self.raw_actions[:, 3],
    #     )
        
    #     # Concatenate existing and new translations and orientations
    #     all_translations = torch.cat((existing_translations, new_translations_w), dim=0)
    #     all_orientations = torch.cat((existing_orientations, new_orientations), dim=0)
        
    #     # Update the stored translations and orientations
    #     self.existing_translations = all_translations
    #     self.existing_orientations = all_orientations
        
    #     # Update the visualization
    #     self.arrow_goal_visualizer.visualize(
    #         translations=all_translations,
    #         orientations=all_orientations,
    #     )


    def _debug_vis_callback(self, event):
        # update the box marker

        raw_actions_xy_from_env_origin = self.raw_actions[:, :2]
        # env_origin_xy = self.robot.data.root_pos_w[:, :2]
        env_origin_xy = self._env.scene.env_origins[:, :2]
        # print("Origins: ", env_origin_xy)

        new_translations_xy_w = raw_actions_xy_from_env_origin + env_origin_xy

        new_translations_w = torch.cat((new_translations_xy_w, self.raw_actions[:, 2].unsqueeze(1)), dim=1)

        self.arrow_goal_visualizer.visualize(
            translations=new_translations_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.raw_actions[:, 3]),
                torch.zeros_like(self.raw_actions[:, 3]),
                self.raw_actions[:, 3],
            ),
        )



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
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
