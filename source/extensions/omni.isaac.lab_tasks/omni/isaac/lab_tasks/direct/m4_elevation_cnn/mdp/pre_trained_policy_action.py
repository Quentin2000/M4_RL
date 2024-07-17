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
        self._low_level_action_term_1: ActionTerm = cfg.low_level_actions_1.class_type(cfg.low_level_actions_1, env)
        self._low_level_action_term_2: ActionTerm = cfg.low_level_actions_2.class_type(cfg.low_level_actions_2, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term_1.action_dim + self._low_level_action_term_2.action_dim, device=self.device)

        # print("self._low_level_action_term_1.action_dim :", self._low_level_action_term_1.action_dim)
        # print("self._low_level_action_term_2.action_dim :", self._low_level_action_term_2.action_dim)
        # print("self._raw_actions: ", self._raw_actions)

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions[:, :2]
        cfg.low_level_observations.velocity_commands.params = dict()
        cfg.low_level_observations.elevation_commands.func = lambda dummy_env: self._raw_actions[:, 2].unsqueeze(1)
        cfg.low_level_observations.elevation_commands.params = dict()

        # print("self._raw_actions[:, 2]: ", self._raw_actions[:, 2])
        # print("cfg.low_level_observations.velocity_commands: ", cfg.low_level_observations.velocity_commands.func)
        # print("cfg.low_level_observations.elevation_commands: ", cfg.low_level_observations.elevation_commands.func)
        # print("cfg.low_level_observations: ", cfg.low_level_observations)

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0
        self._counter2 = 0
        self.episode_length = env.max_episode_length

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

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
        self._raw_actions[:, :2] = self.command.get_command("base_velocity")[:, :2]
        self._raw_actions[:, 2] = actions[:, 2]


    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term_1.process_actions(self.low_level_actions[:, :self._low_level_action_term_1.action_dim])
            self._low_level_action_term_2.process_actions(self.low_level_actions[:, self._low_level_action_term_1.action_dim : self._low_level_action_term_1.action_dim+self._low_level_action_term_2.action_dim])
            self._counter = 0
        self._low_level_action_term_1.apply_actions()
        self._low_level_action_term_2.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        pass


    def _debug_vis_callback(self, event):
        # Get the existing translations and orientations if they exist
        pass



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
    low_level_actions_1: ActionTermCfg = MISSING
    low_level_actions_2: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
