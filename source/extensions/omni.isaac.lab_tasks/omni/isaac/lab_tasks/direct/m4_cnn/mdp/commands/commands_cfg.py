# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .pose_z_command import UniformPoseZCommand
from .velocity_x_command import UniformVelocityXCommand


@configclass
class UniformPoseZCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseZCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_z: tuple[float, float] = MISSING  # min max [m]

    ranges: Ranges = MISSING
    """Ranges for the commands."""


@configclass
class UniformVelocityXCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityXCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    # heading_command: bool = MISSING
    """Whether to use heading command or angular velocity command.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """
    # heading_control_stiffness: float = MISSING
    """Scale factor to convert the heading error to angular velocity command."""
    # rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    # rel_heading_envs: float = MISSING
    """Probability threshold for environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command)."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        # heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""
