# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .pose_3d_command import UniformPose3dCommand
from .pose_z_command import UniformPoseZCommand


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
class UniformPose3dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformPose3dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
