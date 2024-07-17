# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, DCMotorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import RandomizationTermCfg as RandTerm

import omni.isaac.lab_tasks.manager_based.classic.m4_local_planner.mdp as mdp
from omni.isaac.lab_tasks.manager_based.classic.m4_velocity_elevation.m4_velocity_elevation_env_cfg import M4VelocityElevationEnvCfg

##
# Pre-defined configs
##
LOW_LEVEL_ENV_CFG = M4VelocityElevationEnvCfg()

##
# Scene definition
##

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Markers defined in https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/commands/velocity_command.py
    # Blue is current velocity
    # Green is goal velocity


    # The pos_x and pos_y commands are relative to the robot and should tend to 0. 
    # The pos_z command is directly in world frame, not an offset from current position and will stay at this height
    pose_command = mdp.UniformPose3dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(25.0, 25.0),
        debug_vis=True,
        ranges=mdp.UniformPose3dCommandCfg.Ranges(pos_x=(5.0, 5.5), pos_y=(0.0, 0.0), pos_z=(0.25, 0.32), heading=(-math.pi, math.pi))
    )
    
    # z_command = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="base_link",  # will be set by agent env cfg
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.0001, 0.0001), pos_y=(-0.0001, 0.0001), pos_z=(0.28, 0.3475), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     )
    # )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"/home/m4/IsaacLab/logs/rsl_rl/m4_velocity_elevation/velocity_elevation7/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_vel,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_pose = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_pose_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02))
        pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- apply velocity actions
    apply_actions = RewTerm(
        func=mdp.apply_actions, weight=1.0, params={"command_name": "pose_command"}
    )

    # -- task
    position_tracking = RewTerm(
        func=mdp.position_command_error_m4,
        weight=-0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_m4,
        weight=-0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_m4,
        weight=-0.5,
        params={"std": 1.0, "command_name": "pose_command"},
    )


    # -- penalties
    distance_from_geodesic = RewTerm(
        func=mdp.distance_from_geodesic, weight=-10.0, params={"command_name": "pose_command"}
    )
    # lin_speed_limit_reached = RewTerm(
    #     func=mdp.lin_speed_limit_reached,
    #     weight=-1.0,
    #     params={"threshold": 0.5}
    # )
    # ang_speed_limit_reached = RewTerm(
    #     func=mdp.ang_speed_limit_reached,
    #     weight=-1.0,
    #     params={"threshold": 0.6}
    # )
    # lin_speed_limit_reached = RewTerm(
    #     func=mdp.lin_speed_limit_reached_log,
    #     weight=10.00,
    #     params={"threshold": 0.5}
    # )
    # ang_speed_limit_reached = RewTerm(
    #     func=mdp.ang_speed_limit_reached_log,
    #     weight=10.00,
    #     params={"threshold": 0.6}
    # )

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # reversed_robot = DoneTerm(
    #     func=mdp.reversed_robot
    # )
    # reached_goal = DoneTerm(
    #     func=mdp.reached_goal,
    #     params={"threshold": 0.195, "command_name": "pose_command"} # Allowing for delta_x = 0.1, delta_y = 0.1, delta_z = 0.1, delta_heading = 0.09 (5 degrees error) gives a norm of 0.195
    # )
    # lin_speed_limit_reached = DoneTerm(
    #     func=mdp.lin_speed_limit_reached,
    #     params={"threshold": 0.5}
    # )
    # ang_speed_limit_reached = DoneTerm(
    #     func=mdp.ang_speed_limit_reached,
    #     params={"threshold": 0.6}
    # )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

##
# Environment configuration
##


@configclass
class M4LocalPlannerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10 # actions every 0.2 sec (Explained: self.sim.dt [0.005] * self.decimation [4*10])
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
