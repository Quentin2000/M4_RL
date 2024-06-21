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

import omni.isaac.lab_tasks.manager_based.classic.m4_local_planner.mdp as mdp
from omni.isaac.lab_tasks.manager_based.classic.m4_velocity.m4_velocity_env_cfg import M4VelocityEnvCfg

##
# Pre-defined configs
##
LOW_LEVEL_ENV_CFG = M4VelocityEnvCfg()

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

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )

    # pose_command = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     # body_name=".*",
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-2.0, 2.0),
    #         pos_y=(-2.0, 2.0),
    #         pos_z=(-0.1, 0.0),
    #         roll=(0.0, 0.0),
    #         pitch=(0.0, 0.0),
    #         yaw=(-3.14, 3.14),
    #     ),
    # )

    # pose_command = mdp.TerrainBasedPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading = False,
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=True,
    #     ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
    #         heading=(-3.14, 3.14),
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint", ".*leg_joint"], scale=1.0, use_default_offset=True)
    # joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0, use_default_offset=False, debug_vis=True)
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"/home/m4/IsaacLab/logs/rsl_rl/m4_velocity/delfosse-rl4/model_1499.pt",
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
    # apply_actions = RewTerm(
    #     func=mdp.apply_actions, weight=1.0, params={"command_name": "base_velocity"}
    # )

    # -- task
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

    # -- penalties
    # reverse_movement = RewTerm(func=mdp.reverse_movement, weight=-10.0, params={"command_name": "base_velocity"})
    # ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-100)
    # differential_wheels = RewTerm(func=mdp.diff_wheels_tanh, weight=-0.01)
    # differential_wheels = RewTerm(func=mdp.diff_wheels_tanh, weight=1.0, params={"std": 2.0})
    # differential_wheels_fine_grained = RewTerm(func=mdp.diff_wheels_tanh, weight=1.0, params={"std": 0.2})
    # balanced_wheels = RewTerm(func=mdp.balanced_wheels, weight=-0.01) // REMOVED BECAUSE CAN'T TURN WHILE MOVING FORWARD IF BOTH SIDES CAN'T HAVE DIFFERENT SPEEDS


    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

##
# Environment configuration
##


@configclass
class M4LocalPlannerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Init online scores monitor
    # wandb.init(project='M4_RL_Velocity', entity='m4')

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
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        # self.sim.disable_contact_processing = True
        # self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # if self.scene.contact_forces is not None:
        #     self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False
