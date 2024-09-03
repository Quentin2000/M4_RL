# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import wandb
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, DCMotorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, CameraCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.classic.m4_velocity_elevation.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.m4_terrain import M4_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/m4/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Caltech/m4_fixed_blade_leg.usd",
            activate_contact_sensors = True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                # rigid_body_enabled=True,
                retain_accelerations=False,
                # linear_damping=0.0,
                # angular_damping=0.0,
                # max_linear_velocity=100.0,
                # max_angular_velocity=100.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                # fix_root_link=True,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.40), 
            joint_pos={
                ".*hip_joint": 0.0,
                # ".*leg_joint": 0.0,
                ".*wheel_joint": 0.0, 
            },
            joint_vel={
                ".*hip_joint": 0.0,
                # "rear_right_wheel_joint": 0.0,
                # "rear_left_wheel_joint": 5.0,
                # "front_right_wheel_joint": 10.0,
                # "front_left_wheel_joint": 15.0,
                ".*wheel_joint": 0.0,
            },
        ),
        actuators={
            "hip_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_joint"],
                effort_limit=80.0,
                velocity_limit=2.0,
                stiffness= 80.0,
                damping= 4.0,
            ),
            # "leg_motors": ImplicitActuatorCfg(
            #     joint_names_expr=[".*leg_joint"],
            #     effort_limit=80.0,
            #     velocity_limit=2.0,
            #     stiffness= 80.0,
            #     damping= 4.0,
            # ),
            "wheel_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                # saturation_effort=12000, #120
                # effort_limit=70, #40      Torque constant * max A [N-m]
                velocity_limit=0.00001, #10    KV/(V*reduction)
                stiffness=0.0, #10000
                damping=4.0,
            ),
        },
    )

    # See: https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link/camera",
    #     update_period=0.1,
    #     height=5,
    #     width=8,
    #     data_types=["distance_to_camera"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=0.193, focus_distance=400.0, horizontal_aperture=2.00, clipping_range=(0.4, 6.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.24099, 0.0, -0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # See: https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf
    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link/camera",
    #     update_period=0.1,
    #     height=8,
    #     width=9,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=0.193, focus_distance=400.0, horizontal_aperture=3.66, clipping_range=(0.1, 4.0)
    #     ),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.24099, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)
    # lights 

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Markers defined in https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/commands/velocity_command.py
    # Blue is current velocity
    # Green is goal velocity

    base_velocity = mdp.UniformVelocityXCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        # rel_standing_envs=0.02,
        # rel_heading_envs=1.0,
        # heading_command=True,
        # heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), ang_vel_z=(-0.3, 0.3)
        ),
    )

    z_command = mdp.UniformPoseZCommandCfg(
        asset_name="robot",
        body_name="base_link",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseZCommandCfg.Ranges(
            pos_z=(0.25, 0.32)
        )
    )

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.0001, 0.0001), ang_vel_z=(-0.3, 0.3), heading=(-math.pi, math.pi), 
    #     ),
    # )

    # z_command = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="base_link",  # will be set by agent env cfg
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.0001, 0.0001), pos_y=(-0.0001, 0.0001), pos_z=(0.28, 0.3475), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*wheel_joint"], scale=5.0, use_default_offset=False, debug_vis=False)
    # ONLY UNCOMMENT FOLLOWING LINE IF POLICY USED BY M4_ELEVATION POLICY
    hip_pos = mdp.HipJointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint"], scale=1.0, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pose_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # ONLY UNCOMMENT FOLLOWING LINE IF POLICY USED BY M4_ELEVATION POLICY
        elevation_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "z_command"})
        
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
    apply_actions = RewTerm(
        func=mdp.apply_actions, weight=1.0, params={"command_name": "z_command"}
    )

    # -- task
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_m4, weight=-20.0, params={"command_name": "base_velocity"}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_m4, weight=-20.0, params={"command_name": "base_velocity"}
    )

    # -- penalties
    differential_wheels = RewTerm(func=mdp.diff_wheels, weight=-0.01, params={"std": 1.0})

    track_elevation = RewTerm(
        func=mdp.track_elevation, weight=-100.0, params={"elevation_command": "z_command", "std": 0.5}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass

##
# Environment configuration
##


@configclass
class M4VelocityElevationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    viewer = ViewerCfg(resolution=(16, 9))

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # actions every 0.02 sec (self.sim.dt * self.decimation)
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # if self.scene.camera is not None:
        #     self.scene.camera.update_period = self.decimation * self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt