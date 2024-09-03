# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from collections.abc import Sequence
from PIL import Image

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, Camera, CameraCfg, ContactSensorCfg, ContactSensor
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainImporter
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab.terrains.config.m4_terrain import M4_TERRAINS_CFG

output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        noised_image = tensor + noise
        clipped_noised_image = torch.clamp(noised_image, min=0, max=1)  # Clamp values to keep them between 0 and 1
        return clipped_noised_image
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

def randomize_rigid_body_material(robot, static_friction_range, dynamic_friction_range, restitution_range):
    # You would typically fetch and modify physics properties here
    robot.modify_physics_material(static_friction=static_friction_range[0], dynamic_friction=dynamic_friction_range[0], restitution=restitution_range[0])

def randomize_mass(robot, mass_distribution_params):
    # Modify mass
    robot.modify_mass(additional_mass=mass_distribution_params[0])

@configclass
class M4CnnEnvCfg(DirectRLEnvCfg):

    # Env
    decimation = 10
    episode_length_s = 20.0 # 40

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        # render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Robot
    robot_cfg: robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                # fix_root_link=True,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*hip_joint": 0.0,
                # ".*leg_joint": 0.0,
                ".*wheel_joint": 0.0, 
            },
            joint_vel={
                ".*hip_joint": 0.0,
                # ".*leg_joint": 0.0,
                ".*wheel_joint": 3.0,
            },
        ),
        actuators={
            "hip_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_joint"],
                effort_limit=100000.0,
                velocity_limit=4.0,
                stiffness= 80.0,
                damping= 1.0,
            ),
            # "leg_motors": ImplicitActuatorCfg(
            #     joint_names_expr=[".*leg_joint"],
            #     effort_limit=100000.0,
            #     velocity_limit=4.0,
            #     stiffness= 4.0,
            #     damping= 0.5,
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

    # # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        num_envs=2048,
        env_spacing=3.0,
        max_init_terrain_level=5,
        terrain_type="generator",
        # terrain_generator=M4_TERRAINS_CFG.replace(color_scheme="random"),
        terrain_generator=M4_TERRAINS_CFG,
        visual_material=None,
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #     project_uvw=True,
        #     texture_scale=(0.25, 0.25),
        # ),
    )

    # Ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="usd",
    #     usd_path=f"/home/m4/IsaacLab/terrains/parkour_2/USDC/parkour_2.usdc",
    #     collision_group=-1,
    #     num_envs=2048,
    #     env_spacing=3.0,
    #     max_init_terrain_level=5,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     visual_material=None,
    # )

    # See: https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf
    camera2 = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/camera2",
        update_period=0.1,
        height=72,
        width=72,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.193, focus_distance=0.60, horizontal_aperture=1.0, clipping_range=(0.01, 6.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.24099, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    camera3 = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/camera3",
        update_period=0.1,
        height=72,
        width=72,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.193, focus_distance=0.60, horizontal_aperture=1.0, clipping_range=(0.01, 6.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.24099, 0.0, 0.0), rot=(-0.5, 0.5, 0.5, -0.5), convention="ros"),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=3, 
        track_air_time=True)

    # Viewer settings
    viewer = ViewerCfg(eye=(2.0, 2.0, 5.0))

    # Env & architecture parameters
    dual_camera = True
    resnet18 = True #### FORGET TO REMOVE CNN LAYERS FROM rl_games_ppo_cfg.yaml
    rand_fc_layer = True

    # Actions
    # num_actions can take the following:
    # 1 [all hips controlled by 1 same action]
    # 2 [front and rear hips controlled by 1 action]
    # 4 [hips controlled separately by 4 actions] 
    # 8 [hips & legs controlled separately by 8 actions]
    num_actions = 4 # DON"T FORGET TO CHANGE ROBOT USD PATH
    action_scale = 1  # [N]

    # Observations
    # num_velocity_targets can take the following:
    # 2 [Vx, Wz]
    # 0 [only camera based]
    num_velocity_targets = 0 # Vx, Wz
    num_observed_joints = 4

    if resnet18:
        if dual_camera:
            num_channels = 2
        else:
            num_channels = 1
        if rand_fc_layer:
            resnet18_output = 32
        else:
            resnet18_output = 512
        num_observations = num_channels * resnet18_output + num_velocity_targets + num_observed_joints # At output of the Resnet18 pipeline
    else:
        if dual_camera:
            num_channels = 2
        else:
            num_channels = 1
        num_observations = num_channels * camera2.height * camera2.width
    num_states = 0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=20.0, replicate_physics=True) 


class M4CnnEnv(DirectRLEnv):

    cfg: M4CnnEnvCfg

    def __init__(
        self, cfg: M4CnnEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # torch.set_printoptions(threshold=10000)

        if self.resnet18:
            # Load the pretrained ResNet-18 model
            self.resnet18_model = resnet18(pretrained=True)
            # Remove the final classification layer to get feature extraction layer
            self.resnet18_model = nn.Sequential(*list(self.resnet18_model.children())[:-1])
            self.resnet18_model = self.resnet18_model.to(self.sim.device)
            self.resnet18_model.eval()

            # Define the preprocessing transformations
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),  # Convert to PIL Image
                transforms.Resize(224),   # Resize to 224x224
                transforms.ToTensor(),    # Convert to tensor
                AddGaussianNoise(0.0, 0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
            ])

        if self.rand_fc_layer:
            # Initialize FC layer
            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 128),  # Adjust the input features size
                nn.ReLU(),
                nn.Linear(128, 32),  # Second FC layer
                nn.ReLU()
            )
            self.initialize_weights(self.fc_layer)
            self.fc_layer.to(self.sim.device)

            # Optionally save the initial weights
            self.save_weights(self.fc_layer, '/home/m4/IsaacLab/logs/rl_games/m4_cnn/initial_fc_weights2.pth')

            # Load weights (for ensuring consistency)
            self.load_weights(self.fc_layer, '/home/m4/IsaacLab/logs/rl_games/m4_cnn/initial_fc_weights.pth')

            self.fc_layer.eval()
        
        m4_base_body_idx_list, _ = self._m4_robot.find_bodies(".*base_link")
        m4_hip_body_idx_list, _ = self._m4_robot.find_bodies(".*hip")
        m4_wheel_body_idx_list, _ = self._m4_robot.find_bodies(".*wheel")
        m4_leg_body_idx_list, _ = self._m4_robot.find_bodies(".*leg")
        self._m4_base_body_idx = torch.tensor(m4_base_body_idx_list)
        self._m4_hip_body_idx = torch.tensor(m4_hip_body_idx_list)
        self._m4_wheel_body_idx = torch.tensor(m4_wheel_body_idx_list)
        self._m4_leg_body_idx = torch.tensor(m4_leg_body_idx_list)   

        self._m4_hip_idx_list, names = self._m4_robot.find_joints(".*hip_joint")
        # print("self._m4_hip_idx_list, names: ", self._m4_hip_idx_list, names)
        self._m4_hip_idx = torch.tensor(self._m4_hip_idx_list)
        self._m4_wheel_idx_list, _ = self._m4_robot.find_joints(".*wheel_joint")
        self._m4_wheel_idx = torch.tensor(self._m4_wheel_idx_list) 

        self.episode_count = 1

        if self.num_actions == 8: 
            self._m4_leg_idx_list, _ = self._m4_robot.find_joints(".*leg_joint")
            self._m4_leg_idx = torch.tensor(self._m4_leg_idx_list)
        
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self._m4_robot.data.joint_pos
        self.joint_vel = self._m4_robot.data.joint_vel
        self.target_vel = torch.zeros((self.num_envs, 2), device=self.sim.device) # (Vx, Wz)
        self.target_wheel_vel = torch.zeros((self.num_envs, 4), device=self.sim.device)
        self.root_lin_vel_b = self._m4_robot.data.root_lin_vel_b
        self.root_pos_w = self._m4_robot.data.root_pos_w
        self.projected_gravity_b = self._m4_robot.data.projected_gravity_b
        self.contacts = self._contact_forces.data.net_forces_w_history

        self.grid_dim = 5 # Number of rows or columns in the map, given that the map is a square
        self.grid_size = 5 # Size of the side of a grid in meters
        self.grid_centers = []

        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                x = (j - (self.grid_dim - 1) / 2) * self.grid_size
                y = (i - (self.grid_dim - 1) / 2) * self.grid_size
                z = 0.0  # Assuming the z-coordinate is zero
                self.grid_centers.append((x, y, z))

        # Create the initial positions tensor for a corner of the map
        # self.m4_init_grid_pos = torch.zeros((self.num_envs, 3), device=self.sim.device)
        # for i in range(self.num_envs):
        #     self.m4_init_grid_pos[i] = torch.tensor(self.grid_centers[i % (self.grid_dim*self.grid_dim)])

        # Create the initial positions tensor for the center of the map
        spiral_positions = generate_spiral_positions(grid_size=self.grid_size, num_grids=self.grid_dim)
        spiral_positions_3d = [(x, y, 0.0) for x, y in spiral_positions]
        self.m4_init_grid_pos = torch.tensor(spiral_positions_3d, device=self.sim.device)

        # Parameters to save depth images for debug
        self.frame_count = 0
        self.save_img = True

        if len(self.cfg.camera2.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.camera2.data_types}"
            )

    def initialize_weights(self, module):
        """Initialize weights using Kaiming initialization for all layers in a module."""
        for layer in module.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def save_weights(self, module, path):
        """Save the weights of a module."""
        torch.save(module.state_dict(), path)

    def load_weights(self, module, path):
        """Load weights into a module."""
        module.load_state_dict(torch.load(path))

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_velocity_targets = self.cfg.num_velocity_targets
        self.num_observed_joints = self.cfg.num_observed_joints
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states
        self.dual_camera = self.cfg.dual_camera
        self.resnet18 = self.cfg.resnet18
        self.rand_fc_layer = self.cfg.rand_fc_layer

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()

        if self.resnet18:
            self.single_observation_space["policy"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, self.cfg.num_observations),
            )
            if self.num_states > 0:
                self.single_observation_space["critic"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, self.cfg.num_observations),
                )
        else:
            self.single_observation_space["policy"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.cfg.camera2.height, self.cfg.camera2.width, self.cfg.num_channels),
            )
            if self.num_states > 0:
                self.single_observation_space["critic"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.camera2.height, self.cfg.camera2.width, self.cfg.num_channels),
                )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera."""
        
        # Add M4 robot
        self._m4_robot = Articulation(self.cfg.robot_cfg)
        self._camera2 = Camera(self.cfg.camera2)
        if self.cfg.dual_camera:
            self._camera3 = Camera(self.cfg.camera3)
        self._contact_forces = ContactSensor(self.cfg.contact_forces)
       
       # Add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add articultion and sensors to scene
        self.scene.articulations["m4_robot"] = self._m4_robot
        self.scene.sensors["Camera2"] = self._camera2
        if self.cfg.dual_camera:
            self.scene.sensors["Camera3"] = self._camera3
        self.scene.sensors["Contact"] = self._contact_forces
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:

        # Apply target velocity given by the operator to the wheels
        self._m4_robot.set_joint_velocity_target(self.target_wheel_vel, joint_ids=self._m4_wheel_idx_list)
        
        # print("Actions: ", self.actions)

        # Apply joint positions from the model
        if self.num_actions == 1:
            actions_4_hips = self.actions.repeat(1, 4)
            self._m4_robot.set_joint_position_target(torch.clamp(actions_4_hips, min=0, max=1.0), joint_ids=self._m4_hip_idx_list) # 60 degrees is about 1.04 rad ~ 1.0 rad
        elif self.num_actions == 2:
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 0], min=0.0, max=1.0), joint_ids=self._m4_hip_idx_list[0])
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 0], min=0.0, max=1.0), joint_ids=self._m4_hip_idx_list[1])
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 1], min=0.0, max=1.0), joint_ids=self._m4_hip_idx_list[2])
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 1], min=0.0, max=1.0), joint_ids=self._m4_hip_idx_list[3])
        elif self.num_actions == 4:
            # print("torch.clamp(self.actions, min=0.0, max=1.0): ", torch.clamp(self.actions, min=0.0, max=1.0))
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions, min=0.0, max=1.0), joint_ids=self._m4_hip_idx_list) # 60 degrees is about 1.04 rad ~ 1.0 rad
        elif self.num_actions == 8:
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, :4], min=0, max=1.0), joint_ids=self._m4_hip_idx_list)
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 4], min=0, max=1.57), joint_ids=self._m4_leg_idx_list[0])
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 5], min=-1.57, max=0), joint_ids=self._m4_leg_idx_list[1])
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 6], min=-1.57, max=0), joint_ids=self._m4_leg_idx_list[2])
            self._m4_robot.set_joint_position_target(torch.clamp(self.actions[:, 7], min=0, max=1.57), joint_ids=self._m4_leg_idx_list[3])

    def _get_observations(self) -> dict:
        
        if self.resnet18:

            data_type2 = "distance_to_camera"
            depth_data2 = self._camera2.data.output[data_type2].clone()
            depth_data2[torch.isinf(depth_data2)] = 6.0

            # Initialize a list to hold preprocessed images
            preprocessed_images = []

            # Process each agent's depth image
            for i in range(depth_data2.shape[0]):
                # Convert the single-channel depth image to a three-channel image by repeating the single channel
                depth_image = depth_data2[i].unsqueeze(0).repeat(3, 1, 1)

                # Convert the tensor to a NumPy array and then to a PIL image
                depth_image_np = depth_image.cpu().numpy().transpose(1, 2, 0)  # Convert to [H, W, C] for PIL

                # Apply the preprocessing steps (Rescaling [0.0,1.0], Resize, ToTensor, Normalize)
                depth_image_np = depth_image_np / 6.0
                depth_image_tensor = self.preprocess(depth_image_np)
                preprocessed_images.append(depth_image_tensor)

            # Stack the preprocessed images to form a batch
            input_tensor = torch.stack(preprocessed_images).to(self.sim.device)

            # Extract features using ResNet-18
            with torch.no_grad():
                features = self.resnet18_model(input_tensor)    
            # features = features.view(features.size(0), 1, -1)

            with torch.no_grad():
                features = self.fc_layer(features)
            features = features.view(features.size(0), 1, -1) # Flatten the features

            if self.dual_camera:

                preprocessed_images = []

                data_type3 = "distance_to_camera"
                depth_data3 = self._camera3.data.output[data_type3].clone()
                depth_data3[torch.isinf(depth_data3)] = 6.0

                # Process each agent's depth image
                for i in range(depth_data3.shape[0]):
                    # Convert the single-channel depth image to a three-channel image by repeating the single channel
                    depth_image = depth_data3[i].unsqueeze(0).repeat(3, 1, 1)

                    # Convert the tensor to a NumPy array and then to a PIL image
                    depth_image_np = depth_image.cpu().numpy().transpose(1, 2, 0)  # Convert to [H, W, C] for PIL

                    # Apply the preprocessing steps (Rescaling [0.0,1.0], Resize, ToTensor, Normalize)
                    depth_image_np = depth_image_np / 6.0
                    # print("Depth Image: ", depth_image_np)
                    # print("Depth Image Shape: ", depth_image_np.shape)
                    depth_image_tensor = self.preprocess(depth_image_np)
                    # print("Depth Image Processed: ", depth_image_tensor)
                    # print("Depth Image Processed Shape: ", depth_image_tensor.shape)
                    preprocessed_images.append(depth_image_tensor)

                # Stack the preprocessed images to form a batch
                input_tensor = torch.stack(preprocessed_images).to(self.sim.device)

                # Extract features using ResNet-18
                with torch.no_grad():
                    features2 = self.resnet18_model(input_tensor)    
                # features2 = features2.view(features2.size(0), 1, -1)
                
                with torch.no_grad():
                    features2 = self.fc_layer(features2)
                features2 = features.view(features2.size(0), 1, -1) # Flatten the features

                features = torch.cat([features, features2], dim=-1)
        
            if self.num_velocity_targets == 2:
                features = torch.cat([features, self.target_vel.reshape(self.num_envs, 1, 2)], dim=-1)

            if self.num_observed_joints == 4:
                tracked_joint_pos = self.joint_pos[:, self._m4_hip_idx]
                features = torch.cat([features, tracked_joint_pos.reshape(self.num_envs, 1, 4)], dim=-1)


            observations = {"policy": features}
        
        else:

            data_type2 = "distance_to_camera"
            depth_data = self._camera2.data.output[data_type2].clone()
            depth_data[torch.isinf(depth_data)] = 6.0
            depth_data = depth_data.unsqueeze(-1)

            if self.dual_camera:
                
                data_type3 = "distance_to_camera" 
                depth_data3 = self._camera3.data.output[data_type3].clone()   
                depth_data3[torch.isinf(depth_data3)] = 6.0
                depth_data3 = depth_data3.unsqueeze(-1)

                depth_data = torch.cat([depth_data, depth_data3], dim=-1)

            observations = {"policy": depth_data}
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
       
        # Processed contacts
        contact_sensors_wheels_idx = torch.tensor([4,8,12,16]) # For some unknown reason, the index of a contact sensor attached to a body is not the same as the index of that body in the robot
        current_air_time = self._contact_forces.data.current_air_time[:, contact_sensors_wheels_idx]
        tracked_joint_idx = self._m4_hip_idx

        # Process ids of joints to track
        if self.num_actions == 8:
            tracked_joint_idx = torch.cat([self._m4_hip_idx, self._m4_leg_idx], dim=-1)
            
        total_reward = compute_rewards(
            self._m4_base_body_idx,
            self._m4_hip_body_idx,
            self._m4_leg_body_idx,
            tracked_joint_idx,
            self.joint_pos,
            self.target_vel,
            self.root_lin_vel_b[:, 0],
            self.root_lin_vel_b[:, 1],
            self.root_pos_w[:, 2],
            self.projected_gravity_b,
            self.contacts,
            current_air_time,
            self.actions,
            self.prev_actions,
            self.episode_count
        )
        self.prev_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self.root_pos_w = self._m4_robot.data.root_pos_w

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        out_of_bounds = ((self.root_pos_w[:, 0] > (self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 0] < -(self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 1] > (self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 1] < -(self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 2] < -4) | (self.projected_gravity_b[:, 2] > 0.8))

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._m4_robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.episode_count += 1
        print("Episode counter: ", self.episode_count)

        # Assigning a random positions to the hip joints, default positions to other joints
        # random_hip_positions = torch.rand(env_ids.shape[0], device=self.sim.device).unsqueeze(1)  # Generate a unique random number for each environment
        # random_hip_positions = random_hip_positions.repeat(1, 4)  # Repeat the random number across all 4 hip joints
        # default_joint_pos = self._m4_robot.data.default_joint_pos[env_ids]
        # for idx, hip_idx in enumerate(self._m4_hip_idx_list):
        #     default_joint_pos[:, hip_idx] = random_hip_positions[:, idx]

        # Assigning default velocities to all joints
        default_joint_pos = self._m4_robot.data.default_joint_pos[env_ids]
        default_joint_vel = self._m4_robot.data.default_joint_vel[env_ids]

        # randomize_rigid_body_material(self._m4_robot, (0.8, 0.8), (0.6, 0.6), (0.0, 0.0))
        # randomize_mass(self._m4_robot, (0.0, 5.0))

        # Generate random Vx
        rand_lin_speed = torch.full((env_ids.shape[0], 1), 0.3,  device=self.sim.device)
        # rand_lin_speed = torch.rand(env_ids.shape[0], device=self.sim.device) * 0.3 # Gives random Vx velocity betwen [0.0, 0.3] m/s         
        # rand_lin_speed = rand_lin_speed.reshape(env_ids.shape[0],1)
        
        # Generate random Wz
        # rand_ang_speed = torch.rand(env_ids.shape[0], device=self.sim.device) * 0.3 # Gives random Wz velocity betwen [0.0, 0.3] rad/s        
        rand_ang_speed = torch.zeros(env_ids.shape[0], device=self.sim.device) # Sets Wz velocity to 0.0 rad/s          
        rand_ang_speed = rand_ang_speed.reshape(env_ids.shape[0],1)
        self.target_vel[env_ids] = torch.cat([rand_lin_speed, rand_ang_speed], dim=-1)

        # Set wheels velocities based on targets
        rand_wheel_velocities = rand_lin_speed.repeat(1, 4) * 10 # Set wheel joint velocity to a random number betwen [0.0, 3.0] rad/s (given a 0.1 m wheel radius and a random speed between [0.0, 0.3] m/s) 
        self.target_wheel_vel[env_ids] = rand_wheel_velocities
        default_joint_vel[:, self._m4_wheel_idx] = rand_wheel_velocities
        
        # Default positions
        default_root_state = self._m4_robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.m4_init_grid_pos[env_ids]
    
        # Default orientations
        positions = default_root_state[:, :3].clone()
        zero_tensor = torch.zeros_like(positions[:,0], device=self.sim.device)
        rand_yaw = (torch.rand(env_ids.shape[0], device=self.sim.device) * 2 * torch.pi) - torch.pi
        orientations_delta = math_utils.quat_from_euler_xyz(zero_tensor, zero_tensor, rand_yaw)
        orientations = math_utils.quat_mul(default_root_state[:, 3:7], orientations_delta)

        # Update simulation
        self._m4_robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids)
        self._m4_robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._m4_robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)
    

def generate_spiral_positions(grid_size=5, num_grids=7):
    # Calculate the center of the grid
    center_x = center_y = num_grids // 2
    positions = []
    
    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_index = 0
    
    # Start from the center
    x, y = center_x, center_y
    positions.append((x, y))
    
    steps = 1
    while len(positions) < num_grids * num_grids:
        for _ in range(2):
            for _ in range(steps):
                x += directions[direction_index][0]
                y += directions[direction_index][1]
                if 0 <= x < num_grids and 0 <= y < num_grids:
                    positions.append((x, y))
            direction_index = (direction_index + 1) % 4
        steps += 1
    
    # Convert grid positions to actual coordinates
    coordinates = [(grid_size * (pos[0] - center_x), grid_size * (pos[1] - center_y)) for pos in positions]
    return coordinates

@torch.jit.script
def compute_rewards(
    base_idx: torch.Tensor,
    hip_idx: torch.Tensor,
    leg_idx: torch.Tensor,
    tracked_joint_idx: torch.Tensor,
    joint_pos: torch.Tensor,
    target_vel: torch.Tensor,
    root_lin_vel_x: torch.Tensor,
    root_lin_vel_y: torch.Tensor,
    root_pos_z: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    contacts: torch.Tensor,
    current_air_time: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    episode: int
):
    # Tracking Vx
    rew_track_lin_vel_x_m4 = - 500 * torch.square(target_vel[:, 0] - root_lin_vel_x)
    
    # Penalizing Vy
    # rew_track_lin_vel_y_m4 = - 500 * torch.square(root_lin_vel_y)
    
    # Reversed robot
    zero_tensor = torch.zeros_like(projected_gravity_b[:, 0], device="cuda:0")
    rew_reversed_robot = - 50 * torch.where(projected_gravity_b[:, 2] > 0, projected_gravity_b[:, 2], 0)
    
    # Wheels on ground
    rew_air_time = - 10 * torch.sum(current_air_time , dim=1)

    # Hip coordination
    # tracked_joint_pos = joint_pos[:, tracked_joint_idx]
    # FL_hip_pos = tracked_joint_pos[:, 0]
    # FR_hip_pos = tracked_joint_pos[:, 1]
    # RL_hip_pos = tracked_joint_pos[:, 2]
    # RR_hip_pos = tracked_joint_pos[:, 3]
    # front_hip_coordination = abs(FL_hip_pos - FR_hip_pos)
    # rear_hip_coordination = abs(RL_hip_pos - RR_hip_pos)
    # rew_hip_coordination = - (torch.exp(front_hip_coordination / 0.5) + torch.exp(rear_hip_coordination / 0.5))

    # Flat robot
    rew_flat_robot = - 200 * torch.square(torch.norm(projected_gravity_b[:, :2], dim=-1))

    # Rolling
    tracked_joint_pos = joint_pos[:, tracked_joint_idx]
    sum_joint_pos = torch.sum(tracked_joint_pos, dim=-1)
    rew_rolling_not_crawling = - torch.exp(sum_joint_pos) 

    # Contacts
    is_contact_base = (torch.max(torch.norm(contacts[:, :, base_idx], dim=-1), dim=1)[0] > 10).squeeze(1)
    # is_contact_hip =  torch.sum(torch.max(torch.norm(contacts[:, :, hip_idx], dim=-1), dim=1)[0] > 10, dim=-1)
    # is_contact_leg = torch.sum(torch.max(torch.norm(contacts[:, :, leg_idx], dim=-1), dim=1)[0] > 10, dim=-1)
    rew_contact = - 100 * (is_contact_base) * torch.exp(-sum_joint_pos)
    
    # Action rate
    # rew_action_rate = - 10 * torch.square(actions-prev_actions).squeeze(1)

    # print("rew_track_lin_vel_x_m4: ", rew_track_lin_vel_x_m4)
    # print("rew_track_lin_vel_y_m4: ", rew_track_lin_vel_y_m4)
    # print("rew_flat_robot: ", rew_flat_robot)
    # print("rew_reversed_robot: ", rew_reversed_robot)
    # print("rew_rolling_not_crawling: ", rew_rolling_not_crawling)
    # print("rew_contact: ", rew_contact)
    # print("rew_air_time: ", rew_air_time)
    # print("rew_hip_coordination: ", rew_hip_coordination)

    total_reward = (rew_track_lin_vel_x_m4 + rew_reversed_robot + rew_flat_robot + rew_rolling_not_crawling + rew_contact + rew_air_time)
    # print("total_reward: ", total_reward)
    print("mean_reward: ", torch.mean(total_reward))

    return total_reward