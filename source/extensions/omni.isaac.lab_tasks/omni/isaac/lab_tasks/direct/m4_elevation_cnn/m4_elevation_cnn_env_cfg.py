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


import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, Camera, CameraCfg, ContactSensorCfg, ContactSensor
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab.terrains.config.m4_terrain import M4_TERRAINS_CFG

output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

@configclass
class M4ElevationCnnEnvCfg(DirectRLEnvCfg):

    # env
    decimation = 10
    episode_length_s = 20.0

    # simulation
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

    # robot
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
                ".*wheel_joint": 3.0,
            },
        ),
        actuators={
            "hip_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_joint"],
                effort_limit=100000.0,
                velocity_limit=4.0,
                stiffness= 8.0,
                damping= 0.0,
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

    # ground terrain
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

    # camera = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link/camera",
    #     update_period=0.1,
    #     height=72,
    #     width=128,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=0.193, focus_distance=0.6, horizontal_aperture=1.0
    #         # focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.24099, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    # See: https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf
    camera2 = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/camera2",
        update_period=0.1,
        height=72,
        width=72,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.193, focus_distance=0.60, horizontal_aperture=1.0, clipping_range=(0.4, 6.0)
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
            focal_length=0.193, focus_distance=0.60, horizontal_aperture=1.0, clipping_range=(0.4, 6.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.24099, 0.0, 0.0), rot=(-0.5, 0.5, 0.5, -0.5), convention="ros"),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=3, 
        track_air_time=False)

    # change viewer settings
    viewer = ViewerCfg(eye=(2.0, 2.0, 5.0))

    # actions
    action_scale = 1  # [N]
    num_actions = 4

    # observations
    num_channels = 2
    # num_observations = num_channels * camera.height * camera.width
    num_observations = num_channels * camera2.height * camera2.width
    num_states = 0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=20.0, replicate_physics=True)

    # reset
    # max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    # initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005


class M4ElevationCnnEnv(DirectRLEnv):

    cfg: M4ElevationCnnEnvCfg

    def __init__(
        self, cfg: M4ElevationCnnEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        torch.set_printoptions(threshold=10000)

        # Load the pretrained ResNet-18 model
        self.resnet18_model = resnet18(pretrained=True)
        # Remove the final classification layer to get feature extraction layer
        resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1])
        resnet18_model.eval()

        # Define the preprocessing transformations
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Convert to PIL Image
            transforms.Resize(224),   # Resize to 224x224
            transforms.ToTensor(),    # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])
        
        m4_base_body_idx_list, _ = self._m4_robot.find_bodies(".*base_link")
        m4_hip_body_idx_list, _ = self._m4_robot.find_bodies(".*hip")
        m4_leg_body_idx_list, _ = self._m4_robot.find_bodies(".*leg")
        self._m4_base_body_idx = torch.tensor(m4_base_body_idx_list)
        self._m4_hip_body_idx = torch.tensor(m4_hip_body_idx_list)
        self._m4_leg_body_idx = torch.tensor(m4_leg_body_idx_list)

        self._m4_hip_idx_list, _ = self._m4_robot.find_joints(".*hip_joint")
        self._m4_hip_idx = torch.tensor(self._m4_hip_idx_list)
        self._m4_wheel_idx, _ = self._m4_robot.find_joints(".*wheel_joint")
        # self._pole_dof_idx, _ = self._cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self._m4_robot.data.joint_pos
        self.joint_vel = self._m4_robot.data.joint_vel
        self.root_lin_vel_b = self._m4_robot.data.root_lin_vel_b
        self.root_pos_w = self._m4_robot.data.root_pos_w
        self.projected_gravity_b = self._m4_robot.data.projected_gravity_b
        self.contacts = self._contact_forces.data.net_forces_w_history

        self.grid_dim = 7 # Number of rows or columns in the map, given that the map is a square
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

    #    print("self.m4_init_grid_pos: ", self.m4_init_grid_pos)

        # Parameters to save depth images for debug
        self.frame_count = 0
        self.save_img = True

        # if len(self.cfg.camera.data_types) != 1:
        #     raise ValueError(
        #         "The Cartpole camera environment only supports one image type at a time but the following were"
        #         f" provided: {self.cfg.camera.data_types}"
        #     )
        if len(self.cfg.camera2.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.camera2.data_types}"
            )

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        # self.single_observation_space["policy"] = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.cfg.camera.height, self.cfg.camera.width, self.cfg.num_channels),
        # )
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.camera2.height, self.cfg.camera2.width, self.cfg.num_channels),
        )
        # if self.num_states > 0:
        #     self.single_observation_space["critic"] = gym.spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=(self.cfg.camera.height, self.cfg.camera.width, self.cfg.num_channels),
        #     )
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
        self._m4_robot = Articulation(self.cfg.robot_cfg)
        # self._camera = TiledCamera(self.cfg.camera)
        self._camera2 = Camera(self.cfg.camera2)
        self._camera3 = Camera(self.cfg.camera3)
        self._contact_forces = ContactSensor(self.cfg.contact_forces)
       
       # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.terrain)
        # spawn_ground_plane(
        #     prim_path="/World/ground", 
        #     cfg=TerrainImporterCfg(
        #         prim_path="/World/ground",
        #         num_envs=2048,
        #         env_spacing=3.0,
        #         max_init_terrain_level=5,
        #         terrain_type="generator",
        #         # terrain_generator=M4_TERRAINS_CFG.replace(color_scheme="random"),
        #         terrain_generator=M4_TERRAINS_CFG,
        #         visual_material=None,
        #         # visual_material=sim_utils.MdlFileCfg(
        #         #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #         #     project_uvw=True,
        #         #     texture_scale=(0.25, 0.25),
        #         # ),
        #     )
        # )
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["m4_robot"] = self._m4_robot
        # self.scene.sensors["Camera"] = self._camera
        self.scene.sensors["Camera2"] = self._camera2
        self.scene.sensors["Camera3"] = self._camera3
        self.scene.sensors["Contact"] = self._contact_forces
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        velocity_actions = torch.full((self.num_envs, 4), 3.0,  device=self.sim.device) # Set a wheels joint velocity of 3.0 rad/s to get to 0.3 m/s linear speed (given a 10 cm wheel radius)
        self._m4_robot.set_joint_velocity_target(velocity_actions, joint_ids=self._m4_wheel_idx)
        # actions_4_hips = self.actions.repeat(1, 4)
        # self._m4_robot.set_joint_position_target(torch.clamp(actions_4_hips, min=0, max=1.0), joint_ids=self._m4_hip_idx_list) # 60 degrees is about 1.04 rad ~ 1.0 rad
        
        self._m4_robot.set_joint_position_target(torch.clamp(self.actions, min=0, max=1.0), joint_ids=self._m4_hip_idx_list) # 60 degrees is about 1.04 rad ~ 1.0 rad
        

    def _get_observations(self) -> dict:
        # data_type = "depth"
        data_type2 = "distance_to_camera"
        # data_type3 = "distance_to_camera"
        # depth_data = self._camera.data.output[data_type].clone()
        depth_data2 = self._camera2.data.output[data_type2].clone()
        # depth_data3 = self._camera3.data.output[data_type3].clone()
        depth_data2[torch.isinf(depth_data2)] = 6.0
        # depth_data3[torch.isinf(depth_data3)] = 6.0

        # merged_depth_data = torch.cat([depth_data2.unsqueeze(-1), depth_data3.unsqueeze(-1)], dim=-1)
        
        if len(depth_data2.shape) == 2:
            depth_data = np.stack((depth_data2,)*3, axis=-1)

        input_tensor = preprocess(depth_data)
        input_tensor = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        with torch.no_grad():
            features = self.resnet18_model(input_tensor)
        features = features.flatten()

        # if self.save_img:
        #     depth_data_array = depth_data.cpu().numpy().flatten()
        #     np.savetxt('depth_tiled.txt', depth_data_array)
            
        #     depth_data_array = depth_data2.cpu().numpy().flatten()
        #     np.savetxt('depth.txt', depth_data_array)
            
        #     self.save_img = False

        # print("Depth Tiled shape: ", depth_data.shape)
        # print("Depth Camera shape: ", depth_data2.shape)
        # observations = {"policy": depth_data2}
        observations = {"policy": features}
        
        # Live display
        # if self.frame_count % 10 == 0:

        #     # plt.switch_backend('Qt5Agg')
        #     plt.clf()

        #     depth_data_np = depth_data2.cpu().numpy().squeeze()
        #     max_value = np.nanmax(depth_data_np[np.isfinite(depth_data_np)])
        #     depth_data_np[depth_data_np == float('inf')] = max_value + 1
            
        #     img = plt.imshow(depth_data_np, cmap='gray', interpolation='nearest')
        #     plt.colorbar()
        #     filename = os.path.join(output_dir, f'frame_{self.frame_count:03d}.png')
        #     plt.savefig(filename)

        #     plt.pause(0.01)

        # self.frame_count += 1
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self._m4_base_body_idx,
            self._m4_hip_body_idx,
            self._m4_leg_body_idx,
            self._m4_hip_idx,
            self.joint_pos,
            self.root_lin_vel_b[:, 0],
            self.root_pos_w[:, 2],
            self.projected_gravity_b,
            self.contacts,
            self.actions,
            self.prev_actions
        )
        self.prev_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self.root_pos_w = self._m4_robot.data.root_pos_w

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        out_of_bounds = ((self.root_pos_w[:, 0] > (self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 0] < -(self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 1] > (self.grid_dim*self.grid_size)/2) | (self.root_pos_w[:, 1] < -(self.grid_dim*self.grid_size)/2))
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._m4_robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._m4_robot.data.default_joint_pos[env_ids]
        joint_vel = self._m4_robot.data.default_joint_vel[env_ids] 

        # Default positions
        default_root_state = self._m4_robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.m4_init_grid_pos[env_ids]
        # print("default_root_state[:, :3]: ", default_root_state[:, :3])
        # Default orientations
        positions = default_root_state[:, :3].clone()
        zero_tensor = torch.zeros_like(positions[:,0], device=self.sim.device)
        rand_yaw = (torch.rand(env_ids.shape[0], device=self.sim.device) * 2 * torch.pi) - torch.pi
        orientations_delta = math_utils.quat_from_euler_xyz(zero_tensor, zero_tensor, rand_yaw)
        orientations = math_utils.quat_mul(default_root_state[:, 3:7], orientations_delta)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self._m4_robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids)
        self._m4_robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._m4_robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    

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
    hip_joint_idx: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel_x: torch.Tensor,
    root_pos_z: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    contacts: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor
):
    # Tracking Vx
    num_envs = root_lin_vel_x.shape[0]
    rew_track_lin_vel_x_m4 = - 500 * torch.square((torch.full((num_envs, 1), 0.3,  device="cuda:0") - root_lin_vel_x.unsqueeze(dim=1))).squeeze(1)
    
    # Reversed robot
    zero_tensor = torch.zeros_like(projected_gravity_b[:, 0], device="cuda:0")
    rew_reversed_robot = - 50 * torch.where(projected_gravity_b[:, 2] > 0, - 50 * projected_gravity_b[:, 2], 0)
    
    # Flat robot
    rew_flat_robot = - 200 * torch.square(torch.norm(projected_gravity_b[:, :2], dim=-1))

    # Rolling
    rew_rolling_not_crawling = - 50 * abs(root_pos_z.unsqueeze(dim=1)-0.32).squeeze(1)
    
    # Contacts
    is_contact_base = (torch.max(torch.norm(contacts[:, :, base_idx], dim=-1), dim=1)[0] > 10).squeeze(1)
    # print("is_contact_base: ", is_contact_base)
    # print("is_contact_base.shape: ", is_contact_base.shape)
    # is_contact_hip =  torch.sum(torch.max(torch.norm(contacts[:, :, hip_idx], dim=-1), dim=1)[0] > 10, dim=-1)
    # print("is_contact_hip: ", is_contact_hip)
    # is_contact_leg = torch.sum(torch.max(torch.norm(contacts[:, :, leg_idx], dim=-1), dim=1)[0] > 10, dim=-1)
    # print("is_contact_leg.shape: ", is_contact_leg.shape)
    # print("torch.sum(root_pos_z.unsqueeze(dim=1)-0.25, dim=-1): ", torch.sum(root_pos_z.unsqueeze(dim=1)-0.25, dim=-1))
    min_z_elevation = 0.19
    # print("root_pos_z-min_z_elevation: ", root_pos_z-min_z_elevation)
    # print("root_pos_z-min_z_elevation.shape: ", (root_pos_z-min_z_elevation).shape)
    # print("is_contact_base + is_contact_leg", is_contact_base + is_contact_leg)
    rew_contact = - 5 * (is_contact_base) * 100 * abs(root_pos_z-min_z_elevation)

    # Action rate
    # rew_action_rate = - 10 * torch.square(actions-prev_actions).squeeze(1)

    # Joint pos
    # hip_joint_pos = joint_pos[:, hip_joint_idx]
    # print("hip_joint_pos: ", hip_joint_pos)
    # rew_neg_joint_pos = torch.sum(torch.where(actions < 0, 10 * actions, 0), dim=-1)
    # rew_over_joint_pos = torch.sum(torch.where(actions > 1.0, - 30 * (actions - 1.0), 0), dim=-1)
    # print("rew_neg_joint_pos: ", rew_neg_joint_pos)
    # print("rew_over_joint_pos: ", rew_over_joint_pos)
    # rew_joint_pos = rew_neg_joint_pos

    # print("rew_track_lin_vel_x_m4: ", rew_track_lin_vel_x_m4)
    # print("rew_flat_robot: ", rew_flat_robot)
    # print("rew_reversed_robot: ", rew_reversed_robot)
    # print("rew_rolling_not_crawling: ", rew_rolling_not_crawling)
    # print("rew_contact: ", rew_contact)
    # print("rew_action_rate: ", rew_action_rate)
    # print("rew_joint_pos: ", rew_joint_pos)

    total_reward = (rew_track_lin_vel_x_m4 + rew_reversed_robot + rew_flat_robot + rew_rolling_not_crawling + rew_contact)
    # print("total_reward: ", total_reward)
    print("mean_reward: ", torch.mean(total_reward))

    return total_reward