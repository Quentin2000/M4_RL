# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

M4_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=1.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.524,
    use_cache=False,
    sub_terrains={
        # "go_over_boxes": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.2,
        #     box_height_range=[0.1, 0.3],
        #     platform_width=3.0
        # ),
        # "to_avoid_boxes": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.4,
        #     box_height_range=[0.0, 2.0],
        # ),
        # "slopes": terrain_gen.HfTerrainBaseCfg(
        #     proportion=0.1,
        #     border_width=0.0,
        #     horizontal_scale=0.1,
        #     vertical_scale=0.005,
        #     slope_threshold=1.5,
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.2,
        #     gap_width_range=[0.20, 0.30],
        #     platform_width=2.0,
        # ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=[0.0, 0.0],
            platform_width=3.0,
            double_pit=False,
        ),
        # "pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.1,
        #     pit_depth_range=[0.1, 0.2],
        #     platform_width=2.0,
        #     double_pit=True,
        # ),
        # "pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.1,
        #     pit_depth_range=[-0.2, -0.1],
        #     platform_width=2.0,
        #     double_pit=True,
        # ),
        "ring": terrain_gen.MeshFloatingRingTerrainCfg(
            proportion=0.60,
            ring_width_range=[0.4, 0.8],
            ring_height_range=[0.27, 0.35],
            ring_thickness=0.5,
            platform_width=3.0,
        ),
        # "ring": terrain_gen.MeshFloatingRingTerrainCfg(
        #     proportion=0.20,
        #     ring_width_range=[0.2, 1.0],
        #     ring_height_range=[0.30, 0.50],
        #     ring_thickness=0.5,
        #     platform_width=3.0,
        # ),
        # "positive_pyramids": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.30,
        #     slope_range=[0.175, 0.35], # -> [10,25] degrees
        #     platform_width=1.0,
        #     inverted=False,
        # ),
        # "negative_pyramids": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.30,
        #     slope_range=[0.175, 0.35], # -> [10,20] degrees
        #     platform_width=1.0,
        #     inverted=True,
        # ),
        # "waves": terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.1,
        #     amplitude_range=[0.05, 0.2],
        #     num_waves=2,
        # ),
    },
)
"""M4 terrains configuration."""
