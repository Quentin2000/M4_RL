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
    num_rows=7,
    num_cols=7,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.524,
    use_cache=False,
    sub_terrains={
        # "go_over_boxes": terrain_gen.MeshBoxTerrainCfg(
        #     proportion=0.2,
        #     box_height_range=[0.01, 0.3],
        #     platform_width=1.0
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
        #     proportion=0.1,
        #     gap_width_range=[0.05,0.30],
        #     platform_width=1.0,
        # ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.30,
            pit_depth_range=[0.0, 0.0],
            platform_width=3.0,
            double_pit=False,
        ),
        "ring": terrain_gen.MeshFloatingRingTerrainCfg(
            proportion=0.40,
            ring_width_range=[0.5, 0.5],
            ring_height_range=[0.28, 0.30],
            ring_thickness=0.5,
            platform_width=3.0,
        ),
        # "ring": terrain_gen.MeshFloatingRingTerrainCfg(
        #     proportion=0.30,
        #     ring_width_range=[0.5, 0.5],
        #     ring_height_range=[0.30, 0.35],
        #     ring_thickness=0.5,
        #     platform_width=3.0,
        # ),
        # "positive_pyramids": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1,
        #     slope_range=[0.175, 0.436], # -> [10,25] degrees
        #     platform_width=1.0,
        #     inverted=False,
        # ),
        # "negative_pyramids": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1,
        #     slope_range=[0.175, 0.436], # -> [10,25] degrees
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
