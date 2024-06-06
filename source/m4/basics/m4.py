"""Configuration for the M4 robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

M4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/m4/IsaacLab/source/m4/omni.isaac.lab_assets/data/Robots/Caltech/m4_fixed_blade.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={
            ".*wheel_joint": 0.0, 
            ".*hip_joint": 0.0,
            ".*leg_joint": 0.0,
            },
    ),
    actuators={
        "hip_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*leg_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wheel_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit=400.0,
            velocity_limit=1000.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)
"""Configuration for the M4 robot."""
