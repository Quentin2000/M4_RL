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
        usd_path=f"/home/m4/IsaacLab/source/m4/omni.isaac.lab_assets/data/Robots/Caltech/m4.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
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
            ".*blade_joint": 0.0,
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
        "blade_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*blade_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wheel_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        # "wheel_motors": ImplicitActuatorCfg(
        #     joint_names_expr=[".*wheel_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness={".*wheel_joint": 0.0},
        #     damping={".*wheel_joint": 10.0},
        # ),
        # "motor5": ImplicitActuatorCfg(
        #     joint_names_expr=["rear_right_hip_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor7": ImplicitActuatorCfg(
        #     joint_names_expr=["rear_right_wheel_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor8": ImplicitActuatorCfg(
        #     joint_names_expr=["rear_right_blade_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor9": ImplicitActuatorCfg(
        #     joint_names_expr=["front_left_hip_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor11": ImplicitActuatorCfg(
        #     joint_names_expr=["front_left_wheel_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor12": ImplicitActuatorCfg(
        #     joint_names_expr=["front_left_blade_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor13": ImplicitActuatorCfg(
        #     joint_names_expr=["front_right_hip_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor15": ImplicitActuatorCfg(
        #     joint_names_expr=["front_right_wheel_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "motor16": ImplicitActuatorCfg(
        #     joint_names_expr=["front_right_blade_joint"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
    },
)
"""Configuration for the M4 robot."""
