"""
Launch file for Stage 1 Agent with ForzaETH Simulator (Single Agent)

Launches:
1. ForzaETH Gym Bridge (direct Node with sim.yaml - num_agent: 1)
2. Stage 1 Agent (RL model)

Usage:
  ros2 launch agent_mcqueen forza_stage1_launch.py
  ros2 launch agent_mcqueen forza_stage1_launch.py map_name:=hall

Available maps: hall, GLC_smile_small, small_hall, teras, glc_ot_ez, hangar_1905_v0

Note: Requires stack_master/config/SIM/sim.yaml to have num_agent: 1
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    # Get map_name from launch configuration
    map_name = LaunchConfiguration('map_name').perform(context)

    # Package paths
    pkg_share = get_package_share_directory('agent_mcqueen')
    f1tenth_gym_ros_share = get_package_share_directory('f1tenth_gym_ros')
    stack_master_share = get_package_share_directory('stack_master')

    # ForzaETH paths
    forza_maps = '/home/misys/forza_ws/race_stack/stack_master/maps'
    sim_single_config = os.path.join(stack_master_share, 'config', 'SIM', 'sim_single.yaml')
    sim_params = os.path.join(stack_master_share, 'config', 'SIM', 'sim_params.yaml')

    # Construct map yaml path
    map_yaml_path = os.path.join(forza_maps, map_name, f'{map_name}.yaml')

    # Agent config
    agent_config = os.path.join(pkg_share, 'config', 'forza_stage1_config.yaml')

    # Gym bridge node (single agent config)
    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[sim_single_config,
                    {'map_path': map_yaml_path},
                    {'sim_params': sim_params}]
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', '/home/misys/forza_ws/race_stack/forza_rviz.rviz']
    )

    # Map server
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[{'yaml_filename': map_yaml_path},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'},
                    {'use_sim_time': True}]
    )

    # Lifecycle manager
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    # Robot state publisher
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(
            f1tenth_gym_ros_share, 'config', 'ego_racecar.xacro')])}],
        remappings=[('/robot_description', 'ego_robot_description')]
    )

    # Stage 1 Agent node
    stage1_agent_node = Node(
        package='agent_mcqueen',
        executable='stage1_agent',
        name='stage1_agent',
        parameters=[agent_config],
        output='screen'
    )

    return [
        rviz_node,
        bridge_node,
        nav_lifecycle_node,
        map_server_node,
        ego_robot_publisher,
        stage1_agent_node
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value='hall',
            description='Map name in stack_master/maps'
        ),
        OpaqueFunction(function=launch_setup)
    ])
