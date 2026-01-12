"""
Launch file for Overtake Simulation + Agent (2 Agents)

Launches:
1. F1Tenth Gym Bridge (simulator with 2 agents)
2. Map Server (for RViz visualization)
3. Overtake Agent (controls both ego and opp)

Usage:
  ros2 launch agent_mcqueen sim_overtake_launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    pkg_share = get_package_share_directory('agent_mcqueen')
    f1tenth_share = get_package_share_directory('f1tenth_gym_ros')

    # Config files
    sim_config = os.path.join(pkg_share, 'config', 'sim_overtake.yaml')
    agent_config = os.path.join(pkg_share, 'config', 'overtake_config.yaml')

    # Load config to get map_path
    config_dict = yaml.safe_load(open(sim_config, 'r'))
    map_path = config_dict['/sim/bridge']['ros__parameters']['map_path']

    # Simulator bridge node
    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[sim_config]
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(f1tenth_share, 'launch', 'gym_bridge.rviz')]
    )

    # Map server - publishes /sim/map for RViz
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[{'yaml_filename': map_path + '.yaml'},
                    {'topic': 'map'},
                    {'frame_id': 'sim'},
                    {'output': 'screen'},
                    {'use_sim_time': True}]
    )

    # Lifecycle manager for map_server
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    # Robot state publishers
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(f1tenth_share, 'launch', 'ego_racecar.xacro')])}],
        remappings=[('/sim/robot_description', '/sim/ego_robot_description')]
    )

    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(f1tenth_share, 'launch', 'opp_racecar.xacro')])}],
        remappings=[('/sim/robot_description', '/sim/opp_robot_description')]
    )

    # Overtake Agent node (controls both agents)
    overtake_agent_node = Node(
        package='agent_mcqueen',
        executable='overtake_agent',
        name='overtake_agent',
        parameters=[agent_config],
        output='screen'
    )

    return LaunchDescription([
        PushRosNamespace('sim'),
        rviz_node,
        bridge_node,
        nav_lifecycle_node,
        map_server_node,
        ego_robot_publisher,
        opp_robot_publisher,
        overtake_agent_node
    ])
