"""
Launch file for Overtake Agent with ForzaETH Simulator (2 Agents)

Launches:
1. ForzaETH Gym Bridge (simulator with 2 agents)
2. Overtake Agent (controls both ego and opp)

Usage:
  ros2 launch agent_mcqueen forza_overtake_launch.py
  ros2 launch agent_mcqueen forza_overtake_launch.py map_name:=hall

Available maps: hall, GLC_smile_small, small_hall, teras, glc_ot_ez, hangar_1905_v0

Note: Requires stack_master/config/SIM/sim.yaml to have num_agent: 2
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    # Get map_name from launch configuration
    map_name = LaunchConfiguration('map_name').perform(context)

    # Package paths
    pkg_share = get_package_share_directory('agent_mcqueen')

    # ForzaETH paths
    forza_gym_ros = '/home/misys/forza_ws/race_stack/base_system/f110_simulator/f1tenth_gym_ros'
    forza_maps = '/home/misys/forza_ws/race_stack/stack_master/maps'

    # Construct map yaml path
    map_yaml_path = os.path.join(forza_maps, map_name, f'{map_name}.yaml')

    # Agent config
    agent_config = os.path.join(pkg_share, 'config', 'forza_overtake_config.yaml')

    # Include ForzaETH gym_bridge_launch
    gym_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(forza_gym_ros, 'launch', 'gym_bridge_launch.py')
        ),
        launch_arguments={'map_yaml_path': map_yaml_path}.items()
    )

    # Overtake Agent node (controls both agents)
    overtake_agent_node = Node(
        package='agent_mcqueen',
        executable='overtake_agent',
        name='overtake_agent',
        parameters=[agent_config],
        output='screen'
    )

    return [gym_bridge_launch, overtake_agent_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value='hall',
            description='Map name in stack_master/maps'
        ),
        OpaqueFunction(function=launch_setup)
    ])
