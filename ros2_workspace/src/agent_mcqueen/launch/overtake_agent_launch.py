"""
Launch file for Overtake Agent (Two Agents)

Usage:
  ros2 launch agent_mcqueen overtake_agent_launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('agent_mcqueen')
    config_file = os.path.join(pkg_share, 'config', 'overtake_config.yaml')

    overtake_agent_node = Node(
        package='agent_mcqueen',
        executable='overtake_agent',
        name='overtake_agent',
        parameters=[config_file],
        output='screen'
    )

    return LaunchDescription([
        overtake_agent_node
    ])
