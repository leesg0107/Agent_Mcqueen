"""
Launch file for Stage 1 Agent (Single Agent)

Usage:
  ros2 launch agent_mcqueen stage1_agent_launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('agent_mcqueen')
    config_file = os.path.join(pkg_share, 'config', 'stage1_config.yaml')

    stage1_agent_node = Node(
        package='agent_mcqueen',
        executable='stage1_agent',
        name='stage1_agent',
        parameters=[config_file],
        output='screen'
    )

    return LaunchDescription([
        stage1_agent_node
    ])
