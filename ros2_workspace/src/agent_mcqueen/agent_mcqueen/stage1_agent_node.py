"""
Stage 1 Agent Node for F1Tenth ROS2 Simulation

Loads the Stage 1 trained PPO model and publishes drive commands
based on LiDAR scan and odometry data.

Subscribe:
  - /sim/scan (LaserScan): 1080 beam LiDAR
  - /sim/ego_racecar/odom (Odometry): velocity info

Publish:
  - /sim/drive (AckermannDriveStamped): steering and speed

Observation format:
  - Shape: (4, 1081) where 4 = frame_stack, 1081 = num_beams + 1 (velocity)
  - Scans: normalized to [0, 1] by dividing by 10.0 (max range 10m)
  - Velocity: normalized to [0, 1] by dividing by 3.2 (max speed 3.2 m/s)

Action format (PPO raw output is [-1, 1] for both):
  - Action[0]: steering in [-1, 1], scaled to [-0.4189, 0.4189] rad
  - Action[1]: speed in [-1, 1], rescaled to [0, 1] then to [0.0, 3.2] m/s
    (RescaleAction in training: [-1,1] → [0,1])
"""

import os
import sys
import glob
import re
import pickle
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from stable_baselines3 import PPO


# Normalization constants (from training environment)
LIDAR_MAX_RANGE = 10.0  # LiDAR max range in meters (normalized to [0, 1])
MAX_SPEED = 3.2  # Max speed in m/s (normalized to [0, 1])
MAX_STEERING = 0.4189  # Max steering angle in radians


class Stage1AgentNode(Node):
    def __init__(self):
        super().__init__('stage1_agent')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('vecnorm_path', '')
        self.declare_parameter('scan_topic', '/sim/scan')
        self.declare_parameter('odom_topic', '/sim/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/sim/drive')
        self.declare_parameter('num_beams', 1080)
        self.declare_parameter('frame_stack', 4)

        # Get parameters
        model_path = self.get_parameter('model_path').value
        vecnorm_path = self.get_parameter('vecnorm_path').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.num_beams = self.get_parameter('num_beams').value
        self.frame_stack = self.get_parameter('frame_stack').value

        # Frame buffer for stacking - stores 2D array (frame_stack, frame_dim)
        # Each frame: [scans (1080), velocity (1)] = 1081
        self.frame_dim = self.num_beams + 1
        self.frame_buffer = deque(maxlen=self.frame_stack)

        # Current state
        self.current_scan = None
        self.current_velocity = 0.0
        self.model_loaded = False

        # Load model (no VecNormalize needed - observations are pre-normalized)
        if model_path:
            self._load_model(model_path)
        else:
            self.get_logger().error('No model_path provided!')
            return

        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        # Control timer (100Hz to match simulation)
        self.control_timer = self.create_timer(0.01, self.control_callback)

        self.get_logger().info('Stage1 Agent Node initialized')
        self.get_logger().info(f'  Scan topic: {scan_topic}')
        self.get_logger().info(f'  Odom topic: {odom_topic}')
        self.get_logger().info(f'  Drive topic: {drive_topic}')
        self.get_logger().info(f'  Obs shape: ({self.frame_stack}, {self.frame_dim})')

    def _load_model(self, model_path):
        """Load PPO model (no VecNormalize needed - obs are pre-normalized)"""
        self.get_logger().info(f'Loading model: {model_path}')

        try:
            # Load PPO model
            self.model = PPO.load(model_path, device='cpu')
            self.get_logger().info('  [OK] PPO model loaded')
            self.get_logger().info('  [INFO] Observations are pre-normalized (norm_obs=False in training)')
            self.model_loaded = True

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.model_loaded = False

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan data and normalize to [0, 1]"""
        # Convert to numpy and handle inf/nan
        scan = np.array(msg.ranges, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=LIDAR_MAX_RANGE, posinf=LIDAR_MAX_RANGE, neginf=0.0)

        # Resample if needed (should be 1080 beams)
        if len(scan) != self.num_beams:
            indices = np.linspace(0, len(scan) - 1, self.num_beams, dtype=int)
            scan = scan[indices]

        # Flip scan direction to match training environment
        # F1Tenth gym may have different scan direction than ROS2 LaserScan
        scan = scan[::-1]

        # Normalize to [0, 1] (same as training: clip to max_range, divide by max_range)
        scan = np.clip(scan, 0.0, LIDAR_MAX_RANGE)
        scan = scan / LIDAR_MAX_RANGE

        self.current_scan = scan

    def odom_callback(self, msg: Odometry):
        """Process odometry data and normalize velocity to [0, 1]"""
        # Get linear velocity in x direction (forward velocity)
        vx = msg.twist.twist.linear.x

        # Normalize to [0, 1] (same as training: divide by max_speed)
        self.current_velocity = np.clip(vx / MAX_SPEED, 0.0, 1.0)

    def _build_observation(self):
        """Build observation with frame stacking - shape (4, 1081)

        IMPORTANT: FlattenObservation flattens Dict in alphabetical key order!
        Filter keys are ["scans", "linear_vel"], so flattened order is:
        - linear_vel (1) first (alphabetically before 'scans')
        - scans (1080) second
        Total: [linear_vel, scans...] = 1081
        """
        if self.current_scan is None:
            return None

        # Current frame: [linear_vel (1), scan (1080)] = 1081
        # CRITICAL: Order must match FlattenObservation (alphabetical key order)
        current_frame = np.concatenate([
            [self.current_velocity],  # linear_vel comes FIRST (alphabetically)
            self.current_scan         # scans comes SECOND
        ]).astype(np.float32)

        # Add to frame buffer
        self.frame_buffer.append(current_frame)

        # Pad buffer if not full
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.append(current_frame)

        # Stack frames: shape (4, 1081) - matching training environment
        obs = np.stack(list(self.frame_buffer), axis=0)

        return obs

    def control_callback(self):
        """Main control loop - predict and publish drive command"""
        if not self.model_loaded or self.current_scan is None:
            return

        # Build observation - shape (4, 1081)
        obs = self._build_observation()
        if obs is None:
            return

        # Predict action (model expects batch dimension)
        action, _ = self.model.predict(obs, deterministic=True)

        # Debug: log raw action and observation stats
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 100 == 1:  # Log every 100 steps
            self.get_logger().info(f'[DEBUG] raw_action: [{action[0]:.3f}, {action[1]:.3f}]')
            # Obs format: [linear_vel (idx 0), scans (idx 1-1080)]
            self.get_logger().info(f'[DEBUG] obs shape: {obs.shape}, vel: {obs[0,0]:.3f}, scan_min: {obs[0,1:].min():.3f}, scan_max: {obs[0,1:].max():.3f}')

        # PPO outputs raw action in [-1, 1] range
        # RescaleAction in training: [-1,1] x [-1,1] → [-1,1] x [0,1]
        # We need to apply the same transformation here:
        #   steering: [-1, 1] → [-1, 1] (no change)
        #   speed: [-1, 1] → [0, 1] (rescale)

        # Steering: [-1, 1] → [-MAX_STEERING, MAX_STEERING] rad
        steering = float(action[0]) * MAX_STEERING

        # Speed: [-1, 1] → [0, 1] → [0, MAX_SPEED] m/s
        # Apply RescaleAction transformation: (action + 1) / 2
        speed_normalized = (float(action[1]) + 1.0) / 2.0  # [-1,1] → [0,1]
        speed = speed_normalized * MAX_SPEED  # [0,1] → [0, MAX_SPEED]

        # Clip to safe ranges
        steering = np.clip(steering, -MAX_STEERING, MAX_STEERING)
        speed = np.clip(speed, 0.0, MAX_SPEED)

        # Debug: log final command
        if self._log_counter % 100 == 1:
            self.get_logger().info(f'[DEBUG] cmd: steering={steering:.3f} rad, speed={speed:.3f} m/s')

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed

        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)

    node = Stage1AgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
