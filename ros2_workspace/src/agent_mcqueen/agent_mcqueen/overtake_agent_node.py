"""
Overtake Agent Node for F1Tenth ROS2 Simulation (2 Agents)

Loads the Overtake trained model (OvertakePolicy) and publishes drive commands
for both ego (Agent 0, frozen expert) and opp (Agent 1, overtake learner).

Agent 0 (ego): Stage 1 frozen expert, starts AHEAD, runs at 80% speed
Agent 1 (opp): Overtake-trained agent, starts BEHIND, attempts overtaking

Subscribe:
  - /sim/scan (LaserScan): ego LiDAR
  - /sim/opp_scan (LaserScan): opp LiDAR
  - /sim/ego_racecar/odom (Odometry): ego state
  - /sim/opp_racecar/odom (Odometry): opp state

Publish:
  - /sim/drive (AckermannDriveStamped): ego drive
  - /sim/opp_drive (AckermannDriveStamped): opp drive
"""

import os
import sys
import numpy as np
import torch
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# Add overtake_agent path for OvertakePolicy import
# In docker: /home/misys/overtake_agent/common
OVERTAKE_COMMON_PATH = '/home/misys/overtake_agent/common'
if os.path.exists(OVERTAKE_COMMON_PATH):
    sys.path.insert(0, OVERTAKE_COMMON_PATH)
else:
    # Fallback for local testing
    LOCAL_PATH = os.path.join(os.path.dirname(__file__), '../../../../overtake_agent/common')
    if os.path.exists(LOCAL_PATH):
        sys.path.insert(0, LOCAL_PATH)

from overtake_policy import OvertakePolicy


class OvertakeAgentNode(Node):
    def __init__(self):
        super().__init__('overtake_agent')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('ego_scan_topic', '/sim/scan')
        self.declare_parameter('opp_scan_topic', '/sim/opp_scan')
        self.declare_parameter('ego_odom_topic', '/sim/ego_racecar/odom')
        self.declare_parameter('opp_odom_topic', '/sim/opp_racecar/odom')
        self.declare_parameter('ego_drive_topic', '/sim/drive')
        self.declare_parameter('opp_drive_topic', '/sim/opp_drive')
        self.declare_parameter('num_beams', 1080)
        self.declare_parameter('frame_stack', 4)
        self.declare_parameter('agent0_speed_factor', 0.8)  # Handicap for ego
        self.declare_parameter('device', 'cuda')

        # Get parameters
        model_path = self.get_parameter('model_path').value
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        opp_scan_topic = self.get_parameter('opp_scan_topic').value
        ego_odom_topic = self.get_parameter('ego_odom_topic').value
        opp_odom_topic = self.get_parameter('opp_odom_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        opp_drive_topic = self.get_parameter('opp_drive_topic').value
        self.num_beams = self.get_parameter('num_beams').value
        self.frame_stack = self.get_parameter('frame_stack').value
        self.agent0_speed_factor = self.get_parameter('agent0_speed_factor').value
        device = self.get_parameter('device').value

        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            self.get_logger().warn('CUDA not available, using CPU')
        self.device = device

        # Asymmetric dimensions (from training)
        self.OBS_DIM_AGENT0 = 4324  # No opponent info
        self.OBS_DIM_AGENT1 = 4336  # With opponent info
        self.ACTION_DIM = 2
        self.GLOBAL_STATE_DIM = 12

        # Frame buffers for both agents
        self.frame_buffer_ego = deque(maxlen=self.frame_stack)
        self.frame_buffer_opp = deque(maxlen=self.frame_stack)

        # Current state
        self.ego_scan = None
        self.opp_scan = None
        self.ego_pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.opp_pose = [0.0, 0.0, 0.0]
        self.ego_velocity = 0.0
        self.opp_velocity = 0.0
        self.ego_s = 0.0  # Frenet s coordinate (approximate)
        self.opp_s = 0.0
        self.ego_vs = 0.0  # Velocity along track
        self.opp_vs = 0.0

        self.model_loaded = False

        # Load model
        if model_path:
            self._load_model(model_path)
        else:
            self.get_logger().error('No model_path provided!')
            return

        # Publishers
        self.ego_drive_pub = self.create_publisher(
            AckermannDriveStamped, ego_drive_topic, 10)
        self.opp_drive_pub = self.create_publisher(
            AckermannDriveStamped, opp_drive_topic, 10)

        # Subscribers
        self.ego_scan_sub = self.create_subscription(
            LaserScan, ego_scan_topic, self.ego_scan_callback, 10)
        self.opp_scan_sub = self.create_subscription(
            LaserScan, opp_scan_topic, self.opp_scan_callback, 10)
        self.ego_odom_sub = self.create_subscription(
            Odometry, ego_odom_topic, self.ego_odom_callback, 10)
        self.opp_odom_sub = self.create_subscription(
            Odometry, opp_odom_topic, self.opp_odom_callback, 10)

        # Control timer (100Hz)
        self.control_timer = self.create_timer(0.01, self.control_callback)

        # Track progress (simple s approximation using accumulated distance)
        self.ego_prev_pos = None
        self.opp_prev_pos = None

        self.get_logger().info('Overtake Agent Node initialized')
        self.get_logger().info(f'  Ego scan: {ego_scan_topic}, Opp scan: {opp_scan_topic}')
        self.get_logger().info(f'  Ego drive: {ego_drive_topic}, Opp drive: {opp_drive_topic}')
        self.get_logger().info(f'  Agent0 speed factor: {self.agent0_speed_factor}')

    def _load_model(self, model_path):
        """Load OvertakePolicy model"""
        self.get_logger().info(f'Loading model: {model_path}')

        try:
            self.policy = OvertakePolicy(
                obs_dim_agent0=self.OBS_DIM_AGENT0,
                obs_dim_agent1=self.OBS_DIM_AGENT1,
                action_dim=self.ACTION_DIM,
                global_state_dim=self.GLOBAL_STATE_DIM,
                hidden_dims=[32, 32],
                device=self.device
            )

            if os.path.exists(model_path):
                self.policy.load(model_path)
                self.get_logger().info('  [OK] OvertakePolicy loaded')
                self.get_logger().info('  Agent 0: Stage 1 frozen expert (actor_1)')
                self.get_logger().info('  Agent 1: Overtake-trained agent (actor_2)')
                self.model_loaded = True
            else:
                self.get_logger().error(f'  [ERROR] Model not found: {model_path}')
                self.model_loaded = False

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.model_loaded = False

    def _process_scan(self, msg: LaserScan):
        """Process LiDAR scan data (same normalization as training)"""
        scan = np.array(msg.ranges, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=10.0, posinf=10.0, neginf=0.0)

        if len(scan) != self.num_beams:
            indices = np.linspace(0, len(scan) - 1, self.num_beams, dtype=int)
            scan = scan[indices]

        # Reverse scan direction (F110 convention, same as training)
        scan = scan[::-1]

        # Normalize to [0, 1]: clip to 10m, divide by 10 (same as RacingFrenetWrapper)
        scan = np.clip(scan, 0.0, 10.0) / 10.0

        return scan

    def ego_scan_callback(self, msg: LaserScan):
        self.ego_scan = self._process_scan(msg)

    def opp_scan_callback(self, msg: LaserScan):
        self.opp_scan = self._process_scan(msg)

    def ego_odom_callback(self, msg: Odometry):
        self.ego_pose[0] = msg.pose.pose.position.x
        self.ego_pose[1] = msg.pose.pose.position.y

        # Quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ego_pose[2] = np.arctan2(siny_cosp, cosy_cosp)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        # Normalized velocity (same as training: divide by 3.2)
        self.ego_velocity = vx / 3.2  # Use vx directly like training (linear_vels_x / 3.2)

        # Approximate s and vs (along track velocity)
        self.ego_vs = vx  # Raw velocity for relative calculations

        # Update s (accumulated distance)
        if self.ego_prev_pos is not None:
            dx = self.ego_pose[0] - self.ego_prev_pos[0]
            dy = self.ego_pose[1] - self.ego_prev_pos[1]
            self.ego_s += np.sqrt(dx**2 + dy**2)
        self.ego_prev_pos = [self.ego_pose[0], self.ego_pose[1]]

    def opp_odom_callback(self, msg: Odometry):
        self.opp_pose[0] = msg.pose.pose.position.x
        self.opp_pose[1] = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.opp_pose[2] = np.arctan2(siny_cosp, cosy_cosp)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        # Normalized velocity (same as training: divide by 3.2)
        self.opp_velocity = vx / 3.2  # Use vx directly like training

        # Raw velocity for relative calculations
        self.opp_vs = vx

        if self.opp_prev_pos is not None:
            dx = self.opp_pose[0] - self.opp_prev_pos[0]
            dy = self.opp_pose[1] - self.opp_prev_pos[1]
            self.opp_s += np.sqrt(dx**2 + dy**2)
        self.opp_prev_pos = [self.opp_pose[0], self.opp_pose[1]]

    def _build_observations(self):
        """
        Build asymmetric observations for both agents

        Agent 0 (ego): No opponent info (4324 dims)
        Agent 1 (opp): With opponent info (4336 dims)
        """
        if self.ego_scan is None or self.opp_scan is None:
            return None, None

        # Opponent info for Agent 1 (opp observes ego)
        delta_s = self.ego_s - self.opp_s  # Positive if ego ahead
        delta_vs = self.ego_vs - self.opp_vs
        ahead = 1.0 if delta_s > 0 else 0.0

        # Normalize
        delta_s_norm = np.clip(delta_s / 50.0, -1.0, 1.0)
        delta_vs_norm = np.clip(delta_vs / 3.0, -1.0, 1.0)

        # Agent 0 (ego): NO opponent info [1081 per frame]
        frame_ego = np.concatenate([
            self.ego_scan.flatten(),  # [1080]
            [self.ego_velocity]       # [1]
        ]).astype(np.float32)  # [1081]

        # Agent 1 (opp): WITH opponent info [1084 per frame]
        frame_opp = np.concatenate([
            self.opp_scan.flatten(),  # [1080]
            [self.opp_velocity],      # [1]
            [delta_s_norm],           # [1]
            [delta_vs_norm],          # [1]
            [ahead]                   # [1]
        ]).astype(np.float32)  # [1084]

        # Add to frame buffers
        self.frame_buffer_ego.append(frame_ego)
        self.frame_buffer_opp.append(frame_opp)

        # Pad if not full
        while len(self.frame_buffer_ego) < self.frame_stack:
            self.frame_buffer_ego.append(frame_ego)
        while len(self.frame_buffer_opp) < self.frame_stack:
            self.frame_buffer_opp.append(frame_opp)

        # Stack frames
        obs_ego_stacked = np.concatenate(list(self.frame_buffer_ego))  # [4324]
        obs_opp_stacked = np.concatenate(list(self.frame_buffer_opp))  # [4336]

        return obs_ego_stacked, obs_opp_stacked

    def control_callback(self):
        """Main control loop"""
        if not self.model_loaded:
            return

        # Build observations
        obs_ego, obs_opp = self._build_observations()
        if obs_ego is None or obs_opp is None:
            return

        # Select actions (deterministic for evaluation)
        action_ego, _ = self.policy.select_action(
            obs_ego[:self.OBS_DIM_AGENT0], agent_idx=0, deterministic=True)
        action_opp, _ = self.policy.select_action(
            obs_opp, agent_idx=1, deterministic=True)

        # Apply speed handicap to ego (Agent 0)
        action_ego[1] = action_ego[1] * self.agent0_speed_factor

        # Publish ego drive command
        ego_msg = AckermannDriveStamped()
        ego_msg.header.stamp = self.get_clock().now().to_msg()
        ego_msg.header.frame_id = 'ego_racecar/base_link'
        ego_msg.drive.steering_angle = float(np.clip(action_ego[0], -0.4189, 0.4189))
        ego_msg.drive.speed = float(np.clip(action_ego[1], 0.01, 3.2))
        self.ego_drive_pub.publish(ego_msg)

        # Publish opp drive command
        opp_msg = AckermannDriveStamped()
        opp_msg.header.stamp = self.get_clock().now().to_msg()
        opp_msg.header.frame_id = 'opp_racecar/base_link'
        opp_msg.drive.steering_angle = float(np.clip(action_opp[0], -0.4189, 0.4189))
        opp_msg.drive.speed = float(np.clip(action_opp[1], 0.01, 3.2))
        self.opp_drive_pub.publish(opp_msg)


def main(args=None):
    rclpy.init(args=args)

    node = OvertakeAgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
