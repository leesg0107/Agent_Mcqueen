"""
Racing Wrappers for Multi-Agent Competitive Racing

Key components:
- RacingFrenetWrapper: Multi-agent Frenet coordinate conversion
- CompetitiveReward: Competitive reward function for overtaking
- GlobalStateWrapper: Constructs global state for centralized critic
"""

import gym
import numpy as np
from gym import spaces
from sklearn.neighbors import KDTree
from typing import Tuple, Dict


NUM_BEAMS = 1080
DTYPE = np.float32


class RacingFrenetWrapper(gym.ObservationWrapper):
    """
    Multi-agent Frenet Coordinate Wrapper for Racing

    Converts Cartesian coordinates to Frenet coordinates for all agents
    """

    def __init__(self, env):
        super().__init__(env)

        self.num_agents = env.num_agents

        # Map data for Frenet conversion
        if hasattr(env, 'map_data') and env.map_data is not None:
            self.map_data = env.map_data.to_numpy()
        else:
            raise ValueError("Environment does not have map_data attribute")

        self.kdtree = KDTree(self.map_data[:, 1:3])

        # Update observation space
        self.observation_space = spaces.Dict({
            "ego_idx": spaces.Box(0, self.num_agents - 1, (1,), np.int32),
            "scans": spaces.Box(0, 1, (self.num_agents, NUM_BEAMS), DTYPE),
            "poses_x": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
            "poses_y": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
            "poses_theta": spaces.Box(-2 * np.pi, 2 * np.pi, (self.num_agents,), DTYPE),
            "linear_vels_x": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            "linear_vels_y": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            "ang_vels_z": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            "collisions": spaces.Box(0, 1, (self.num_agents,), DTYPE),
            "lap_times": spaces.Box(0, 1e6, (self.num_agents,), DTYPE),
            "lap_counts": spaces.Box(0, 999, (self.num_agents,), np.int32),
            "poses_s": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
            "poses_d": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
            "linear_vels_s": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            "linear_vels_d": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            "linear_vel": spaces.Box(0, 1, (self.num_agents,), DTYPE),
        })

    def step(self, action):
        """
        Step with multi-agent actions, preserving all agents' scans

        Override F110Env.step() to preserve multi-agent scans
        """
        # ⚠️ IMPORTANT: Use self.env.step() to respect wrapper chain (e.g., RescaleAction)
        # DO NOT call sim.step() directly as it bypasses all wrappers!

        # Access unwrapped F110Env
        f110_env = self.env
        while hasattr(f110_env, 'env'):
            f110_env = f110_env.env

        # Call simulator step to get full multi-agent observations
        # action is already processed by RescaleAction wrapper if it's in the chain
        obs_raw = f110_env.sim.step(action)

        # Update time
        f110_env.current_time += f110_env.timestep

        # CRITICAL: Update F110Env state for lap tracking
        # _update_state updates poses_x, poses_y, poses_theta, collisions
        f110_env._update_state(obs_raw)

        # CRITICAL: Call _check_done to update lap_times and lap_counts
        # Lap tracking logic is inside _check_done() (lines 400-422 in f110_env.py)
        _, _ = f110_env._check_done()

        # Now add updated lap times and counts to observation
        obs_raw['lap_times'] = f110_env.lap_times
        obs_raw['lap_counts'] = f110_env.lap_counts

        # ==================== ASYMMETRIC TRAINING: Episode Termination ====================
        # CRITICAL: Episode ends when Agent 1 (learner) finishes ONLY
        # Agent 0 (frozen expert) completion does NOT terminate episode
        # Reason: Agent 1 needs full episode experience to learn completing laps
        #         If Agent 0 finishes first (always), Agent 1 never learns to finish

        # FIX: Episode ends ONLY on collision, NOT on lap complete
        # Reason: F110 gym lap tracking doesn't check direction → backward lap complete is possible
        #         Agent 1 was learning "reverse to finish fast" strategy (vs < 0)
        #         Now: collision ends episode, lap complete gives reward but continues
        done = False

        # Check if Agent 1 collided (ONLY collision, not lap complete)
        agent1_collision = obs_raw['collisions'][1] > 0

        if agent1_collision:
            done = True

        # Build info
        info = {
            'checkpoint_done': done,
            'max_s': f110_env.map_max_s if hasattr(f110_env, 'map_max_s') else 0,
            'lap_count': obs_raw['lap_counts'],
            "is_success": obs_raw['lap_counts'][0] >= 1
        }

        # Process multi-agent scans (KEEP ALL, don't reduce to just agent 0)
        # obs_raw['scans'] is a list: [scan_agent0, scan_agent1]
        scans_array = []
        for agent_scan in obs_raw['scans']:
            scans_array.append(np.array(agent_scan)[::-1])  # Reverse (F110 convention)
        obs_raw['scans'] = np.array(scans_array, dtype=DTYPE)  # Shape: [num_agents, num_beams]

        # Convert to numpy arrays
        for key, value in obs_raw.items():
            if isinstance(value, list):
                obs_raw[key] = np.array(value)

        # Format observation (convert to proper types)
        obs = {}
        for key, value in obs_raw.items():
            if key not in ['ego_idx', 'lap_counts']:
                # Clip values to avoid overflow when converting to float32
                value_array = np.array(value, dtype=np.float64)
                value_clipped = np.clip(value_array, -1e10, 1e10)
                obs[key] = value_clipped.astype(DTYPE)
            else:
                obs[key] = np.array(value, dtype=np.int32)

        # Apply observation wrapper (Frenet conversion, etc.)
        obs = self.observation(obs)

        # Set render_obs for visualization (F110Env expects this)
        f110_env.render_obs = obs_raw

        # Return in gym 0.x API format: (obs, reward, done, info)
        return obs, 0, done, info

    def reset(self, side_by_side=False, **kwargs):
        """
        Reset with configurable starting positions

        Args:
            side_by_side: If True, both agents start at same s-coordinate (racing start)
                         If False, Agent 0 starts ahead (training for overtaking)

        Default (side_by_side=False) for training:
        - Agent 0 starts AHEAD (higher s-coordinate)
        - Agent 1 must learn to overtake the slower opponent
        - ~10-15 waypoints separation (approximately 5-10 meters)
        """
        # Note: Map data is already initialized in __init__
        # If map changes during training, update in observation() method

        base_idx = np.random.randint(30, 50)
        max_idx = len(self.map_data) - 1

        if side_by_side:
            # SIDE-BY-SIDE: Same s-coordinate, lateral offset (racing start)
            start_idx = min(base_idx, max_idx)
            waypoint = self.map_data[start_idx]
            x_center = waypoint[1]
            y_center = waypoint[2]
            psi = waypoint[3]

            # Track direction: psi + pi/2
            # Lateral direction (perpendicular to track): psi
            lateral_offset = 0.8  # meters
            lat_x = np.cos(psi)  # lateral (side-to-side) direction
            lat_y = np.sin(psi)

            poses = np.array([
                [
                    x_center + lateral_offset * lat_x,  # Agent 0: one side
                    y_center + lateral_offset * lat_y,
                    psi + np.pi/2  # heading along track
                ],
                [
                    x_center - lateral_offset * lat_x,  # Agent 1: other side
                    y_center - lateral_offset * lat_y,
                    psi + np.pi/2  # heading along track
                ]
            ])
        else:
            # AHEAD-BEHIND: Agent 0 starts ahead (for overtaking training)
            ahead_offset = np.random.randint(10, 16)  # 10-15 waypoints ahead
            start_idx_agent1 = base_idx
            start_idx_agent0 = base_idx + ahead_offset

            # Ensure indices are valid
            if start_idx_agent0 > max_idx:
                start_idx_agent0 = max_idx
            if start_idx_agent1 > max_idx:
                start_idx_agent1 = max_idx

            # Get separate waypoints for each agent
            # map_data columns: [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2, width]
            waypoint_0 = self.map_data[start_idx_agent0]  # Agent 0 ahead
            waypoint_1 = self.map_data[start_idx_agent1]  # Agent 1 behind

            # Agent 0 position (ahead, on centerline)
            x_0 = waypoint_0[1]
            y_0 = waypoint_0[2]
            psi_0 = waypoint_0[3]

            # Agent 1 position (behind, on centerline)
            x_1 = waypoint_1[1]
            y_1 = waypoint_1[2]
            psi_1 = waypoint_1[3]

            # Small lateral offset to prevent exact overlap (helps with collision detection)
            lateral_noise = np.random.uniform(-0.2, 0.2)  # ±20cm

            poses = np.array([
                [
                    x_0 + lateral_noise * np.cos(psi_0),  # x coordinate
                    y_0 + lateral_noise * np.sin(psi_0),  # y coordinate
                    psi_0 + np.pi/2 + np.random.uniform(-np.pi/72, np.pi/72)  # theta along track
                ],
                [
                    x_1 - lateral_noise * np.cos(psi_1),  # x coordinate (opposite offset)
                    y_1 - lateral_noise * np.sin(psi_1),  # y coordinate
                    psi_1 + np.pi/2 + np.random.uniform(-np.pi/72, np.pi/72)  # theta along track
                ]
            ])

        # Reset with custom competitive poses
        # We need to manually reset to preserve multi-agent scans
        f110_env = self.env
        while hasattr(f110_env, 'env'):
            f110_env = f110_env.env

        # Manual reset (simplified version of F110Env.reset)
        f110_env.current_time = 0.0
        f110_env.collisions = np.zeros((f110_env.num_agents,))
        f110_env.lap_times = np.zeros((f110_env.num_agents,))
        f110_env.lap_counts = np.zeros((f110_env.num_agents,), dtype=np.int32)

        # CRITICAL: Update lap tracking start positions
        # F110Env._check_done() uses these to determine lap completion
        f110_env.num_toggles = 0
        f110_env.near_start = True
        f110_env.near_starts = np.array([True] * f110_env.num_agents)
        f110_env.toggle_list = np.zeros((f110_env.num_agents,))
        f110_env.start_xs = poses[:, 0]
        f110_env.start_ys = poses[:, 1]
        f110_env.start_thetas = poses[:, 2]
        f110_env.start_rot = np.array([[np.cos(-f110_env.start_thetas[f110_env.ego_idx]),
                                        -np.sin(-f110_env.start_thetas[f110_env.ego_idx])],
                                       [np.sin(-f110_env.start_thetas[f110_env.ego_idx]),
                                        np.cos(-f110_env.start_thetas[f110_env.ego_idx])]])

        f110_env.sim.reset(poses)

        # Get initial observation with zero action
        action = np.zeros((f110_env.num_agents, 2))
        obs, _, _, _ = self.step(action)  # Use our custom step that preserves multi-agent scans

        return obs

    def observation(self, obs):
        """Convert multi-agent observations to Frenet coordinates"""
        if isinstance(obs, tuple):
            obs = obs[0]

        new_obs = {}

        # Convert all values to numpy arrays
        for key, value in obs.items():
            if isinstance(value, (list, tuple)):
                new_obs[key] = np.array(value)
            else:
                new_obs[key] = value

        # Update map data if changed (for multi-map training)
        if hasattr(self.env, 'map_data') and self.env.map_data is not None:
            current_map_data = self.env.map_data.to_numpy()
            self.map_data = current_map_data
            self.kdtree = KDTree(self.map_data[:, 1:3])

        # Convert to Frenet for all agents
        num_agents = len(new_obs["poses_x"])

        poses_s_list = []
        poses_d_list = []
        vels_s_list = []
        vels_d_list = []

        for agent_idx in range(num_agents):
            x = float(new_obs["poses_x"][agent_idx])
            y = float(new_obs["poses_y"][agent_idx])
            vel = float(new_obs["linear_vels_x"][agent_idx])
            theta = float(new_obs["poses_theta"][agent_idx])

            s, d, vs, vd = self._convert_to_frenet(x, y, vel, theta)

            poses_s_list.append(s)
            poses_d_list.append(d)
            vels_s_list.append(vs)
            vels_d_list.append(vd)

        new_obs["poses_s"] = np.array(poses_s_list, dtype=DTYPE)
        new_obs["poses_d"] = np.array(poses_d_list, dtype=DTYPE)
        new_obs["linear_vels_s"] = np.array(vels_s_list, dtype=DTYPE)
        new_obs["linear_vels_d"] = np.array(vels_d_list, dtype=DTYPE)

        # Process scans (normalize to [0, 1])
        scans = new_obs["scans"]
        if isinstance(scans, np.ndarray) and len(scans.shape) > 1:
            processed_scans = []
            for agent_idx in range(num_agents):
                agent_scans = np.array(scans[agent_idx], dtype=DTYPE).flatten()
                agent_scans = np.clip(agent_scans, None, 10) / 10.0
                processed_scans.append(agent_scans)
            new_obs["scans"] = np.array(processed_scans, dtype=DTYPE)
        else:
            scans = np.array(scans, dtype=DTYPE).flatten()
            new_obs["scans"] = np.clip(scans, None, 10) / 10.0

        # Normalized linear velocity (same as Stage 1: 3.2 m/s max speed)
        linear_vels_x = np.array(new_obs["linear_vels_x"], dtype=DTYPE)
        new_obs["linear_vel"] = (linear_vels_x / 3.2).astype(DTYPE)

        return new_obs

    def _convert_to_frenet(self, x, y, vel_magnitude, pose_theta):
        """Convert Cartesian to Frenet coordinates"""
        # Check all inputs for NaN/inf
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(vel_magnitude) or not np.isfinite(pose_theta):
            return 0.0, 0.0, 0.0, 0.0

        _, indices = self.kdtree.query(np.array([[x, y]]), k=1)
        closest_idx = indices[0, 0]
        closest_point = self.map_data[closest_idx]
        s_m, x_m, y_m, psi_rad = closest_point[0:4]

        # Check map data validity
        if not np.isfinite(psi_rad):
            return 0.0, 0.0, 0.0, 0.0

        dx = x - x_m
        dy = y - y_m

        # Frenet coordinate transformation (same as Stage 1)
        s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
        d = dx * np.cos(psi_rad) + dy * np.sin(psi_rad)

        # Velocity in Frenet frame (same as Stage 1)
        angle_diff = pose_theta - psi_rad
        vs = vel_magnitude * np.sin(angle_diff)
        vd = vel_magnitude * np.cos(angle_diff)

        # Final NaN check
        if not np.isfinite(s) or not np.isfinite(d) or not np.isfinite(vs) or not np.isfinite(vd):
            return 0.0, 0.0, 0.0, 0.0

        return s, d, vs, vd


class CompetitiveReward(gym.Wrapper):
    """
    Competitive Reward Function for Overtake Training

    Design Philosophy:
    - Both agents get Stage 1 base rewards → maintain driving ability
    - Agent 1 gets ADDITIONAL competitive rewards → learn overtaking/winning

    Base rewards (both agents, Stage 1 scale):
    - Forward progress: +10*vs
    - Stop penalty: -2
    - Centerline, angular velocity, collision penalties

    Competitive bonuses (Agent 1 only):
    - Overtaking Agent 0: +500 (one-time event)
    - Winning race (lap_count first): +1000 (terminal)
    """

    def __init__(self, env, position_weight=0.5, overtake_bonus=500.0, agent_collision_penalty=0.0):
        super().__init__(env)
        # Legacy parameters (kept for compatibility, not used in sparse rewards)
        self.position_weight = position_weight
        self.overtake_bonus = overtake_bonus
        self.agent_collision_penalty = agent_collision_penalty

        # Track previous relative positions for overtake detection
        self.prev_relative_positions = None

        # Track if win/loss reward has been given
        self.win_loss_given = False

    def reset(self, **kwargs):
        self.prev_relative_positions = None
        self.win_loss_given = False  # Reset win/loss flag
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step with multi-agent actions"""
        obs, _, done, info = self.env.step(action)

        # Compute competitive rewards (now with lap_count for win/loss)
        rewards = self._compute_rewards(obs, info, done)

        # Add info for monitoring
        info['competitive_rewards'] = rewards
        info['position_agent0'] = obs['poses_s'][0]
        info['position_agent1'] = obs['poses_s'][1]

        return obs, rewards, done, info

    def _compute_rewards(self, obs, info, done) -> np.ndarray:
        """
        Competitive Rewards = Stage 1 Base Rewards + Competitive Bonuses

        Philosophy:
        - Both agents get SAME base rewards (Stage 1 driving behavior)
        - Agent 1 gets ADDITIONAL competitive rewards (overtaking, winning)
        - This preserves Stage 1 ability while learning competition

        Base rewards (both agents, same as Stage 1):
        - Forward progress: +10*vs
        - Stop penalty: -2
        - Other penalties: centerline, angular velocity, etc.

        Competitive bonuses (Agent 1 only):
        - Overtaking Agent 0: +500 (one-time)
        - Finishing first (lap_count=1): +1000 (terminal)
        """
        num_agents = 2
        rewards = np.zeros(num_agents, dtype=np.float32)

        # ==================== Base Rewards (BOTH agents, Stage 1 behavior) ====================
        for agent_idx in range(num_agents):
            vs = obs["linear_vels_s"][agent_idx]
            vd = obs["linear_vels_d"][agent_idx]
            d = obs["poses_d"][agent_idx]
            w = obs["ang_vels_z"][agent_idx]

            reward = 0.0

            # Stop penalty (Stage 1: 0.25 m/s threshold, -2.0 penalty)
            if abs(obs["linear_vels_x"][agent_idx]) <= 0.25:
                reward -= 2.0

            # Forward progress (Stage 1 scale: +10*vs)
            reward += 10.0 * vs

            # Reverse driving penalty (CRITICAL: Prevent backward motion)
            if vs < 0:
                reward -= 10.0  # Strong penalty for going backward

            # Centerline deviation penalty
            reward -= 0.05 * abs(d)

            # Lateral velocity penalty
            reward -= 0.01 * abs(vd)

            # Angular velocity penalty (INCREASED 20x to prevent spinning)
            reward -= 2.0 * abs(w)

            # Wall collision penalty
            if obs["collisions"][agent_idx] > 0:
                reward -= 5.0

            rewards[agent_idx] = reward

        # ==================== Competitive Bonuses (Agent 1 ONLY) ====================
        s_agent0 = obs["poses_s"][0]
        s_agent1 = obs["poses_s"][1]
        relative_position = s_agent0 - s_agent1  # Positive if agent0 ahead

        # Overtaking bonus
        if self.prev_relative_positions is not None:
            # Agent 1 overtook Agent 0
            if self.prev_relative_positions > 0 and relative_position < 0:
                rewards[1] += 500.0

        self.prev_relative_positions = relative_position

        # Win bonus (lap_count-based)
        if done and not self.win_loss_given:
            lap_counts = info.get('lap_count', [0, 0])

            # Agent 1 finished first → WIN
            if lap_counts[1] >= 1 and lap_counts[0] < 1:
                rewards[1] += 1000.0

            self.win_loss_given = True

        return rewards


class GlobalStateWrapper(gym.Wrapper):
    """
    Wrapper to construct global state for centralized critic

    Global state includes:
    - All agents' Frenet positions (s, d)
    - All agents' velocities (vs, vd)
    - Relative position (delta_s)
    - Distance between agents
    - Collision status
    """

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents

    def get_global_state(self, obs: Dict) -> np.ndarray:
        """
        Construct global state from observation

        Args:
            obs: Multi-agent observation dict

        Returns:
            global_state: [global_state_dim] array
        """
        # Extract relevant information
        s_values = obs["poses_s"]  # [num_agents]
        d_values = obs["poses_d"]  # [num_agents]
        vs_values = obs["linear_vels_s"]  # [num_agents]
        vd_values = obs["linear_vels_d"]  # [num_agents]

        # Relative position
        relative_s = s_values[0] - s_values[1]

        # Distance between agents
        distance = np.sqrt((obs["poses_x"][0] - obs["poses_x"][1])**2 +
                          (obs["poses_y"][0] - obs["poses_y"][1])**2)

        # Collision flags
        collisions = obs["collisions"]  # [num_agents]

        # Construct global state
        global_state = np.concatenate([
            s_values,      # [2] - Progress along track
            d_values,      # [2] - Lateral deviation
            vs_values,     # [2] - Forward velocity
            vd_values,     # [2] - Lateral velocity
            [relative_s],  # [1] - Relative position
            [distance],    # [1] - Distance between agents
            collisions     # [2] - Collision flags
        ]).astype(np.float32)

        # Global state dim: 2 + 2 + 2 + 2 + 1 + 1 + 2 = 12

        return global_state
