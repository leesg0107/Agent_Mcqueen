"""
Gym wrappers for F1TENTH RL (f1tenth_RL style).
간결하고 검증된 구현.
"""

import gym
from gym import spaces
from copy import copy
import numpy as np
from sklearn.neighbors import KDTree


# Constants
NUM_BEAMS = 1080
DTYPE = np.float32


class DelayedAction(gym.Wrapper):
    """Simulate action delay and dropout (domain randomization)."""

    def __init__(self, env, delay_prob=0.1, drop_prob=0.05):
        super().__init__(env)
        self.delay_prob = delay_prob
        self.drop_prob = drop_prob
        self.last_action = None
        self.last_executed_action = None

    def reset(self, **kwargs):
        self.last_action = None
        self.last_executed_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        # Randomly delay the action
        if self.last_action is not None and np.random.random() < self.delay_prob:
            action_to_take = self.last_action
        else:
            action_to_take = action

        # Randomly drop the action
        if np.random.random() < self.drop_prob:
            action_to_take = None

        # Remember current action
        self.last_action = action

        # If action was dropped, repeat last executed action
        if action_to_take is None:
            action_to_take = self.last_executed_action if self.last_executed_action is not None else self.env.action_space.sample()

        obs, reward, done, info = self.env.step(action_to_take)

        # Remember last executed action
        self.last_executed_action = action_to_take

        return obs, reward, done, info


class LidarRandomizer(gym.ObservationWrapper):
    """Add noise to LiDAR for robustness (domain randomization)."""

    def __init__(self, env, epsilon=0.05, zone_p=0.1, extreme_p=0.05):
        super().__init__(env)
        self.epsilon = epsilon
        self.zone_p = zone_p
        self.extreme_p = extreme_p

    def observation(self, obs):
        lidar_data = obs["scans"]

        # Handle both 1D and 2D arrays
        if isinstance(lidar_data, np.ndarray):
            lidar_data = lidar_data.flatten()
        else:
            lidar_data = np.array(lidar_data).flatten()

        # Uniform noise
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=lidar_data.shape)
        lidar_data = lidar_data + noise

        # Randomly modify zones (20% of readings)
        if np.random.random() < self.zone_p:
            size = int(len(lidar_data) * 0.2)
            start = np.random.randint(0, len(lidar_data) - size)
            end = start + size
            change = np.random.uniform(-0.1, 0.1)
            lidar_data[start:end] += change

        # Randomly set extreme values
        if np.random.random() < self.extreme_p:
            index = np.random.randint(len(lidar_data))
            lidar_data[index] = np.random.choice([0, 1])

        # Clip to [0, 1]
        lidar_data = np.clip(lidar_data, 0, 1)

        obs["scans"] = lidar_data
        return obs


class ActionRandomizer(gym.ActionWrapper):
    """Add noise to actions for robustness (domain randomization)."""

    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=action.shape)
        action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return action

class RewardWrapper(gym.Wrapper):
    """
    Reward function (Paper-proven design).
    Simple and effective reward based on successful paper implementation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_action = None

    def reward(self, obs):
        """
        f1tenth_RL reward function (proven to work).

        Key components:
        - Base reward: +0.01 (survival)
        - Stop penalty: -2.0 (force movement)
        - Forward progress: +1.0 * vs
        - Lateral drift: -0.01 * |vd|
        - Centerline distance: -0.05 * |d|
        - Angular velocity: -0.05 * |w|
        - Wall proximity: -0.01 * penalty
        - Collision: -1000.0
        """
        ego_idx = 0
        vs = obs["linear_vels_s"][ego_idx]
        vd = obs["linear_vels_d"][ego_idx]
        d = obs["poses_d"][ego_idx]
        w = obs["ang_vels_z"][ego_idx]

        # Base reward (small positive for survival)
        reward = 0.01

        # Stop penalty (f1tenth_RL original)
        # Speed range is [0, 1] m/s
        if abs(obs["linear_vels_x"][ego_idx]) <= 0.25:
            reward -= 2.0

        # Forward progress (f1tenth_RL original: use vs directly)
        reward += 1.0 * vs

        # Lateral drift penalty
        reward -= 0.01 * abs(vd)

        # Collision penalty
        if self.env.collisions[0]:
            reward -= 1000.0

        # Centerline distance penalty
        reward -= 0.05 * abs(d)

        # Angular velocity penalty (discourage sharp turns)
        reward -= 0.05 * abs(w)

        # Wall proximity penalty
        min_distance = abs(np.min(obs["scans"]))
        distance_threshold = 0.5
        if min_distance < distance_threshold:
            reward -= 0.01 * (distance_threshold - min_distance)

        return reward

    def step(self, action):
        # F110Env expects 2D array [[steering, speed]]
        if action.ndim == 1:
            action = action.reshape(1, -1)

        obs, _, done, info = self.env.step(action)

        # Compute custom reward
        new_reward = self.reward(obs)

        # Add info for monitoring
        info['poses_s'] = obs['poses_s'][0] if hasattr(obs['poses_s'], '__getitem__') else obs['poses_s']
        info['collision'] = int(not self.env.collisions[0])
        # Success = 1 lap completed AND no collision
        info['is_success'] = bool(info.get('lap_count', [0])[0] >= 1 and not self.env.collisions[0])

        return obs, new_reward.item(), done, info


class FrenetObsWrapper(gym.ObservationWrapper):
    """
    Convert to Frenet coordinates (f1tenth_RL style).
    Adds s, d, vs, vd to observation.
    """

    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        # Use map_data from environment (multi-map F110Env provides this)
        if hasattr(env, 'map_data') and env.map_data is not None:
            self.map_data = env.map_data.to_numpy()
        else:
            raise ValueError("Environment does not have map_data attribute")

        self.kdtree = KDTree(self.map_data[:, 1:3])

        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Box(0, self.num_agents - 1, (1,), np.int32),
                "scans": spaces.Box(0, 1, (NUM_BEAMS,), DTYPE),
                "poses_x": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
                "poses_y": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
                "poses_theta": spaces.Box(-2 * np.pi, 2 * np.pi, (self.num_agents,), DTYPE),
                "linear_vels_x": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "linear_vels_y": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "ang_vels_z": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "collisions": spaces.Box(0, 1, (self.num_agents,), DTYPE),
                "lap_times": spaces.Box(0, 1e6, (self.num_agents,), DTYPE),
                "lap_counts": spaces.Box(0, 999, (self.num_agents,), np.int32),
                "poses_s": spaces.Box(-1000, 1000, (1,), DTYPE),
                "poses_d": spaces.Box(-1000, 1000, (1,), DTYPE),
                "linear_vels_s": spaces.Box(-10, 10, (1,), DTYPE),
                "linear_vels_d": spaces.Box(-10, 10, (1,), DTYPE),
                "linear_vel": spaces.Box(0, 1, (self.num_agents,), DTYPE),
            }
        )

    def observation(self, obs):
        # Handle tuple return from reset (old gym API)
        if isinstance(obs, tuple):
            obs = obs[0]

        new_obs = {}

        # Convert all values to numpy arrays
        for key, value in obs.items():
            if isinstance(value, (list, tuple)):
                new_obs[key] = np.array(value)
            else:
                new_obs[key] = value

        # ⭐ CRITICAL FIX: Update map_data and kdtree when map changes!
        # The environment's map_data changes on every reset() via _set_random_map()
        # We MUST rebuild the KDTree with the new map's centerline for correct Frenet coordinates
        # ALWAYS update to ensure we catch all map changes (previous conditional check was unreliable)
        if hasattr(self.env, 'map_data') and self.env.map_data is not None:
            current_map_data = self.env.map_data.to_numpy()
            self.map_data = current_map_data
            self.kdtree = KDTree(self.map_data[:, 1:3])

        # Convert to Frenet coordinates
        frenet_coords = convert_to_frenet(
            float(new_obs["poses_x"][0]) if hasattr(new_obs["poses_x"], '__getitem__') else float(new_obs["poses_x"]),
            float(new_obs["poses_y"][0]) if hasattr(new_obs["poses_y"], '__getitem__') else float(new_obs["poses_y"]),
            float(new_obs["linear_vels_x"][0]) if hasattr(new_obs["linear_vels_x"], '__getitem__') else float(new_obs["linear_vels_x"]),
            float(new_obs["poses_theta"][0]) if hasattr(new_obs["poses_theta"], '__getitem__') else float(new_obs["poses_theta"]),
            self.map_data,
            self.kdtree
        )

        new_obs["poses_s"] = np.array([frenet_coords[0]], dtype=DTYPE).reshape((1,))
        new_obs["poses_d"] = np.array([frenet_coords[1]], dtype=DTYPE).reshape((1,))
        new_obs["linear_vels_s"] = np.array([frenet_coords[2]], dtype=DTYPE).reshape((1,))
        new_obs["linear_vels_d"] = np.array([frenet_coords[3]], dtype=DTYPE).reshape((1,))

        # Scale scans to [0, 1] and add noise for far readings
        scans = np.array(new_obs["scans"], dtype=DTYPE).flatten()
        clipped_indices = np.where(scans >= 10)
        noise = np.random.uniform(-0.5, 0, clipped_indices[0].shape).astype(DTYPE)

        scans = np.clip(scans, None, 10)
        scans[clipped_indices] += noise
        scans /= 10.0
        new_obs["scans"] = scans

        # Normalized linear velocity (same as Stage 1: 3.2 m/s max speed)
        linear_vels_x = np.array(new_obs["linear_vels_x"], dtype=DTYPE)
        new_obs["linear_vel"] = (linear_vels_x / 3.2).astype(DTYPE)

        return new_obs


def get_closest_point_index(x, y, kdtree):
    """Find closest waypoint index using KDTree."""
    # Safety check for inf/nan values
    if not np.isfinite(x) or not np.isfinite(y):
        return 0  # Return first waypoint as fallback

    _, indices = kdtree.query(np.array([[x, y]]), k=1)
    closest_point_index = indices[0, 0]
    return closest_point_index


def convert_to_frenet(x, y, vel_magnitude, pose_theta, map_data, kdtree):
    """
    Convert Cartesian coordinates to Frenet coordinates.

    Returns:
        s: progress along track
        d: lateral deviation from centerline
        vs: velocity along track
        vd: lateral velocity
    """
    # Safety check for inf/nan values
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(vel_magnitude) or not np.isfinite(pose_theta):
        # Return safe default values
        return 0.0, 0.0, 0.0, 0.0

    closest_point_index = get_closest_point_index(x, y, kdtree)
    closest_point = map_data[closest_point_index]
    s_m, x_m, y_m, psi_rad = closest_point[0:4]

    dx = x - x_m
    dy = y - y_m

    s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
    d = dx * np.cos(psi_rad) + dy * np.sin(psi_rad)

    # ⭐ CRITICAL FIX: vs (velocity along track), vd (lateral velocity)
    # Original f1tenth_RL uses sin/cos (not cos/sin)
    vs = vel_magnitude * np.sin(pose_theta - psi_rad)
    vd = vel_magnitude * np.cos(pose_theta - psi_rad)

    return s, d, vs, vd
