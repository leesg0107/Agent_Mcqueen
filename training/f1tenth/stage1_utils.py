"""
Utility functions for Stage 1 single-agent environment creation
"""

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers import FilterObservation, TimeLimit, FlattenObservation, FrameStack, RescaleAction

from wrappers import FrenetObsWrapper, RewardWrapper, LidarRandomizer, ActionRandomizer, DelayedAction


def create_stage1_env(maps, num_beams=1080, seed=42, domain_randomize=True):
    """
    Create Stage 1 single-agent environment with all wrappers.

    Args:
        maps: List of map indices (e.g., [0, 1, 2]) or single map index
        num_beams: Number of LiDAR beams
        seed: Random seed
        domain_randomize: Enable domain randomization

    Wrapper order:
    1. FrenetObsWrapper (Frenet coordinate conversion)
    2. RescaleAction (action scaling)
    3. RewardWrapper (reward computation)
    4. FilterObservation (scans + linear_vel only)
    5. TimeLimit (episode limit)
    6. Domain Randomization (LiDAR, Action, Delay)
    7. FlattenObservation (Dict -> 1D array)
    8. FrameStack (4 frames)
    9. DummyVecEnv (Vectorize)
    10. VecNormalize (reward normalization)
    """
    # 1. Base environment (single agent)
    env = gym.make(
        "f110_gym:f110-v0",
        maps=maps,
        num_agents=1,
        num_beams=num_beams,
        seed=seed,
    )

    # Set numpy seed for domain randomization
    np.random.seed(seed)

    # 2. Frenet coordinates (adds s, d, vs, vd to observation)
    env = FrenetObsWrapper(env)

    # 3. RescaleAction
    # Agent output: [-1, 1] x [-1, 1]
    # After RescaleAction: steering [-1.0, 1.0] x speed [0.0, 1.0]
    env = RescaleAction(env, np.array([-1.0, 0.0]), np.array([1.0, 1.0]))

    # 4. Custom reward function (AFTER RescaleAction so it sees real speeds)
    env = RewardWrapper(env)

    # 5. Filter observation to only scans + linear_vel
    env = FilterObservation(env, filter_keys=["scans", "linear_vel"])

    # 6. Episode time limit
    env = TimeLimit(env, max_episode_steps=10000)

    # 7. Domain randomization (optional)
    if domain_randomize:
        env = LidarRandomizer(env, epsilon=0.05, zone_p=0.1, extreme_p=0.05)
        env = ActionRandomizer(env, epsilon=0.1)
        env = DelayedAction(env, delay_prob=0.1, drop_prob=0.05)

    # 8. Flatten Dict observation to 1D array
    env = FlattenObservation(env)

    # 9. Frame Stack (4 frames)
    env = FrameStack(env, 4)

    # 10. Vectorize (SB3 requires vectorized env)
    env = DummyVecEnv([lambda: env])

    # 11. VecNormalize (reward normalization only, obs already normalized)
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    return env
