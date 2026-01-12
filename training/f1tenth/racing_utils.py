"""
Utility functions for racing environment creation
"""

import gym
import numpy as np
from gym.wrappers import FilterObservation, TimeLimit, FlattenObservation, FrameStack, RescaleAction
from racing_wrappers import RacingFrenetWrapper, CompetitiveReward, GlobalStateWrapper
from wrappers import LidarRandomizer, ActionRandomizer, DelayedAction


def create_racing_env(
    maps,
    num_beams=1080,
    seed=42,
    domain_randomize=True,
    position_weight=0.5,
    overtake_bonus=5.0
):
    """
    Create racing environment for competitive 2-agent racing

    Args:
        maps: List of map indices
        num_beams: Number of LiDAR beams
        seed: Random seed
        domain_randomize: Enable domain randomization
        position_weight: Weight for relative position reward
        overtake_bonus: Bonus for overtaking

    Returns:
        env: Racing environment with Frenet coordinates and competitive rewards
    """
    # Base environment (2 agents)
    env = gym.make(
        "f110_gym:f110-v0",
        maps=maps,
        num_agents=2,  # Multi-agent
        num_beams=num_beams,
        seed=seed,
    )

    # Set numpy seed
    np.random.seed(seed)

    # Frenet coordinates (MUST come first - will call sim.step directly)
    env = RacingFrenetWrapper(env)

    # ⚠️ RescaleAction does NOT work with RacingFrenetWrapper!
    # RacingFrenetWrapper.step() calls sim.step() directly, bypassing all wrappers
    # Solution: RacingFrenetWrapper must handle action rescaling internally
    # Agents output physical range directly: [-0.4189, 0.4189] x [0.01, 3.2]

    # Competitive reward
    env = CompetitiveReward(
        env,
        position_weight=position_weight,
        overtake_bonus=overtake_bonus
    )

    # Global state wrapper
    env = GlobalStateWrapper(env)

    # Time limit
    env = TimeLimit(env, max_episode_steps=10000)

    # Domain randomization (optional)
    if domain_randomize:
        env = LidarRandomizer(env, epsilon=0.05, zone_p=0.1, extreme_p=0.05)
        env = ActionRandomizer(env, epsilon=0.1)
        env = DelayedAction(env, delay_prob=0.1, drop_prob=0.05)

    # NOTE: Multi-agent환경이므로 FilterObservation, FlattenObservation 사용 안 함
    # Observation 처리는 train_mappo.py에서 수동으로 처리

    return env


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule

    Args:
        initial_value: Initial learning rate

    Returns:
        schedule: Function that returns LR based on progress
    """
    def func(progress_remaining: float):
        return progress_remaining * initial_value

    return func


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance (R^2 score)

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        explained_var: Explained variance
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def normalize_advantages(advantages: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages to have mean 0 and std 1

    Args:
        advantages: Advantage values
        epsilon: Small constant for numerical stability

    Returns:
        normalized_advantages: Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + epsilon)
