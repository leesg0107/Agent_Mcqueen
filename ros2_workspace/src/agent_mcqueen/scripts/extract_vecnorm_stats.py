#!/usr/bin/env python3
"""
Extract VecNormalize statistics to numpy format.

Run this script in your training environment where VecNormalize can be loaded properly.
It will create a .npz file that can be loaded without circular reference issues.

Usage:
    python extract_vecnorm_stats.py <vecnorm_pkl_path>

Example:
    python extract_vecnorm_stats.py f1tenth_ppo_stage1_v1_vecnormalize.pkl
    # Creates: f1tenth_ppo_stage1_v1_vecnormalize_stats.npz
"""

import sys
import os
import pickle
import numpy as np


def extract_stats(vecnorm_path):
    """Extract obs_rms stats from VecNormalize pickle file"""

    print(f"Loading: {vecnorm_path}")

    # Increase recursion limit
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100000)

    try:
        # Try standard pickle first
        with open(vecnorm_path, 'rb') as f:
            vecnorm = pickle.load(f)
    except RecursionError:
        print("Standard pickle failed, trying cloudpickle...")
        try:
            import cloudpickle
            with open(vecnorm_path, 'rb') as f:
                vecnorm = cloudpickle.load(f)
        except Exception as e:
            print(f"Cloudpickle also failed: {e}")
            print("\nTrying VecNormalize.load method...")

            # Last resort: use VecNormalize.load with dummy env
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            import gymnasium as gym

            # Try to create a dummy env with matching obs space
            def make_dummy():
                env = gym.make('CartPole-v1')
                return env

            dummy_venv = DummyVecEnv([make_dummy])
            vecnorm = VecNormalize.load(vecnorm_path, dummy_venv)
    finally:
        sys.setrecursionlimit(old_limit)

    # Extract stats
    if hasattr(vecnorm, 'obs_rms'):
        obs_mean = vecnorm.obs_rms.mean
        obs_var = vecnorm.obs_rms.var
        clip_obs = getattr(vecnorm, 'clip_obs', 10.0)
        epsilon = getattr(vecnorm, 'epsilon', 1e-8)

        print(f"  obs_mean shape: {obs_mean.shape}")
        print(f"  obs_var shape: {obs_var.shape}")
        print(f"  clip_obs: {clip_obs}")
        print(f"  epsilon: {epsilon}")

        # Save to npz
        output_path = vecnorm_path.replace('.pkl', '_stats.npz')
        np.savez(output_path,
                 mean=obs_mean,
                 var=obs_var,
                 clip_obs=np.array([clip_obs]),
                 epsilon=np.array([epsilon]))

        print(f"\nSaved to: {output_path}")
        return output_path
    else:
        print("ERROR: No obs_rms found in VecNormalize object")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    vecnorm_path = sys.argv[1]
    if not os.path.exists(vecnorm_path):
        print(f"ERROR: File not found: {vecnorm_path}")
        sys.exit(1)

    extract_stats(vecnorm_path)
