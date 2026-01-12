#!/usr/bin/env python3
"""
Extract VecNormalize statistics from Stage 1 model.

This script loads VecNormalize the same way render_stage1.py does,
then extracts the obs_rms statistics to a numpy file that can be
loaded without needing the full environment.

Run from the overtake_agent/f1tenth directory:
    python extract_stage1_stats.py

Output: Creates _stats.npz files alongside the .pkl files
"""

import os
import sys
import glob
import re
import numpy as np

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'common'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from f1tenth_env import F1TenthEnv


def find_all_vecnormalize_files(models_dir):
    """Find all VecNormalize .pkl files in the models directory."""
    pattern = os.path.join(models_dir, '**', '*vecnormalize*.pkl')
    return glob.glob(pattern, recursive=True)


def extract_stats(vecnorm_path, env):
    """Extract statistics from a VecNormalize file."""
    print(f"\nProcessing: {vecnorm_path}")

    try:
        # Wrap env in DummyVecEnv for VecNormalize
        if not isinstance(env, (DummyVecEnv, SubprocVecEnv)):
            vec_env = DummyVecEnv([lambda: env])
        else:
            vec_env = env

        # Load VecNormalize with the real environment
        vecnorm = VecNormalize.load(vecnorm_path, vec_env)
        vecnorm.training = False
        vecnorm.norm_reward = False

        # Extract statistics
        if hasattr(vecnorm, 'obs_rms') and vecnorm.obs_rms is not None:
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

            print(f"  Saved: {output_path}")
            return output_path
        else:
            print("  WARNING: No obs_rms found")
            return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    # Stage 1 models directory
    models_dir = os.path.join(SCRIPT_DIR, '..', 'common', 'models', 'stage1')

    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}")
        sys.exit(1)

    # Create a single environment for loading VecNormalize
    print("Creating F1Tenth environment...")
    map_path = os.path.join(SCRIPT_DIR, 'maps', 'map0', 'map0')
    env = F1TenthEnv(
        map_path=map_path,
        random_start=False,
        num_beams=1080,
        frame_stack=4,
        render_mode=None
    )
    print("  [OK] Environment created")

    # Find all VecNormalize files
    vecnorm_files = find_all_vecnormalize_files(models_dir)

    if not vecnorm_files:
        print(f"No VecNormalize files found in {models_dir}")
        sys.exit(1)

    print(f"\nFound {len(vecnorm_files)} VecNormalize file(s)")

    # Extract stats from each file
    extracted = 0
    for vecnorm_path in vecnorm_files:
        result = extract_stats(vecnorm_path, env)
        if result:
            extracted += 1

    env.close()
    print(f"\nDone! Extracted {extracted}/{len(vecnorm_files)} files")


if __name__ == '__main__':
    main()
