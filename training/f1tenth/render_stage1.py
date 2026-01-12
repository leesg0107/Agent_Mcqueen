"""
Render Stage 1 Single Agent (Trained with PPO)

Loads the Stage 1 trained model (stable_baselines3 PPO) and renders
a single agent driving around the track.
"""

import os
import sys
import glob
import re
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'common')
sys.path.append(SCRIPT_DIR)
sys.path.append(COMMON_DIR)

from stage1_utils import create_stage1_env

# Configuration
STAGE1_MODEL_DIR = os.path.join(COMMON_DIR, "models/stage1/f1tenth_ppo_stage1_v1_20251110_133811")
STAGE1_MODEL = os.path.join(STAGE1_MODEL_DIR, "f1tenth_ppo_stage1_v1_20251110_133811_final.zip")

# Maps
MAPS = [0, 1, 2]  # Test on 3 clean maps
NUM_BEAMS = 1080
SEED = 42

NUM_EPISODES = 3
MAX_STEPS = 10000
SPEED_MULTIPLIER = 1.0  # 2x speed (1.0 = real-time, 2.0 = 2x faster)


def find_vecnormalize(model_path):
    """Find the appropriate VecNormalize file for the model."""
    model_dir = os.path.dirname(model_path)
    model_basename = os.path.basename(model_path).replace('.zip', '')

    # Try direct match first
    vecnormalize_path = model_path.replace('.zip', '_vecnormalize.pkl')

    if not os.path.exists(vecnormalize_path) and model_basename.endswith('_final'):
        # For final model, find the latest vecnormalize
        base_name = model_basename.replace('_final', '')
        pattern = os.path.join(model_dir, f"{base_name}_vecnormalize_*_steps.pkl")
        vecnorm_files = glob.glob(pattern)
        if vecnorm_files:
            def get_step_number(path):
                match = re.search(r'_(\d+)_steps\.pkl$', path)
                return int(match.group(1)) if match else 0
            vecnorm_files_sorted = sorted(vecnorm_files, key=get_step_number)
            vecnormalize_path = vecnorm_files_sorted[-1]

    elif not os.path.exists(vecnormalize_path) and '_steps' in model_basename:
        # For checkpoint model, find matching vecnormalize
        step_num = model_basename.split('_')[-2] + '_steps'
        base_name = '_'.join(model_basename.split('_')[:-2])
        vecnormalize_alt = os.path.join(model_dir, f"{base_name}_vecnormalize_{step_num}.pkl")
        if os.path.exists(vecnormalize_alt):
            vecnormalize_path = vecnormalize_alt

    return vecnormalize_path if os.path.exists(vecnormalize_path) else None


def render_stage1(model_path, num_episodes=3, test_maps=None):
    """
    Render Stage 1 trained agent.

    Args:
        model_path: Path to trained model (.zip file)
        num_episodes: Number of episodes to render
        test_maps: List of map indices to test on
    """
    if test_maps is None:
        test_maps = MAPS

    print("=" * 80)
    print("Stage 1 Single Agent Rendering")
    print("=" * 80)

    # Create policy environment (with all wrappers)
    print("\n[1] Creating environment...")
    policy_env = create_stage1_env(
        maps=test_maps,
        num_beams=NUM_BEAMS,
        seed=SEED,
        domain_randomize=False
    )
    print("  [OK] Environment created")

    # Load VecNormalize
    print("\n[2] Loading VecNormalize...")
    vecnormalize_path = find_vecnormalize(model_path)
    if vecnormalize_path:
        policy_env = VecNormalize.load(vecnormalize_path, policy_env)
        policy_env.training = False
        policy_env.norm_reward = False
        print(f"  [OK] Loaded: {os.path.basename(vecnormalize_path)}")
    else:
        print("  [WARNING] VecNormalize not found, using unnormalized environment")

    # Load model
    print(f"\n[3] Loading Stage 1 model...")
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=policy_env)
        print(f"  [OK] Loaded: {os.path.basename(model_path)}")
    else:
        print(f"  [ERROR] Model not found: {model_path}")
        sys.exit(1)

    # Get base environment for rendering
    base_env = policy_env.venv.envs[0]
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    # Run episodes
    print(f"\n[4] Running {num_episodes} episodes...")
    print("=" * 80)

    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")

        # Reset environment
        obs = policy_env.reset()

        done = False
        step = 0
        total_reward = 0
        collision_detected = False
        collision_step = None

        # Real-time sync with speed multiplier
        sim_timestep = 0.01  # F110 default: 100 Hz
        render_interval = 5  # Render every N steps (100Hz/5 = 20 FPS)
        episode_start_time = time.time()
        sim_time = 0.0

        while not done and step < MAX_STEPS:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done_vec, info = policy_env.step(action)
            done = done_vec[0]

            # Check collision from base environment
            if base_env.sim.collisions[0] and not collision_detected:
                collision_detected = True
                collision_step = step + 1

            # Render at reduced frequency
            if step % render_interval == 0:
                base_env.render(mode='human')

            # Advance simulation time (scaled by speed multiplier)
            sim_time += sim_timestep

            # Real-time sync with speed multiplier
            # SPEED_MULTIPLIER=2.0 means simulation runs 2x faster than real-time
            target_real_time = episode_start_time + (sim_time / SPEED_MULTIPLIER)
            current_real_time = time.time()
            if target_real_time > current_real_time:
                time.sleep(target_real_time - current_real_time)

            total_reward += reward[0]
            step += 1

            # Print periodic info
            if step % 100 == 0:
                print(f"  Step {step}: reward={total_reward:.2f}")

        # Episode summary
        final_collision = base_env.sim.collisions[0]

        if collision_detected:
            status = f"Collision (at step {collision_step})"
        elif done:
            if final_collision:
                status = "Episode ended with collision"
            else:
                status = "Episode completed successfully"
        elif step >= MAX_STEPS:
            status = "Max steps reached"
        else:
            status = "Unknown"

        print(f"\n  Episode {ep + 1} complete:")
        print(f"    Steps: {step}")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    Collision: {collision_detected}")
        print(f"    Status: {status}")

    policy_env.close()

    print("\n" + "=" * 80)
    print("Rendering complete!")
    print("=" * 80)
    print("\nModel configuration:")
    print("  Stage 1: Single agent PPO trained on 450 maps")
    print("  Action: steering [-1, 1], speed [0, 1] m/s")
    print("\nExpected behavior:")
    print("  Agent should drive smoothly around the track, avoiding walls.")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else STAGE1_MODEL
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_EPISODES

    render_stage1(model_path, num_episodes)
