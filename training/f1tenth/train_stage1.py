"""
Stage 1: Single-Agent PPO Training for Track Navigation

Trains a single agent to drive fast around various tracks using PPO.
This model serves as the foundation for Stage 2 overtake training.

Requirements:
- f1tenth_gym installed
- maps/ and centerline/ folders populated
"""

import os
import sys
import random
from datetime import datetime
from torch.nn import Mish
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'common')
sys.path.append(SCRIPT_DIR)
sys.path.append(COMMON_DIR)

from stage1_utils import create_stage1_env


# ==================== Configuration ====================
SAVE_INTERVAL = 50000  # Save every 50k steps
TOTAL_TIMESTEPS = 10_000_000  # 10M total steps

# Output paths
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_NAME = f"f1tenth_ppo_stage1_{TIMESTAMP}"
SAVE_PATH = os.path.join(COMMON_DIR, f"models/stage1/{LOG_NAME}")
LOG_DIR = os.path.join(SCRIPT_DIR, "runs")

# Environment
MAPS = list(range(0, 450))  # 450 random tracks
NUM_BEAMS = 1080
SEED = 42

# PPO Hyperparameters (paper-proven)
LEARNING_RATE = 0.0001
BATCH_SIZE = 2048
MINIBATCH_SIZE = 256
NUM_EPOCHS = 10
GAMMA = 0.998
CLIP_RANGE = 0.25
ENTROPY_COEF = 0.0


def linear_schedule(initial_lr: float):
    """Linear learning rate decay"""
    def schedule(progress_remaining: float):
        return initial_lr * progress_remaining
    return schedule


def train_stage1():
    """Main Stage 1 training function"""

    print("=" * 80)
    print("Stage 1: Single-Agent PPO Training")
    print("=" * 80)

    # Create directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Set seeds
    random.seed(SEED)

    # Create environment
    print("\n[1] Creating multi-map environment...")
    print(f"  Training on {len(MAPS)} random tracks")

    env = create_stage1_env(
        maps=MAPS,
        num_beams=NUM_BEAMS,
        seed=SEED,
        domain_randomize=True
    )

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Create evaluation environment
    print("\n[2] Creating evaluation environment...")
    eval_env = create_stage1_env(
        maps=MAPS,
        num_beams=NUM_BEAMS,
        seed=SEED + 1,
        domain_randomize=False
    )

    # Network architecture
    policy_kwargs = dict(
        activation_fn=Mish,
        net_arch=dict(pi=[32, 32], vf=[32, 32])
    )

    # Create PPO model
    print("\n[3] Creating PPO model...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Minibatch size: {MINIBATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Gamma: {GAMMA}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=BATCH_SIZE,
        ent_coef=ENTROPY_COEF,
        learning_rate=linear_schedule(LEARNING_RATE),
        batch_size=MINIBATCH_SIZE,
        gamma=GAMMA,
        n_epochs=NUM_EPOCHS,
        clip_range=CLIP_RANGE,
        tensorboard_log=LOG_DIR,
        device="cuda",
        policy_kwargs=policy_kwargs
    )

    # Callbacks
    print("\n[4] Setting up callbacks...")

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_INTERVAL,
        save_path=SAVE_PATH,
        name_prefix=LOG_NAME,
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_PATH, "best"),
        log_path=LOG_DIR,
        n_eval_episodes=20,
        eval_freq=SAVE_INTERVAL,
        deterministic=True,
        verbose=1
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Training
    print("\n[5] Starting training...")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  TensorBoard: tensorboard --logdir={LOG_DIR}")
    print(f"  Models saved to: {SAVE_PATH}")
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=LOG_NAME,
    )

    # Save final model
    final_path = os.path.join(SAVE_PATH, f"{LOG_NAME}_final")
    model.save(final_path)
    print(f"\n[OK] Final model saved: {final_path}")

    env.close()
    eval_env.close()

    print("\n" + "=" * 80)
    print("Stage 1 Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    train_stage1()
