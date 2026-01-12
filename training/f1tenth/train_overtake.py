"""
Stage 2: Overtake Training with Residual Policy Learning

Training Method: Single-agent PPO against frozen expert opponent
- Agent 0: Frozen Stage 1 expert (80% speed handicap)
- Agent 1: Learns overtaking via residual learning

Key Innovation:
- Agent 1's lidar_net, mean_head, log_std_head are FROZEN from Stage 1
- Only opponent_net and opponent_adjustment_net are trainable
- This preserves Stage 1 driving ability while learning opponent awareness

Requirements:
- Stage 1 trained model (run train_stage1.py first)
- f1tenth_gym installed
"""

import os
import sys
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'common')
sys.path.append(SCRIPT_DIR)
sys.path.append(COMMON_DIR)

from overtake_policy import OvertakePolicy
from overtake_buffer import OvertakeBuffer, MinibatchSampler
from racing_utils import create_racing_env


# ==================== Configuration ====================
# Stage 1 model path (UPDATE THIS)
STAGE1_MODEL_DIR = os.path.join(COMMON_DIR, "models/stage1/f1tenth_ppo_stage1_v1_20251110_133811")
STAGE1_MODEL = os.path.join(STAGE1_MODEL_DIR, "f1tenth_ppo_stage1_v1_20251110_133811_final.zip")

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(COMMON_DIR, f"models/overtake_{TIMESTAMP}")
LOG_DIR = os.path.join(SCRIPT_DIR, f"runs/overtake_{TIMESTAMP}")

# Training hyperparameters
TOTAL_TIMESTEPS = 5_000_000  # 5M steps
ROLLOUT_LENGTH = 2048
BATCH_SIZE = 256
NUM_EPOCHS = 10
GAMMA = 0.998
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.25
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
MAX_GRAD_NORM = 0.5
VALUE_LOSS_COEF = 0.5

# Environment
NUM_BEAMS = 1080
SEED = 42
MAPS = list(range(0, 450))

# Competitive settings
POSITION_WEIGHT = 0.5
OVERTAKE_BONUS = 5.0
AGENT0_SPEED_FACTOR = 0.8  # 80% speed handicap for opponent

# Dimensions
OBS_DIM_AGENT0 = 4324  # No opponent info
OBS_DIM_AGENT1 = 4336  # With opponent info
ACTION_DIM = 2
GLOBAL_STATE_DIM = 12

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_stage1_weights_full(base_model_path: str, actor_network, device='cuda'):
    """Load full Stage 1 weights for Agent 0 (frozen expert)"""
    print(f"  Loading Stage 1 model: {os.path.basename(base_model_path)}")

    stage1_model = PPO.load(base_model_path, device=device)
    stage1_state_dict = stage1_model.policy.state_dict()
    actor_state_dict = actor_network.state_dict()

    transfer_mapping = {
        'mlp_extractor.policy_net.0.weight': 'feature_net.0.weight',
        'mlp_extractor.policy_net.0.bias': 'feature_net.0.bias',
        'mlp_extractor.policy_net.2.weight': 'feature_net.2.weight',
        'mlp_extractor.policy_net.2.bias': 'feature_net.2.bias',
        'action_net.weight': 'mean_head.weight',
        'action_net.bias': 'mean_head.bias',
    }

    for stage1_key, actor_key in transfer_mapping.items():
        if stage1_key in stage1_state_dict and actor_key in actor_state_dict:
            if stage1_state_dict[stage1_key].shape == actor_state_dict[actor_key].shape:
                actor_state_dict[actor_key] = stage1_state_dict[stage1_key].clone()
                print(f"  [OK] {actor_key}")

    actor_network.load_state_dict(actor_state_dict)
    return actor_network


def load_stage1_weights_residual(base_model_path: str, actor_network, device='cuda'):
    """Load Stage 1 weights for Agent 1 with residual initialization"""
    print(f"  Loading Stage 1 model: {os.path.basename(base_model_path)}")

    stage1_model = PPO.load(base_model_path, device=device)
    stage1_state_dict = stage1_model.policy.state_dict()
    actor_state_dict = actor_network.state_dict()

    # Transfer lidar_net and mean_head from Stage 1
    transfer_mapping = {
        'mlp_extractor.policy_net.0.weight': 'lidar_net.0.weight',
        'mlp_extractor.policy_net.0.bias': 'lidar_net.0.bias',
        'mlp_extractor.policy_net.2.weight': 'lidar_net.2.weight',
        'mlp_extractor.policy_net.2.bias': 'lidar_net.2.bias',
        'action_net.weight': 'mean_head.weight',
        'action_net.bias': 'mean_head.bias',
    }

    for stage1_key, actor_key in transfer_mapping.items():
        if stage1_key in stage1_state_dict and actor_key in actor_state_dict:
            if stage1_state_dict[stage1_key].shape == actor_state_dict[actor_key].shape:
                actor_state_dict[actor_key] = stage1_state_dict[stage1_key].clone()
                print(f"  [OK] {actor_key}")

    # Initialize opponent networks with small values (residual starts near zero)
    with torch.no_grad():
        actor_state_dict['opponent_adjustment_net.0.weight'].normal_(0, 0.001)
        actor_state_dict['opponent_adjustment_net.0.bias'].zero_()
        actor_state_dict['opponent_net.0.weight'].normal_(0, 0.01)
        actor_state_dict['opponent_net.0.bias'].zero_()

    print(f"  [OK] opponent_net: small initialization")
    print(f"  [OK] opponent_adjustment_net: near-zero initialization")

    actor_network.load_state_dict(actor_state_dict)
    return actor_network


def process_observation(obs, frame_buffer_0, frame_buffer_1):
    """Process multi-agent observation with frame stacking"""
    scans_0 = obs['scans'][0] if len(obs['scans'].shape) > 1 else obs['scans']
    scans_1 = obs['scans'][1] if len(obs['scans'].shape) > 1 else obs['scans']

    vel_0 = obs['linear_vel'][0] if len(obs['linear_vel']) > 1 else obs['linear_vel']
    vel_1 = obs['linear_vel'][1] if len(obs['linear_vel']) > 1 else obs['linear_vel']

    if hasattr(vel_0, '__len__'):
        vel_0 = vel_0[0] if len(vel_0) > 0 else 0.0
    if hasattr(vel_1, '__len__'):
        vel_1 = vel_1[0] if len(vel_1) > 0 else 0.0

    # Agent 0: Only LiDAR + velocity
    frame_0 = np.concatenate([scans_0.flatten(), [vel_0]]).astype(np.float32)

    # Agent 1: LiDAR + velocity + opponent info
    s_0, s_1 = obs['poses_s'][0], obs['poses_s'][1]
    vs_0, vs_1 = obs['linear_vels_s'][0], obs['linear_vels_s'][1]

    delta_s = np.clip((s_0 - s_1) / 50.0, -1.0, 1.0)
    delta_vs = np.clip((vs_0 - vs_1) / 3.0, -1.0, 1.0)
    ahead = 1.0 if s_0 > s_1 else 0.0

    frame_1 = np.concatenate([
        scans_1.flatten(), [vel_1], [delta_s], [delta_vs], [ahead]
    ]).astype(np.float32)

    frame_buffer_0.append(frame_0)
    frame_buffer_1.append(frame_1)

    while len(frame_buffer_0) < 4:
        frame_buffer_0.append(frame_0)
    while len(frame_buffer_1) < 4:
        frame_buffer_1.append(frame_1)

    obs_0_stacked = np.concatenate(list(frame_buffer_0))
    obs_1_stacked = np.concatenate(list(frame_buffer_1))

    # Pad Agent 0 to match Agent 1 dimension
    obs_0_padded = np.concatenate([obs_0_stacked, np.zeros(12, dtype=np.float32)])

    return np.stack([obs_0_padded, obs_1_stacked])


def train_overtake():
    """Main overtake training function"""

    print("=" * 80)
    print("Stage 2: Overtake Training (Residual Policy Learning)")
    print("=" * 80)

    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Agent 0 speed: {AGENT0_SPEED_FACTOR * 100:.0f}%")

    # Create environment
    print("\n[1] Creating competitive racing environment...")
    env = create_racing_env(
        maps=MAPS,
        num_beams=NUM_BEAMS,
        seed=SEED,
        domain_randomize=False,
        position_weight=POSITION_WEIGHT,
        overtake_bonus=OVERTAKE_BONUS
    )
    print("  [OK] Environment created")

    # Create policy
    print("\n[2] Creating OvertakePolicy...")
    policy = OvertakePolicy(
        obs_dim_agent0=OBS_DIM_AGENT0,
        obs_dim_agent1=OBS_DIM_AGENT1,
        action_dim=ACTION_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        hidden_dims=[32, 32],
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        device=DEVICE
    )

    # Load Stage 1 weights
    print("\n[3] Loading Stage 1 weights...")
    if os.path.exists(STAGE1_MODEL):
        # Agent 0: Full transfer (frozen expert)
        print("  Agent 0: Full Stage 1 transfer")
        load_stage1_weights_full(STAGE1_MODEL, policy.actor_1, DEVICE)

        # Agent 1: Partial transfer (residual learning)
        print("  Agent 1: Partial transfer (residual)")
        load_stage1_weights_residual(STAGE1_MODEL, policy.actor_2, DEVICE)

        # Freeze Agent 0
        for param in policy.actor_1.parameters():
            param.requires_grad = False
        policy.optimizer_actor_1 = None

        # Freeze Agent 1's Stage 1 components
        for param in policy.actor_2.lidar_net.parameters():
            param.requires_grad = False
        for param in policy.actor_2.mean_head.parameters():
            param.requires_grad = False
        for param in policy.actor_2.log_std_head.parameters():
            param.requires_grad = False

        print(f"  [OK] Agent 0: Frozen ({AGENT0_SPEED_FACTOR*100:.0f}% speed)")
        print(f"  [OK] Agent 1: lidar_net, mean_head, log_std_head FROZEN")
        print(f"  [OK] Agent 1: opponent_net, opponent_adjustment_net TRAINABLE")
    else:
        print(f"  [ERROR] Stage 1 model not found: {STAGE1_MODEL}")
        print("  Run train_stage1.py first!")
        return

    # Create buffer
    buffer = OvertakeBuffer(
        buffer_size=ROLLOUT_LENGTH,
        obs_dim=OBS_DIM_AGENT1,
        action_dim=ACTION_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        num_agents=2
    )

    # Tensorboard
    writer = SummaryWriter(LOG_DIR)

    # Training loop
    print("\n[4] Starting training...")
    print("=" * 80)

    num_updates = TOTAL_TIMESTEPS // ROLLOUT_LENGTH
    global_step = 0

    frame_buffer_0 = deque(maxlen=4)
    frame_buffer_1 = deque(maxlen=4)

    for update in tqdm(range(num_updates), desc="Training"):
        buffer.clear()
        obs = env.reset()
        frame_buffer_0.clear()
        frame_buffer_1.clear()

        episode_rewards = {0: 0.0, 1: 0.0}

        for step in range(ROLLOUT_LENGTH):
            obs_array = process_observation(obs, frame_buffer_0, frame_buffer_1)
            global_state = env.get_global_state(obs)

            # Select actions
            action0, log_prob0 = policy.select_action(obs_array[0][:OBS_DIM_AGENT0], 0, False)
            action1, log_prob1 = policy.select_action(obs_array[1], 1, False)

            # Apply speed handicap to Agent 0
            action0[1] *= AGENT0_SPEED_FACTOR

            actions = np.stack([action0, action1])
            log_probs = np.array([log_prob0, log_prob1])
            value = policy.evaluate_value(global_state)

            obs_next, rewards, done, info = env.step(actions)

            buffer.add(obs_array, global_state, actions, rewards, log_probs, value, done)

            obs = obs_next
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]
            global_step += 1

            if done:
                obs = env.reset()
                frame_buffer_0.clear()
                frame_buffer_1.clear()
                episode_rewards = {0: 0.0, 1: 0.0}

        # Compute advantages
        global_state_final = env.get_global_state(obs)
        last_value = policy.evaluate_value(global_state_final)
        buffer.compute_returns_and_advantages(last_value, GAMMA, GAE_LAMBDA)

        # Update policy (Agent 1 only)
        sampler = MinibatchSampler(buffer, BATCH_SIZE, NUM_EPOCHS)

        actor_losses = []
        critic_losses = []

        for batch in sampler.sample_minibatches(agent_idx=1):
            obs_batch = torch.FloatTensor(batch['observations']).to(DEVICE)
            actions_batch = torch.FloatTensor(batch['actions']).to(DEVICE)
            log_probs_old = torch.FloatTensor(batch['log_probs']).to(DEVICE)
            advantages = torch.FloatTensor(batch['advantages']).to(DEVICE)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs_new, entropy = policy.actor_2.evaluate_actions(obs_batch, actions_batch)

            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            policy.optimizer_actor_2.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor_2.parameters(), MAX_GRAD_NORM)
            policy.optimizer_actor_2.step()

            actor_losses.append(actor_loss.item())

        # Update critic
        for batch in sampler.sample_critic_minibatches():
            global_states = torch.FloatTensor(batch['global_states']).to(DEVICE)
            returns = torch.FloatTensor(batch['returns']).to(DEVICE)
            returns = torch.clamp(returns, -50000, 50000)

            values_pred = policy.critic(global_states).squeeze()
            critic_loss = VALUE_LOSS_COEF * ((values_pred - returns) ** 2).mean()

            policy.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), MAX_GRAD_NORM)
            policy.optimizer_critic.step()

            critic_losses.append(critic_loss.item())

        # Logging
        writer.add_scalar('Training/Actor_Loss', np.mean(actor_losses), global_step)
        writer.add_scalar('Training/Critic_Loss', np.mean(critic_losses), global_step)

        if update % 100 == 0 and update > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"overtake_checkpoint_{update}.pth")
            policy.save(checkpoint_path)
            print(f"\n  [SAVE] {checkpoint_path}")

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "overtake_final.pth")
    policy.save(final_path)
    print(f"\n[OK] Final model saved: {final_path}")

    writer.close()
    env.close()

    print("\n" + "=" * 80)
    print("Overtake Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    train_overtake()
