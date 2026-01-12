"""
Render Residual Competitive Policy (Overtake Agent)

Training method: Residual Policy Learning with Frozen Expert Opponent
- Agent 0: Stage 1 frozen expert (no learning)
- Agent 1: Residual learning on top of Stage 1 (only opponent_net trains)

This is NOT MAPPO - it's single-agent PPO against a frozen opponent.
"""

import os
import sys
import numpy as np
import torch
import time
from collections import deque

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'common')
sys.path.append(SCRIPT_DIR)
sys.path.append(COMMON_DIR)

from overtake_policy import OvertakePolicy  # from common/
from racing_utils import create_racing_env  # from f1tenth/

# Configuration
OVERTAKE_MODEL = os.path.join(COMMON_DIR, "models/overtake_final.pth")

# Maps
MAPS = [0, 1, 2]  # Test on 3 clean maps
NUM_BEAMS = 1080
SEED = 42

# Asymmetric dimensions
OBS_DIM_AGENT0 = 4324  # No opponent info
OBS_DIM_AGENT1 = 4336  # With opponent info
ACTION_DIM = 2
GLOBAL_STATE_DIM = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EPISODES = 3
MAX_STEPS = 10000
AGENT0_SPEED_FACTOR = 0.8  # Agent 0 runs at 80% speed (handicap)
SPEED_MULTIPLIER = 1.5  # 2x speed (1.0 = real-time, 2.0 = 2x faster)


def process_observation(obs, frame_buffer_0, frame_buffer_1):
    """
    Process multi-agent observation with ASYMMETRIC observations

    Agent 0: No opponent info (4324 dims)
    Agent 1: With opponent info (4336 dims)
    """
    scans_0 = obs['scans'][0]
    scans_1 = obs['scans'][1]

    vel_0 = obs['linear_vel'][0]
    vel_1 = obs['linear_vel'][1]

    # Extract opponent information (ONLY for Agent 1)
    s_0 = obs['poses_s'][0]
    s_1 = obs['poses_s'][1]
    vs_0 = obs['linear_vels_s'][0]
    vs_1 = obs['linear_vels_s'][1]

    # Agent 1's opponent info (relative to Agent 0)
    delta_s_1 = s_0 - s_1  # Positive if Agent 0 ahead
    delta_vs_1 = vs_0 - vs_1  # Positive if Agent 0 faster
    ahead_1 = 1.0 if delta_s_1 > 0 else 0.0

    # Normalize
    delta_s_1_norm = np.clip(delta_s_1 / 50.0, -1.0, 1.0)
    delta_vs_1_norm = np.clip(delta_vs_1 / 3.0, -1.0, 1.0)

    # Agent 0: NO opponent info (1081 per frame -> 4324 stacked)
    frame_0 = np.concatenate([
        scans_0.flatten(),  # [1080]
        [vel_0]             # [1]
    ]).astype(np.float32)  # [1081]

    # Agent 1: WITH opponent info (1084 per frame -> 4336 stacked)
    frame_1 = np.concatenate([
        scans_1.flatten(),      # [1080]
        [vel_1],                # [1]
        [delta_s_1_norm],       # [1]
        [delta_vs_1_norm],      # [1]
        [ahead_1]               # [1]
    ]).astype(np.float32)  # [1084]

    frame_buffer_0.append(frame_0)
    frame_buffer_1.append(frame_1)

    while len(frame_buffer_0) < 4:
        frame_buffer_0.append(frame_0)
    while len(frame_buffer_1) < 4:
        frame_buffer_1.append(frame_1)

    obs_0_stacked = np.concatenate(list(frame_buffer_0))  # [4324]
    obs_1_stacked = np.concatenate(list(frame_buffer_1))  # [4336]

    # Pad Agent 0 to 4336 for uniform dimension (matching training setup)
    padding = np.zeros(12, dtype=np.float32)
    obs_0_padded = np.concatenate([obs_0_stacked, padding])  # [4336]

    return np.stack([obs_0_padded, obs_1_stacked])  # [2, 4336]


print("="*80)
print("PPO-to-Compete Final Model Rendering")
print("="*80)

# Create environment
print("\n[1] Creating environment...")
env = create_racing_env(
    maps=MAPS,
    num_beams=NUM_BEAMS,
    seed=SEED,
    domain_randomize=False,
    position_weight=0.5,
    overtake_bonus=5.0
)
print("  [OK] Environment created")

# Create Overtake policy (Residual Competitive)
print("\n[2] Creating Overtake policy...")
policy = OvertakePolicy(
    obs_dim_agent0=OBS_DIM_AGENT0,
    obs_dim_agent1=OBS_DIM_AGENT1,
    action_dim=ACTION_DIM,
    global_state_dim=GLOBAL_STATE_DIM,
    hidden_dims=[32, 32],
    device=DEVICE
)
print("  [OK] Overtake policy created")

# Load final trained model
print(f"\n[3] Loading final trained model...")
if os.path.exists(OVERTAKE_MODEL):
    policy.load(OVERTAKE_MODEL)
    print(f"  [OK] Loaded: {OVERTAKE_MODEL}")
    print(f"  Agent 0: Stage 1 frozen expert (actor_1)")
    print(f"  Agent 1: Overtake-trained agent (actor_2)")
else:
    print(f"  [ERROR] Model not found: {OVERTAKE_MODEL}")
    print("  Cannot render without trained model")
    sys.exit(1)

# Run episodes
print(f"\n[4] Running {NUM_EPISODES} episodes...")
print("="*80)

for ep in range(NUM_EPISODES):
    # Reset with Agent 0 starting AHEAD (same as training)
    obs = env.reset(side_by_side=False)

    frame_buffer_0 = deque(maxlen=4)
    frame_buffer_1 = deque(maxlen=4)

    episode_rewards = {0: 0.0, 1: 0.0}
    episode_lengths = {0: 0, 1: 0}

    done = False
    step = 0

    print(f"\nEpisode {ep + 1}/{NUM_EPISODES}")
    print(f"  Initial positions: Agent0 s={obs['poses_s'][0]:.2f}, Agent1 s={obs['poses_s'][1]:.2f}")
    if obs['poses_s'][0] > obs['poses_s'][1]:
        print(f"  -> Agent 0 starts AHEAD")
    else:
        print(f"  -> Agent 1 starts AHEAD")

    # Real-time sync with speed multiplier
    sim_timestep = 0.01  # F110 default: 100 Hz
    render_interval = 5  # Render every N steps (100Hz/5 = 20 FPS)
    episode_start_time = time.time()
    sim_time = 0.0

    while not done and step < MAX_STEPS:
        # Process observation
        obs_array = process_observation(obs, frame_buffer_0, frame_buffer_1)

        # Select actions (deterministic for evaluation)
        # Agent 0: Use only first 4324 dims (ActorNetwork expects this)
        # Agent 1: Use all 4336 dims (ActorNetworkWithOpponentInfo expects this)
        action0, _ = policy.select_action(obs_array[0][:OBS_DIM_AGENT0], agent_idx=0, deterministic=True)
        action1, _ = policy.select_action(obs_array[1], agent_idx=1, deterministic=True)

        # Agent 0 speed scaling (1.0 = full speed, same as Agent 1)
        action0[1] = action0[1] * AGENT0_SPEED_FACTOR

        actions = np.stack([action0, action1])

        # Step
        obs, rewards, done, info = env.step(actions)

        # Render at reduced frequency
        if step % render_interval == 0:
            env.render()

        # Advance simulation time (scaled by speed multiplier)
        sim_time += sim_timestep

        # Real-time sync with speed multiplier
        # SPEED_MULTIPLIER=2.0 means simulation runs 2x faster than real-time
        target_real_time = episode_start_time + (sim_time / SPEED_MULTIPLIER)
        current_real_time = time.time()
        if target_real_time > current_real_time:
            time.sleep(target_real_time - current_real_time)

        episode_rewards[0] += rewards[0]
        episode_rewards[1] += rewards[1]
        episode_lengths[0] += 1
        episode_lengths[1] += 1
        step += 1

        # Print periodic info
        if step % 100 == 0:
            vel_0 = obs['linear_vels_x'][0]
            vel_1 = obs['linear_vels_x'][1]
            vs_0 = obs.get('linear_vels_s', [0, 0])[0]
            vs_1 = obs.get('linear_vels_s', [0, 0])[1]
            print(f"  Step {step}:")
            print(f"    Velocity: Agent0 vx={vel_0:.2f}m/s vs={vs_0:.2f}m/s, Agent1 vx={vel_1:.2f}m/s vs={vs_1:.2f}m/s")
            print(f"    Rewards: r0={episode_rewards[0]:.1f}, r1={episode_rewards[1]:.1f}")
            print(f"    Actions: Agent0 steer={action0[0]:.2f} speed={action0[1]:.2f}, Agent1 steer={action1[0]:.2f} speed={action1[1]:.2f}")

    # Episode summary
    print(f"\n  Episode {ep + 1} complete:")
    print(f"    Agent 0: reward={episode_rewards[0]:.2f}, length={episode_lengths[0]}")
    print(f"    Agent 1: reward={episode_rewards[1]:.2f}, length={episode_lengths[1]}")

    lap_counts = info.get('lap_count', [0, 0])
    collisions = obs.get('collisions', [0, 0])
    print(f"    Lap counts: Agent0={lap_counts[0]}, Agent1={lap_counts[1]}")
    print(f"    Collisions: Agent0={collisions[0]}, Agent1={collisions[1]}")

    # Check behavior
    if episode_lengths[0] > 50:
        avg_reward_0 = episode_rewards[0] / episode_lengths[0]
        avg_reward_1 = episode_rewards[1] / episode_lengths[1]
        print(f"    Average reward per step: Agent0={avg_reward_0:.2f}, Agent1={avg_reward_1:.2f}")

        if avg_reward_0 < -2:
            print(f"    [!] Agent 0: Very negative rewards - likely stopping or moving backward")
        if avg_reward_1 < -2:
            print(f"    [!] Agent 1: Very negative rewards - likely stopping or moving backward")

env.close()

print("\n" + "="*80)
print("Rendering complete!")
print("="*80)
print("\nModel configuration:")
print("  Agent 0: Stage 1 frozen expert (80% speed, starts AHEAD)")
print("  Agent 1: Overtake-trained agent (100% speed, starts BEHIND)")
print("\nExpected behavior:")
print("  Agent 1 should catch up and attempt overtaking maneuvers.")
print("  Agent 0 drives consistently but slower (handicap).")
