# Overtake Agent

Autonomous racing agent trained with **Residual Policy Learning** for overtaking opponents.

## Training Method

### Stage 1: Single-Agent Track Navigation (PPO)
- **Goal**: Learn to drive fast around various tracks
- **Algorithm**: Proximal Policy Optimization (PPO) with Stable-Baselines3
- **Training Data**: 450 randomly generated tracks
- **Steps**: 10M timesteps
- **Architecture**: MLP [32, 32] with Mish activation

### Stage 2: Overtake Learning (Residual Policy Learning)
- **Goal**: Learn to overtake a slower opponent
- **Method**: NOT MAPPO - Single-agent PPO against frozen expert
- **Key Innovation**: Residual learning on frozen Stage 1 weights

#### Asymmetric Architecture
| Component | Agent 0 (Opponent) | Agent 1 (Learner) |
|-----------|-------------------|-------------------|
| Role | Frozen Stage 1 Expert | Learns overtaking |
| Speed | 80% (handicapped) | 100% |
| Observation | 4324 dims (no opponent) | 4336 dims (with opponent) |
| Training | Frozen (no updates) | Residual learning |

#### Residual Learning Strategy
Agent 1's network structure:
```
lidar_net (FROZEN)         -> [32] features
opponent_net (TRAINABLE)   -> [8] features
opponent_adjustment (TRAINABLE) -> [32] adjustment

action = mean_head(lidar_features + 0.01 * opponent_adjustment)
```
- `lidar_net`, `mean_head`, `log_std_head`: Frozen from Stage 1
- `opponent_net`, `opponent_adjustment_net`: Trainable (residual)
- `opponent_scale = 0.01`: Max 1% adjustment to Stage 1 behavior

## Observation Space

### Stage 1 (Single Agent)
- **LiDAR**: 1080 beams, normalized to [0, 1]
- **Velocity**: normalized by 3.2 m/s
- **Frame stacking**: 4 frames
- **Total**: (1080 + 1) × 4 = **4324 dims**

### Stage 2 (Multi-Agent)
Agent 0 (no opponent info):
- Same as Stage 1: **4324 dims**

Agent 1 (with opponent info):
- LiDAR + velocity: 1081 dims
- Opponent info: 3 dims (delta_s, delta_vs, ahead_flag)
- Frame stacking: 4 frames
- Total: (1081 + 3) × 4 = **4336 dims**

## Action Space
- **Steering**: [-0.4189, 0.4189] radians
- **Speed**: [0.01, 3.2] m/s (Stage 1: rescaled to [0, 1])

## Reward Function

### Stage 1
```python
reward = 0.01                    # survival
reward += 1.0 * vs              # forward progress
reward -= 2.0 if stopped        # stop penalty
reward -= 0.05 * |d|            # centerline deviation
reward -= 0.01 * |vd|           # lateral velocity
reward -= 0.05 * |w|            # angular velocity
reward -= 1000.0 if collision   # collision penalty
```

### Stage 2 (Competitive)
Base rewards (both agents):
```python
reward += 10.0 * vs             # forward progress (scaled up)
reward -= 2.0 if stopped        # stop penalty
reward -= 2.0 * |w|             # angular velocity (increased)
reward -= 0.05 * |d|            # centerline deviation
reward -= 5.0 if collision      # wall collision
```

Competitive bonuses (Agent 1 only):
```python
reward += 500.0 if overtake     # one-time overtaking bonus
reward += 1000.0 if win         # finish first bonus
```

## Hyperparameters

### Stage 1 (PPO)
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 (linear decay) |
| Batch Size | 2048 |
| Minibatch Size | 256 |
| Epochs | 10 |
| Gamma | 0.998 |
| Clip Range | 0.25 |
| Entropy Coef | 0.0 |

### Stage 2 (Residual PPO)
| Parameter | Value |
|-----------|-------|
| Actor LR | 3e-4 |
| Critic LR | 1e-3 |
| Rollout Length | 2048 |
| Batch Size | 256 |
| Epochs | 10 |
| Gamma | 0.998 |
| GAE Lambda | 0.95 |
| Clip Range | 0.25 |
| Max Grad Norm | 0.5 |

## Folder Structure

```
overtake_agent/
├── README.md                    # This file
├── common/                      # Simulator-independent
│   ├── overtake_policy.py       # OvertakePolicy, ActorNetwork, etc.
│   ├── overtake_buffer.py       # Rollout buffer for Stage 2
│   └── models/
│       ├── overtake_final.pth   # Stage 2 trained model
│       └── stage1/              # Stage 1 trained model
├── f1tenth/                     # F1TENTH simulator specific
│   ├── train_stage1.py          # Stage 1 training script
│   ├── train_overtake.py        # Stage 2 training script
│   ├── render_stage1.py         # Stage 1 visualization
│   ├── render_final.py          # Stage 2 visualization
│   ├── stage1_utils.py          # Stage 1 environment creation
│   ├── racing_utils.py          # Stage 2 environment creation
│   ├── racing_wrappers.py       # Multi-agent wrappers
│   ├── wrappers.py              # Single-agent wrappers
│   ├── maps/                    # Track map files
│   └── centerline/              # Track centerline data
└── forzaeth/                    # ForzaETH simulator (for sim2sim)
    └── README.md
```

## Usage

### Rendering
```bash
cd overtake_agent/f1tenth

# Stage 1: Single agent driving
python render_stage1.py

# Stage 2: Overtake racing
python render_final.py
```

### Training
```bash
cd overtake_agent/f1tenth

# Stage 1: Train single-agent driving (requires f1tenth_gym)
python train_stage1.py

# Stage 2: Train overtaking (requires Stage 1 model)
python train_overtake.py
```

## Sim2Sim Transfer

This code is designed for transfer between simulators:

1. **Simulator-independent** (`common/`):
   - Policy networks (PyTorch)
   - Trained weights
   - Buffer implementation

2. **Simulator-specific** (`f1tenth/`, `forzaeth/`):
   - Environment wrappers
   - Observation processing
   - Rendering scripts

To transfer to a new simulator:
1. Create new folder (e.g., `newsim/`)
2. Implement wrappers matching observation/action spaces
3. Use `common/overtake_policy.py` directly
4. Load `common/models/overtake_final.pth`

## Dependencies
- PyTorch
- NumPy
- stable-baselines3 (Stage 1)
- gym
- f1tenth_gym (for F1TENTH simulator)
- scikit-learn (KDTree for Frenet conversion)
