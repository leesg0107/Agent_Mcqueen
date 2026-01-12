# Agent McQueen - Architecture

## Overview

ROS2 package for integrating reinforcement learning-trained autonomous racing agents with F1TENTH/ForzaETH simulators.

### Agent Types

| Agent | Description | Model |
|-------|-------------|-------|
| **Stage 1** | Single-agent track navigation | stable_baselines3 PPO |
| **Overtake** | Two-agent overtaking simulation | Custom OvertakePolicy |

### Overtake Configuration

- **Agent 0 (ego)**: Stage 1 frozen expert, 80% speed, starts in front
- **Agent 1 (opp)**: Overtake-trained, 100% speed, starts behind and attempts to overtake

---

## Directory Structure

```
ros2_workspace/
└── src/agent_mcqueen/
    ├── agent_mcqueen/
    │   ├── stage1_agent_node.py        # Stage 1 ROS2 node
    │   └── overtake_agent_node.py      # Overtake ROS2 node
    ├── config/
    │   ├── sim_stage1.yaml             # F1TENTH Stage 1 simulator config
    │   ├── stage1_config.yaml          # F1TENTH Stage 1 agent parameters
    │   ├── sim_overtake.yaml           # F1TENTH Overtake simulator config
    │   ├── overtake_config.yaml        # F1TENTH Overtake agent parameters
    │   ├── forza_stage1_config.yaml    # ForzaETH Stage 1 agent parameters
    │   └── forza_overtake_config.yaml  # ForzaETH Overtake agent parameters
    └── launch/
        ├── sim_stage1_launch.py        # F1TENTH + Stage 1 launch
        ├── sim_overtake_launch.py      # F1TENTH + Overtake launch
        ├── forza_stage1_launch.py      # ForzaETH + Stage 1 launch
        └── forza_overtake_launch.py    # ForzaETH + Overtake launch
```

---

## Prerequisites

### Docker Environment

Run inside ForzaETH Docker container:

```bash
# On host
cd /path/to/forzaeth
./run_docker.sh
```

### Required Workspaces

| Path | Description |
|------|-------------|
| `/home/misys/f1tenth_ws` | F1TENTH Gym ROS2 bridge |
| `/home/misys/forza_ws/race_stack` | ForzaETH Race Stack |
| `/home/misys/AgentMcqueen_ws` | This package |
| `/home/misys/overtake_agent` | Trained models and common libraries |

### Required Model Files

| Model | Path |
|-------|------|
| Stage 1 | `/home/misys/overtake_agent/common/models/stage1/.../final.zip` |
| Overtake | `/home/misys/overtake_agent/common/models/overtake_final.pth` |

---

## Technical Details

### Observation Processing

#### LiDAR Scan
```python
scan = np.clip(scan, 0.0, 10.0) / 10.0  # Normalize to [0, 1]
scan = scan[::-1]                        # Reverse direction (F110 convention)
```

#### Velocity
```python
velocity = vx / 3.2  # Normalize by max speed
```

#### Frame Stacking
- 4 frames stacked → shape: (4, 1081)
- Each frame: 1080 LiDAR beams + 1 velocity

### Observation Order (Stage 1)

`FlattenObservation` sorts Dict keys alphabetically:
```python
# Correct order: [linear_vel, scans] (alphabetical)
obs = np.concatenate([velocity, scan])  # NOT [scan, velocity]
```

### Overtake Asymmetric Observations

| Agent | Dimension | Composition |
|-------|-----------|-------------|
| Agent 0 | 4324 | scan(1080) + vel(1) × 4 frames |
| Agent 1 | 4336 | scan(1080) + vel(1) + delta_s(1) + delta_vs(1) + ahead(1) × 4 frames |

---

## ROS2 Topics

### F1TENTH Stage 1

| Topic | Type | Direction |
|-------|------|-----------|
| `/sim/scan` | LaserScan | Subscribe |
| `/sim/ego_racecar/odom` | Odometry | Subscribe |
| `/sim/drive` | AckermannDriveStamped | Publish |

### F1TENTH Overtake

| Topic | Type | Direction | Agent |
|-------|------|-----------|-------|
| `/sim/scan` | LaserScan | Subscribe | Ego |
| `/sim/opp_scan` | LaserScan | Subscribe | Opp |
| `/sim/ego_racecar/odom` | Odometry | Subscribe | Ego |
| `/sim/opp_racecar/odom` | Odometry | Subscribe | Opp |
| `/sim/drive` | AckermannDriveStamped | Publish | Ego |
| `/sim/opp_drive` | AckermannDriveStamped | Publish | Opp |

### ForzaETH Stage 1

| Topic | Type | Direction |
|-------|------|-----------|
| `/scan` | LaserScan | Subscribe |
| `/car_state/odom_GT` | Odometry | Subscribe |
| `/drive` | AckermannDriveStamped | Publish |

### ForzaETH Overtake

| Topic | Type | Direction | Agent |
|-------|------|-----------|-------|
| `/scan` | LaserScan | Subscribe | Ego |
| `/opp_scan` | LaserScan | Subscribe | Opp |
| `/car_state/odom_GT` | Odometry | Subscribe | Ego |
| `/opp_racecar/odom` | Odometry | Subscribe | Opp |
| `/drive` | AckermannDriveStamped | Publish | Ego |
| `/opp_drive` | AckermannDriveStamped | Publish | Opp |

---

## Configuration Reference

### sim_overtake.yaml

```yaml
# Agent start positions
sx: 0.2663860       # Agent 0 (front)
sx1: 10.2663860     # Agent 1 (behind, ~10m gap)
stheta: 3.208718    # ≈ π, -x direction → lower x is in front

# Map settings
map_path: '/home/misys/overtake_agent/f1tenth/maps/map0'
num_agent: 2
```

### overtake_config.yaml

```yaml
# Speed handicap
agent0_speed_factor: 0.8  # Agent 0 runs at 80% speed

# Model path
model_path: '/home/misys/overtake_agent/common/models/overtake_final.pth'

# Device
device: 'cuda'  # Auto-fallback to 'cpu' if GPU unavailable
```

---

## Integration Status

| Simulator | Stage 1 | Overtake | Status |
|-----------|---------|----------|--------|
| F1TENTH Gym | ✓ | ✓ | Complete |
| ForzaETH | ✓ | ✓ | Complete |

### ForzaETH Available Maps

| Map Name | Description |
|----------|-------------|
| `hall` | Wide corridor environment |
| `GLC_smile_small` | Small track |
| `small_hall` | Small hall |
| `teras` | Terrace/corridor environment |
| `glc_ot_ez` | Overtaking training track |
| `hangar_1905_v0` | Hangar environment |
