# Agent McQueen - Autonomous Racing with Overtaking

> Reinforcement learning agent for autonomous racing and overtaking in F1TENTH environments.

## Demo Videos

### F1TENTH Physical Simulator

**Real-world visualization (ROS2 + RViz)**

| Stage 1: Single Agent | Stage 2: Overtaking |
|:---------------------:|:-------------------:|
| <video src="https://github.com/user-attachments/assets/ab8db412-77f6-4daa-884c-11d03753b5b9" width="100%"/> | <video src="https://github.com/user-attachments/assets/3bfaabe3-2da9-47ad-afa7-457bf9c9a8fe" width="100%"/> |
| Agent navigating with LiDAR visualization | Overtaking maneuver with both agents visible in RViz |

### ForzaETH Race Stack

**High-fidelity simulator**

| Stage 1: Single Agent | Stage 2: Overtaking |
|:---------------------:|:-------------------:|
| <video src="https://github.com/user-attachments/assets/445f2bf3-6635-490e-8b37-b70e3781df27" width="100%"/> | <video src="https://github.com/user-attachments/assets/4a10aa38-5c20-4f5a-b40c-bacd437ee932" width="100%"/> |
| Agent navigating the ForzaETH hall map | Overtaking behavior in high-fidelity physics simulation |

## Overview

Agent McQueen implements a two-stage reinforcement learning system:

- **Stage 1**: Single-agent learns fast track navigation using PPO
- **Stage 2**: Multi-agent overtaking using residual learning

Supports both F1TENTH Gym and ForzaETH Race Stack simulators.

## Repository Structure

| Directory                                       | Description                                           |
| ----------------------------------------------- | ----------------------------------------------------- |
| [docs/](docs/)                                     | Setup guide, architecture, and training documentation |
| [ros2_workspace/](ros2_workspace/)                 | ROS2 integration package for both simulators          |
| [training/](training/)                             | Training scripts, models, and wrappers                |
| [simulators/f1tenth_gym/](simulators/f1tenth_gym/) | F1TENTH Gym simulator                                 |

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA drivers installed on host

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Agent_Mcqueen.git
cd Agent_Mcqueen
```

### 2. Build Docker Image

```bash
docker build -f Dockerfile_misys_forza_full.desktop -t misys:forza_full .
```

This will automatically install:
- ROS2 Humble
- F1TENTH Gym ROS2 bridge (`f1tenth_ws`)
- ForzaETH Race Stack (`forza_ws`)
- All required dependencies

### 3. Run Docker Container

```bash
./dkrun.sh misys:forza_full
```

The container will mount:
- `simulators/f1tenth_gym` → `/home/misys/f1tenth_gym`
- `ros2_workspace` → `/home/misys/AgentMcqueen_ws`
- `training` → `/home/misys/overtake_agent`

### 4. Inside Container - Initial Setup

See [docs/SETUP.md](docs/SETUP.md) for complete instructions.

## Quick Start - F1TENTH Gym

**Note**: Do NOT source forza_ws when using F1TENTH Gym.

### Initial Setup (One-time)

```bash
source /opt/ros/humble/setup.bash

cd /home/misys/f1tenth_ws
rm -rf build/f1tenth_gym_ros install/f1tenth_gym_ros
colcon build --packages-select f1tenth_gym_ros

cd /home/misys/AgentMcqueen_ws
rm -rf build install log
colcon build --packages-select agent_mcqueen

pip install stable_baselines3
```

### Stage 1 (Single Agent)

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen sim_stage1_launch.py
```

### Overtake (Two Agents)

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen sim_overtake_launch.py
```

## Quick Start - ForzaETH Race Stack

**Note**: MUST source forza_ws when using ForzaETH.

### Initial Setup (One-time)

See [docs/SETUP.md](docs/SETUP.md) for complete ForzaETH setup including sim_single.yaml creation.

### Stage 1

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/forza_ws/race_stack/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen forza_stage1_launch.py map_name:=hall
```

### Overtake

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/forza_ws/race_stack/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen forza_overtake_launch.py map_name:=hall
```

## Available Maps

| Simulator   | Maps                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------- |
| F1TENTH Gym | `map0` (default)                                                                          |
| ForzaETH    | `hall`, `GLC_smile_small`, `small_hall`, `teras`, `glc_ot_ez`, `hangar_1905_v0` |

## Important Notes

1. **Switching simulators requires rebuild** - `f1tenth_gym_ros` conflicts between F1TENTH and ForzaETH
2. **Container restart** requires `pip install stable_baselines3` again
3. **F1TENTH Gym**: Do NOT source `forza_ws`
4. **ForzaETH**: MUST source `forza_ws`

## Troubleshooting

| Issue                                      | Cause                      | Solution                          |
| ------------------------------------------ | -------------------------- | --------------------------------- |
| `gym_bridge` error                       | Wrong f1tenth_gym_ros      | Re-run setup for that option      |
| `ModuleNotFoundError: stable_baselines3` | Missing pip package        | `pip install stable_baselines3` |
| Map not visible (F1TENTH)                  | RViz Fixed Frame           | Set to `sim`                    |
| Map not visible (ForzaETH)                 | RViz Fixed Frame           | Set to `map`                    |
| Agent not moving                           | Topic mismatch             | Check config files                |
| ForzaETH Stage1 shows 2 agents             | sim_single.yaml wrong path | Recreate in INSTALL directory     |

## Documentation

- [Setup Guide](docs/SETUP.md) - Detailed setup for both simulators
- [Architecture](docs/ARCHITECTURE.md) - ROS2 topics, observation processing
- [Training Guide](docs/TRAINING.md) - Residual learning methodology

## Features

- Pre-trained models included
- 1,362 track maps for training and testing
- Dual-simulator support (F1TENTH Gym + ForzaETH)
- GPU auto-detection (falls back to CPU)

## Acknowledgments

- [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) - Simulation environment
- ForzaETH Race Stack - High-fidelity simulator
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL library


## Contacts
