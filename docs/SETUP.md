# Agent McQueen - Setup Guide

This guide explains how to run Agent McQueen in both F1TENTH Gym and ForzaETH simulators.

## Prerequisites

The Docker image (`Dockerfile_misys_forza_full.desktop`) automatically installs:
- ROS2 Humble
- F1TENTH Gym ROS2 bridge (`f1tenth_ws`)
- ForzaETH Race Stack (`forza_ws`)
- All required dependencies

## Docker Setup

1. **Build the Docker image** (first time only):
```bash
cd Agent_Mcqueen
docker build -f Dockerfile_misys_forza_full.desktop -t misys:forza_full .
```

2. **Run the container**:
```bash
./dkrun.sh misys:forza_full
```

The `dkrun.sh` script automatically mounts:
- `ros2_workspace` → `/home/misys/AgentMcqueen_ws`
- `training` → `/home/misys/overtake_agent`
- `simulators/f1tenth_gym` → `/home/misys/f1tenth_gym`

---

# Option A: F1TENTH Gym

To use F1TENTH Gym, **DO NOT source forza_ws**.

## A-1. F1TENTH Setup (Inside Container, One-time)

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

## A-2. Running F1TENTH

**Run in a new terminal (DO NOT source forza_ws!)**

### Stage 1:

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen sim_stage1_launch.py
```

### Overtake:

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen sim_overtake_launch.py

# If only one agent appears in RViz:
# Add -> RobotModel -> Select /sim/opp_robot_description in description topic -> SAVE!!!
```

---

# Option B: ForzaETH Race Stack

To use ForzaETH, **MUST source forza_ws**.

## B-1. ForzaETH Setup (Inside Container, One-time)

```bash
# 1. Create sim_single.yaml (for Stage 1 - num_agent: 1)
# IMPORTANT: Must copy to INSTALL directory!
cp /home/misys/forza_ws/race_stack/stack_master/config/SIM/sim.yaml \
   /home/misys/forza_ws/race_stack/install/stack_master/share/stack_master/config/SIM/sim_single.yaml

# Change num_agent to 1
sed -i 's/num_agent: 2/num_agent: 1/' \
   /home/misys/forza_ws/race_stack/install/stack_master/share/stack_master/config/SIM/sim_single.yaml

# Verify
grep num_agent /home/misys/forza_ws/race_stack/install/stack_master/share/stack_master/config/SIM/sim_single.yaml

# 2. Register ForzaETH f1tenth_gym with pip
cd /home/misys/forza_ws/race_stack/base_system/f110_simulator/f1tenth_gym
pip install -e .

# 3. Build Agent McQueen (ForzaETH environment)
source /opt/ros/humble/setup.bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/forza_ws/race_stack/install/setup.bash
cd /home/misys/AgentMcqueen_ws
rm -rf build install log
colcon build --packages-select agent_mcqueen

# 4. Install Python packages
pip install stable_baselines3
```

## B-2. Running ForzaETH

**Note**: Even if the simulator starts and the agent isn't visible, wait 10-15 seconds for rendering. Currently only the `hall` map is supported, as start positions are configured for this map.

### Stage 1:

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/forza_ws/race_stack/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen forza_stage1_launch.py map_name:=hall
```

### Overtake:

```bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/forza_ws/race_stack/install/setup.bash
source /home/misys/AgentMcqueen_ws/install/setup.bash
ros2 launch agent_mcqueen forza_overtake_launch.py map_name:=hall
```

---

## Available Maps

| Simulator | Maps |
| --------- | ---- |
| F1TENTH   | `map0` (default) |
| ForzaETH  | `hall`, `GLC_smile_small`, `small_hall`, `teras`, `glc_ot_ez`, `hangar_1905_v0` |

---

## Important Notes

1. **Rebuild required when switching between F1TENTH and ForzaETH** - `f1tenth_gym_ros` conflicts between them
2. **After container restart** - Need to run `pip install stable_baselines3` again
3. **When running ForzaETH** - MUST source `forza_ws`
4. **When running F1TENTH** - DO NOT source `forza_ws`

---

## Troubleshooting

| Issue | Cause | Solution |
| ----- | ----- | -------- |
| `gym_bridge` error | Using wrong f1tenth_gym_ros | Re-run setup for that option |
| `ModuleNotFoundError: stable_baselines3` | Missing pip package | `pip install stable_baselines3` |
| Map not visible (F1TENTH) | RViz Fixed Frame wrong | Change to `sim` |
| Map not visible (ForzaETH) | RViz Fixed Frame wrong | Change to `map` |
| Agent not moving | Topic mismatch | Check config files (already fixed) |
| ForzaETH Stage1 renders 2 agents | sim_single.yaml path error | Re-run B-1 setup for sim_single.yaml (in INSTALL directory!) |
| `ros2 topic list` empty | sim_single.yaml missing | Re-run B-1 setup |

---

## See Also

- [Architecture](ARCHITECTURE.md) - ROS2 topics and observation processing
- [Training Guide](TRAINING.md) - Training methodology and hyperparameters
- [Repository Root](../README.md) - Project overview
