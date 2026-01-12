# ROS2 Workspace - Agent McQueen

ROS2 Humble package for running Agent McQueen in F1TENTH Gym and ForzaETH simulators.

## Structure

- `src/agent_mcqueen/agent_mcqueen/` - Python nodes (stage1_agent_node.py, overtake_agent_node.py)
- `src/agent_mcqueen/config/` - 6 YAML configuration files
- `src/agent_mcqueen/launch/` - 6 ROS2 launch files
- `src/agent_mcqueen/models/` - Pre-trained models
- `maps/` - 1,362 track map YAML files

## Quick Build

```bash
source /opt/ros/humble/setup.bash
cd ros2_workspace
colcon build --packages-select agent_mcqueen
source install/setup.bash
```

## Usage

See [../docs/SETUP.md](../docs/SETUP.md) for complete instructions.

## Documentation

- [Setup Guide](../docs/SETUP.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [Training](../docs/TRAINING.md)
