# ForzaETH Simulator Integration

This folder is for ForzaETH simulator-specific files.

## Required files to implement:

1. `render_final.py` - Rendering script adapted for ForzaETH
2. `mappo_wrappers.py` - Environment wrappers for ForzaETH
3. `mappo_utils.py` - Environment creation utilities
4. `wrappers.py` - Domain randomization wrappers
5. `maps/` - Map data (may need format conversion)
6. `centerline/` - Centerline CSV files

## Notes:

- Use `../common/mappo_policy.py` for the policy network
- Use `../common/models/mappo_final.pth` for the trained weights
- Adapt observation/action interfaces to match ForzaETH API
