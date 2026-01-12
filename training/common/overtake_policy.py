"""
Overtake Policy Networks for Competitive Head-to-Head Racing

Training Method: Residual Policy Learning with Frozen Expert Opponent
- Agent 0: Stage 1 frozen expert (no learning)
- Agent 1: Residual learning on top of Stage 1 (only opponent_net trains)

This is NOT MAPPO - it's single-agent PPO against a frozen opponent.

Architecture:
- Centralized Critic: Uses global state (both agents' information)
- Decentralized Actors: Each agent uses only local observation
- No Parameter Sharing: Agent 0 and Agent 1 have separate actor networks
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from typing import Tuple, Dict


class MishActivation(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class ActorNetwork(nn.Module):
    """
    Decentralized Actor Network

    Input: Local observation (LiDAR + velocity, frame-stacked)
    Output: Action distribution (Gaussian)

    Each agent has its own actor network (no parameter sharing)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=[32, 32]):
        super(ActorNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Feature extractor
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(MishActivation())
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Final layer: smaller initialization for stability
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

        # log_std_head: Initialize to small negative values for conservative exploration
        # This prevents extreme action values at the start of training
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # std = exp(-1.0) ≈ 0.37

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-2, max=2)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Clip actions to valid ranges
        action = torch.clamp(action,
                            min=torch.tensor([-0.4189, 0.01], device=action.device),
                            max=torch.tensor([0.4189, 3.2], device=action.device))

        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of actions"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class ActorNetworkWithOpponentInfo(nn.Module):
    """
    Separate Network Architecture for Competitive Racing

    Key Innovation:
    - Splits observation into LiDAR features [4324] and opponent info [12]
    - LiDAR network: Uses Stage 1 weights (can be frozen or fine-tuned)
    - Opponent network: Learns from scratch (independent)
    - Combines both features before action output

    This preserves Stage 1's LiDAR processing abilities while learning
    opponent-aware behavior separately, avoiding "pollution" of trained features.

    Input: [4336] = [4324 LiDAR features] + [12 opponent info]
           - LiDAR features: (1080 scans + 1 velocity) × 4 frames = 4324
           - Opponent info: (delta_s, delta_vs, ahead_flag) × 4 frames = 12

    Output: Action distribution (Gaussian)
    """

    def __init__(self, obs_dim: int = 4336, action_dim: int = 2, hidden_dims=[32, 32]):
        super(ActorNetworkWithOpponentInfo, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lidar_obs_dim = 4324  # LiDAR + velocity, 4 frames
        self.opponent_obs_dim = 12  # Opponent info, 4 frames

        assert obs_dim == self.lidar_obs_dim + self.opponent_obs_dim, \
            f"obs_dim {obs_dim} must equal lidar_obs_dim {self.lidar_obs_dim} + opponent_obs_dim {self.opponent_obs_dim}"

        # ==================== LiDAR Network (Stage 1 Transfer) ====================
        # This network processes LiDAR + velocity features
        # Will be initialized with Stage 1 weights (can be frozen or fine-tuned)
        self.lidar_net = nn.Sequential(
            nn.Linear(self.lidar_obs_dim, 32),
            MishActivation(),
            nn.Linear(32, 32),
            MishActivation()
        )

        # ==================== Opponent Network (NEW, Trainable) ====================
        # This network processes opponent information independently
        # Learns from scratch without interfering with Stage 1 knowledge
        self.opponent_net = nn.Sequential(
            nn.Linear(self.opponent_obs_dim, 8),
            MishActivation()
        )

        # ==================== Opponent Adjustment Network (RESIDUAL) ====================
        # Instead of combining features, learn a SMALL adjustment to lidar features
        # This preserves Stage 1 lidar features while adding opponent awareness
        # opponent_features [8] → adjustment [32] (small scale)
        # final_features = lidar_features + small_scale * adjustment
        self.opponent_adjustment_net = nn.Sequential(
            nn.Linear(8, 32),
            MishActivation()
        )

        # Scale factor for opponent adjustment (FIXED at 0.01, NOT learned)
        # Prevents opponent info from dominating Stage 1 driving behavior
        # 0.01 allows max 1% influence on lidar features (REDUCED to prevent divergence)
        self.register_buffer('opponent_scale', torch.tensor(0.01, dtype=torch.float32))

        # ==================== Action Heads ====================
        self.mean_head = nn.Linear(32, action_dim)
        self.log_std_head = nn.Linear(32, action_dim)

        # Initialize weights (LiDAR network will be overwritten by Stage 1 transfer)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Final layer: smaller initialization for stability
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

        # log_std_head: Conservative exploration
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # std = exp(-1.0) ≈ 0.37

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with separate processing

        Args:
            obs: [batch_size, 4336] Full observation

        Returns:
            mean: [batch_size, action_dim] Action mean
            std: [batch_size, action_dim] Action std
        """
        # Split observation into LiDAR and opponent info
        lidar_obs = obs[:, :self.lidar_obs_dim]  # [batch, 4324]
        opponent_obs = obs[:, self.lidar_obs_dim:]  # [batch, 12]

        # Process separately
        lidar_features = self.lidar_net(lidar_obs)  # [batch, 32]
        opponent_features = self.opponent_net(opponent_obs)  # [batch, 8]

        # RESIDUAL: Add small opponent adjustment to lidar features
        # This preserves Stage 1 lidar processing while learning opponent awareness
        opponent_adjustment = self.opponent_adjustment_net(opponent_features)  # [batch, 32]

        # CRITICAL: Bound adjustment to [-1, 1] range to prevent divergence
        opponent_adjustment = torch.tanh(opponent_adjustment)  # [batch, 32] ∈ [-1, 1]

        combined = lidar_features + self.opponent_scale * opponent_adjustment  # [batch, 32]

        # Output actions (mean_head receives Stage 1-like features + small adjustment)
        mean = self.mean_head(combined)
        log_std = self.log_std_head(combined)

        # Clamp log_std for stability
        log_std = torch.clamp(log_std, min=-2, max=2)
        std = torch.exp(log_std)

        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy

        Args:
            obs: [batch_size, obs_dim]
            deterministic: If True, return mean (no noise)

        Returns:
            action: [batch_size, action_dim] (clipped to valid F110 ranges)
            log_prob: [batch_size]
        """
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Clip actions to PHYSICAL ranges
        # Action[0] (steering): [-0.4189, 0.4189] radians
        # Action[1] (speed): [0.01, 3.2] m/s
        action = torch.clamp(action,
                            min=torch.tensor([-0.4189, 0.01], device=action.device),
                            max=torch.tensor([0.4189, 3.2], device=action.device))

        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions

        Args:
            obs: [batch_size, obs_dim]
            actions: [batch_size, action_dim]

        Returns:
            log_prob: [batch_size]
            entropy: [batch_size]
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized Critic Network

    Input: Global state (all agents' observations + relative information)
    Output: State value

    Shared by all agents for centralized training
    """

    def __init__(self, global_state_dim: int, hidden_dims=[32, 32]):
        super(CriticNetwork, self).__init__()

        self.global_state_dim = global_state_dim

        # Feature extractor
        layers = []
        prev_dim = global_state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(MishActivation())
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Value head
        self.value_head = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Final layer: VERY small initialization for competitive setting
        # Prevents value explosion when transitioning from single→multi agent
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            global_state: [batch_size, global_state_dim]

        Returns:
            value: [batch_size, 1]
        """
        features = self.feature_net(global_state)
        value = self.value_head(features)
        return value


class OvertakePolicy:
    """
    Asymmetric Policy for Overtake Training (Residual Policy Learning)

    Training Method: Single-agent PPO against a frozen expert opponent
    - NOT real MAPPO (Multi-Agent PPO uses cumulative rewards)
    - Better suited for zero-sum competitive racing

    Components:
    - actor_1: Agent 0's actor network (frozen Stage 1 expert, NO opponent info)
    - actor_2: Agent 1's actor network (learner WITH opponent info, residual learning)
    - critic: Centralized critic for Agent 1 only

    Key Architecture:
    - Agent 0: ActorNetwork (obs_dim=4324, Stage 1 full transfer, frozen)
    - Agent 1: ActorNetworkWithOpponentInfo (obs_dim=4336, Stage 1 partial + opponent learning)
      - lidar_net, mean_head, log_std_head: Frozen from Stage 1
      - opponent_net, opponent_adjustment_net: Trainable (residual learning)
    """

    def __init__(
        self,
        obs_dim_agent0: int,  # 4324 (no opponent info)
        obs_dim_agent1: int,  # 4336 (with opponent info)
        action_dim: int,
        global_state_dim: int,
        hidden_dims=[32, 32],
        lr_actor=3e-4,
        lr_critic=1e-3,
        device='cuda'
    ):
        self.obs_dim_agent0 = obs_dim_agent0
        self.obs_dim_agent1 = obs_dim_agent1
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.device = device

        # Agent 0: Stage 1 expert (frozen, no opponent info)
        # Uses OLD ActorNetwork with full Stage 1 transfer
        self.actor_1 = ActorNetwork(obs_dim_agent0, action_dim, hidden_dims).to(device)

        # Agent 1: Learner with opponent awareness
        # Uses NEW ActorNetworkWithOpponentInfo
        self.actor_2 = ActorNetworkWithOpponentInfo(obs_dim_agent1, action_dim, hidden_dims).to(device)

        self.critic = CriticNetwork(global_state_dim, hidden_dims).to(device)

        # Optimizers
        self.optimizer_actor_1 = torch.optim.Adam(self.actor_1.parameters(), lr=lr_actor)
        self.optimizer_actor_2 = torch.optim.Adam(self.actor_2.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_actor(self, agent_idx: int) -> ActorNetwork:
        """Get actor network for specific agent"""
        return self.actor_1 if agent_idx == 0 else self.actor_2

    def get_optimizer_actor(self, agent_idx: int):
        """Get actor optimizer for specific agent"""
        return self.optimizer_actor_1 if agent_idx == 0 else self.optimizer_actor_2

    def select_action(self, obs: np.ndarray, agent_idx: int, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action for one agent

        Args:
            obs: Local observation for the agent
            agent_idx: 0 or 1
            deterministic: If True, return mean action

        Returns:
            action: numpy array
            log_prob: float
        """
        actor = self.get_actor(agent_idx)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = actor.get_action(obs_tensor, deterministic)

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()

        return action, log_prob

    def evaluate_value(self, global_state: np.ndarray) -> float:
        """
        Evaluate state value using centralized critic

        Args:
            global_state: Global state containing all agents' information

        Returns:
            value: float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
            return value.cpu().item()

    def save(self, path: str):
        """Save all networks"""
        checkpoint = {
            'actor_1': self.actor_1.state_dict(),
            'actor_2': self.actor_2.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor_2': self.optimizer_actor_2.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }

        # Only save optimizer_actor_1 if it exists (may be None for frozen agent)
        if self.optimizer_actor_1 is not None:
            checkpoint['optimizer_actor_1'] = self.optimizer_actor_1.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load all networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_1.load_state_dict(checkpoint['actor_1'])
        self.actor_2.load_state_dict(checkpoint['actor_2'])
        self.critic.load_state_dict(checkpoint['critic'])

        # Only load optimizer_actor_1 if it exists in checkpoint and is not None
        if 'optimizer_actor_1' in checkpoint and self.optimizer_actor_1 is not None:
            self.optimizer_actor_1.load_state_dict(checkpoint['optimizer_actor_1'])

        self.optimizer_actor_2.load_state_dict(checkpoint['optimizer_actor_2'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
