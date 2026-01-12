"""
Overtake Rollout Buffer for Residual Policy Learning

Stores experiences from both agents and global states for centralized critic training.
"""

import numpy as np
from typing import Dict


class OvertakeBuffer:
    """
    Rollout buffer for Overtake training with 2 agents

    Stores:
    - Local observations (for each agent)
    - Global states (for centralized critic)
    - Actions (for each agent)
    - Rewards (for each agent)
    - Log probabilities (for each agent)
    - Values (from centralized critic)
    - Done flags
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, global_state_dim: int, num_agents: int = 2):
        """
        Args:
            buffer_size: Maximum number of timesteps to store
            obs_dim: Dimension of local observation
            action_dim: Dimension of action
            global_state_dim: Dimension of global state
            num_agents: Number of agents (default: 2)
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.num_agents = num_agents

        # Current position in buffer
        self.ptr = 0
        self.full = False

        # Storage
        self.observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.global_states = np.zeros((buffer_size, global_state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        # For GAE calculation
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,  # [num_agents, obs_dim]
        global_state: np.ndarray,  # [global_state_dim]
        actions: np.ndarray,  # [num_agents, action_dim]
        rewards: np.ndarray,  # [num_agents]
        log_probs: np.ndarray,  # [num_agents]
        value: float,
        done: bool
    ):
        """Add one timestep of experience"""
        self.observations[self.ptr] = obs
        self.global_states[self.ptr] = global_state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.998,
        gae_lambda: float = 0.95
    ):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)

        ASYMMETRIC TRAINING: GAE for Agent 1 ONLY
        - Critic predicts Agent 1's return (trained on Agent 1's returns only)
        - Agent 0 is frozen (doesn't need advantages)
        - Agent 1 is learning (needs accurate advantages)

        Args:
            last_value: Value of the last state (for bootstrapping)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        buffer_len = self.buffer_size if self.full else self.ptr

        # Initialize
        last_gae_lam = np.zeros(self.num_agents, dtype=np.float32)

        # Compute advantages backward
        for step in reversed(range(buffer_len)):
            if step == buffer_len - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            # ONLY compute for Agent 1 (index 1)
            agent_idx = 1
            # TD error for Agent 1
            delta = self.rewards[step, agent_idx] + gamma * next_value * next_non_terminal - self.values[step]

            # GAE for Agent 1
            last_gae_lam[agent_idx] = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam[agent_idx]
            self.advantages[step, agent_idx] = last_gae_lam[agent_idx]

            # Agent 0: Set advantage to 0 (frozen, not trained)
            self.advantages[step, 0] = 0.0

        # Compute returns
        for agent_idx in range(self.num_agents):
            self.returns[:buffer_len, agent_idx] = self.advantages[:buffer_len, agent_idx] + self.values[:buffer_len]

    def get_data(self, agent_idx: int) -> Dict[str, np.ndarray]:
        """Get all data for one agent"""
        buffer_len = self.buffer_size if self.full else self.ptr

        return {
            'observations': self.observations[:buffer_len, agent_idx],
            'global_states': self.global_states[:buffer_len],
            'actions': self.actions[:buffer_len, agent_idx],
            'rewards': self.rewards[:buffer_len, agent_idx],
            'log_probs': self.log_probs[:buffer_len, agent_idx],
            'advantages': self.advantages[:buffer_len, agent_idx],
            'returns': self.returns[:buffer_len, agent_idx],
        }

    def get_all_data(self) -> Dict[str, np.ndarray]:
        """Get all data from buffer"""
        buffer_len = self.buffer_size if self.full else self.ptr

        return {
            'observations': self.observations[:buffer_len],
            'global_states': self.global_states[:buffer_len],
            'actions': self.actions[:buffer_len],
            'rewards': self.rewards[:buffer_len],
            'log_probs': self.log_probs[:buffer_len],
            'values': self.values[:buffer_len],
            'advantages': self.advantages[:buffer_len],
            'returns': self.returns[:buffer_len],
        }

    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.full = False

    def __len__(self):
        """Return current buffer size"""
        return self.buffer_size if self.full else self.ptr


class MinibatchSampler:
    """Minibatch sampler for training"""

    def __init__(self, buffer: OvertakeBuffer, batch_size: int, num_epochs: int):
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def sample_minibatches(self, agent_idx: int):
        """Generate minibatches for one agent"""
        data = self.buffer.get_data(agent_idx)
        buffer_len = len(self.buffer)

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(buffer_len)

            for start_idx in range(0, buffer_len, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_len)
                batch_indices = indices[start_idx:end_idx]

                yield {
                    'observations': data['observations'][batch_indices],
                    'global_states': data['global_states'][batch_indices],
                    'actions': data['actions'][batch_indices],
                    'log_probs': data['log_probs'][batch_indices],
                    'advantages': data['advantages'][batch_indices],
                    'returns': data['returns'][batch_indices],
                }

    def sample_critic_minibatches(self):
        """Generate minibatches for critic training (Agent 1 returns only)"""
        data = self.buffer.get_all_data()
        buffer_len = len(self.buffer)

        # Critic predicts Agent 1's return ONLY
        agent1_returns = data['returns'][:, 1]

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(buffer_len)

            for start_idx in range(0, buffer_len, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_len)
                batch_indices = indices[start_idx:end_idx]

                yield {
                    'global_states': data['global_states'][batch_indices],
                    'returns': agent1_returns[batch_indices],
                    'values': data['values'][batch_indices],
                }
