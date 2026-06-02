"""
Actor and Critic neural network definitions for PF-MADDPG.

Architecture (Section V-A):
  - 3 fully-connected hidden layers with 64 neurons each, ReLU activation.
  - Actor output: tanh scaled by max_action (continuous action in [-1, 1]^5).
  - Critic input: concatenated global state + joint actions; output: scalar Q-value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Local Actor network π(o_n | θ_n).

    Maps local observation → 5-dim continuous action.
    """

    def __init__(self, args, agent_id: int):
        super().__init__()
        self.max_action = args.high_action
        obs_dim = args.obs_shape[agent_id]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, obs_dim)

        Returns
        -------
        actions : Tensor, shape (batch, 5)  ∈ [-max_action, max_action]
        """
        return self.max_action * torch.tanh(self.net(x))


class Critic(nn.Module):
    """
    Centralized Critic network Q(s, a_1, …, a_N | µ_n).

    Input : concatenation of all agents' observations + all agents' actions.
    Output: scalar Q-value estimate.
    """

    def __init__(self, args):
        super().__init__()
        self.max_action = args.high_action
        # Global state dim = Σ obs_shape_i;  joint action dim = n_agents × 5
        state_dim = sum(args.obs_shape)
        action_dim = args.n_agents * 5

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states: list, actions: list) -> torch.Tensor:
        """
        Parameters
        ----------
        states  : list of Tensor, each (batch, obs_shape_i)
        actions : list of Tensor, each (batch, 5)

        Returns
        -------
        q_value : Tensor, shape (batch, 1)
        """
        x = torch.cat(states + actions, dim=1)
        return self.net(x)