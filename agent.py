"""
Agent wrapper: bridges the Runner loop with PFMADDPGAgent.

Handles OUNoise-based exploration and delegates learning/federation
to the underlying PFMADDPGAgent.
"""

import numpy as np
from maddpg.pf_maddpg import PFMADDPGAgent


class OUNoise:
    """
    Ornstein-Uhlenbeck noise process for temporally correlated exploration.

    Section V-A: θ = 0.15, σ = 0.2, linearly decays over first N_decay episodes.
    """

    def __init__(self, action_dim: int, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(action_dim)

    def reset(self) -> None:
        self.state = np.zeros(self.action_dim)

    def sample(self) -> np.ndarray:
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()


class Agent:
    """
    High-level agent used by Runner.

    Parameters
    ----------
    agent_id : int
    args     : argparse.Namespace
    """

    def __init__(self, agent_id: int, args):
        self.agent_id = agent_id
        self.args = args
        self.policy = PFMADDPGAgent(args, agent_id)
        self.noise = OUNoise(action_dim=5, theta=args.noise_theta, sigma=args.noise_rate)

    def select_action(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        """
        Choose action with optional OUNoise exploration.

        Parameters
        ----------
        obs         : np.ndarray  local observation o_n(t)
        noise_scale : float       current noise amplitude (0 during evaluation)
        """
        action = self.policy.select_action(obs, noise_scale=0.0)
        if noise_scale > 0:
            action += noise_scale * self.noise.sample()
        return np.clip(action, -self.args.high_action, self.args.high_action)

    def learn(self, transitions: dict) -> dict:
        return self.policy.learn(transitions)

    def federated_personalize(self, episode: int) -> None:
        self.policy.federated_personalize(episode)

    def update_alpha(self, episode_return: float) -> None:
        self.policy.update_alpha(episode_return)