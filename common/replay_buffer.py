"""
Experience replay buffer shared across all agents.

Stores transitions as named numpy arrays indexed by agent ID.
Thread-safe via a single lock.
"""

import threading
import numpy as np


class ReplayBuffer:
    """
    Circular replay buffer storing (o, u, r, o') tuples for N agents.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: buffer_size, n_agents, obs_shape, batch_size
    """

    def __init__(self, args):
        self.size = args.buffer_size
        self.n_agents = args.n_agents
        self.current_size = 0
        self._ptr = 0

        # Pre-allocate arrays for each agent's data
        self.buffer = {}
        for i in range(self.n_agents):
            self.buffer[f"o_{i}"] = np.empty((self.size, args.obs_shape[i]), dtype=np.float32)
            self.buffer[f"u_{i}"] = np.empty((self.size, 5), dtype=np.float32)
            self.buffer[f"r_{i}"] = np.empty((self.size,), dtype=np.float32)
            self.buffer[f"o_next_{i}"] = np.empty((self.size, args.obs_shape[i]), dtype=np.float32)

        self._lock = threading.Lock()

    def store_episode(self, obs: list, actions: list, rewards: list, obs_next: list) -> None:
        """
        Store a single transition for all agents simultaneously.

        Parameters
        ----------
        obs, actions, rewards, obs_next : list of length n_agents
        """
        with self._lock:
            idx = self._ptr
            for i in range(self.n_agents):
                self.buffer[f"o_{i}"][idx] = obs[i]
                self.buffer[f"u_{i}"][idx] = actions[i]
                self.buffer[f"r_{i}"][idx] = rewards[i]
                self.buffer[f"o_next_{i}"][idx] = obs_next[i]
            self._ptr = (self._ptr + 1) % self.size
            self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size: int) -> dict:
        """Sample a random mini-batch of transitions."""
        idx = np.random.randint(0, self.current_size, batch_size)
        return {key: self.buffer[key][idx] for key in self.buffer}

    def __len__(self) -> int:
        return self.current_size