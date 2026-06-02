"""
Training loop for PF-MADDPG.

Implements Algorithm 1:
  - Episode loop with per-step MADDPG local updates
  - Federated aggregation every τ_agg episodes (lines 10-25)
  - Adaptive α update (Algorithm 2, line 24)
  - Reward logging and CSV export
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from agent import Agent
from common.replay_buffer import ReplayBuffer


class Runner:
    """
    Orchestrates training of N PF-MADDPG agents in the IoV environment.

    Parameters
    ----------
    args : argparse.Namespace
    env  : MultiAgentEnv
    """

    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_agents = args.n_agents

        # Instantiate agents (one per UAV)
        self.agents = [Agent(i, args) for i in range(self.n_agents)]

        # Replay buffer (shared, indexed by agent ID)
        self.buffer = ReplayBuffer(args)

        # Output directories
        self.log_dir = os.path.join(args.log_dir, args.scenario_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.scenario_name, "global"), exist_ok=True)

        # Noise schedule: linearly decay over the first N_decay episodes
        self._noise_init = args.noise_rate
        self._noise_decay_episodes = args.noise_decay_episodes

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute Algorithm 1: full training loop."""
        episode_returns = []

        for episode in tqdm(range(1, self.args.n_episodes + 1), desc="Episodes"):
            ep_return = self._run_episode(episode)
            episode_returns.append(ep_return)

            # ── Federated aggregation (Algorithm 1, lines 10-25) ──────────────
            if (episode % self.args.agg_interval == 0
                    and self.args.training_mode in {1, 2}
                    and len(self.buffer) >= self.args.batch_size):

                for agent in self.agents:
                    agent.federated_personalize(episode)

                # Adaptive α update (Algorithm 2)
                if self.args.training_mode == 2:
                    for agent in self.agents:
                        agent.update_alpha(ep_return)

            # Logging
            tqdm.write(
                f"[Episode {episode:4d}]  Return: {ep_return:.4f}"
                f"  α: {self.agents[0].policy.alpha:.3f}"
            )

        self._save_results(episode_returns)

    # ── Episode execution ─────────────────────────────────────────────────────

    def _run_episode(self, episode: int) -> float:
        """Run one full episode; return total discounted return for agent 0."""
        obs = self.env.reset()
        for agent in self.agents:
            agent.noise.reset()

        # Compute current noise scale (linear decay, Section V-A)
        noise_scale = self._noise_init * max(
            0.0, 1.0 - (episode - 1) / max(self._noise_decay_episodes, 1)
        )

        ep_return = 0.0
        for t in range(self.args.max_episode_len):
            # ── Action selection (Algorithm 1, line 5) ────────────────────────
            actions = []
            with __import__("torch").no_grad():
                for i, agent in enumerate(self.agents):
                    act = agent.select_action(obs[i], noise_scale)
                    actions.append(act)

            # ── Environment step ──────────────────────────────────────────────
            obs_next, rewards, dones, info = self.env.step(actions)

            # ── Store transition (Algorithm 1, line 7) ────────────────────────
            self.buffer.store_episode(
                obs[:self.n_agents],
                actions,
                rewards[:self.n_agents],
                obs_next[:self.n_agents],
            )

            # ── Local MADDPG update (Algorithm 1, line 8) ─────────────────────
            if len(self.buffer) >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    agent.learn(transitions)

            ep_return += rewards[0]
            obs = obs_next

            if all(dones):
                break

        return ep_return

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_results(self, returns: list) -> None:
        """Save episode return history to a CSV file."""
        mode_names = {0: "MADDPG", 1: "PF-MADDPG_fixed", 2: "PF-MADDPG_adaptive"}
        name = mode_names.get(self.args.training_mode, "unknown")
        out_path = os.path.join(self.log_dir, f"returns_{name}.csv")

        df = pd.DataFrame({"episode": range(1, len(returns) + 1), "return": returns})
        df.to_csv(out_path, index=False)
        print(f"\nTraining complete. Results saved to: {out_path}")