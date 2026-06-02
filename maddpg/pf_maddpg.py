"""
PF-MADDPG agent implementation.

Supports three training modes (--training-mode):
  0 : Standard MADDPG (no federation)
  1 : PF-MADDPG with fixed personalization weight α
  2 : PF-MADDPG with adaptive α (Algorithm 2)

Lightweight mode (--lightweight): aggregate Critic only (≈50% comm. saving).
"""

import os
import collections
import numpy as np
import torch
import torch.nn as nn

from maddpg.actor_critic import Actor, Critic
import maddpg.fed_average as fed_average


class PFMADDPGAgent:
    """
    One UAV agent in the PF-MADDPG framework.

    Parameters
    ----------
    args     : argparse.Namespace
    agent_id : int  index n ∈ {0, …, N-1}
    """

    def __init__(self, args, agent_id: int):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor = Actor(args, agent_id)
        self.critic = Critic(args)
        self.actor_target = Actor(args, agent_id)
        self.critic_target = Critic(args)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimizers ────────────────────────────────────────────────────────
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        # ── Adaptive weight state (Algorithm 2) ───────────────────────────────
        self.alpha = args.alpha          # current personalization weight
        self._alpha_dir = 1             # search direction d ∈ {+1, -1}
        self._ema_return = None          # smoothed episode return R̃_k
        self._best_ema = -float("inf")  # R̃_best

    # ── Public API ────────────────────────────────────────────────────────────

    def select_action(self, obs: "np.ndarray", noise_scale: float) -> "np.ndarray":
        """
        π(o_n | θ_n) + exploration noise.

        The OUNoise is applied externally by Runner for full control over decay.
        Here we simply add Gaussian noise if noise_scale > 0.
        """
        import numpy as np
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).numpy()
        self.actor.train()
        if noise_scale > 0:
            action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -self.args.high_action, self.args.high_action)

    def learn(self, transitions: dict) -> dict:
        """
        One gradient update step (Eqs. 10-11) + optional federated personalization.

        Parameters
        ----------
        transitions : dict  mini-batch sampled from ReplayBuffer

        Returns
        -------
        info : dict  {"actor_loss": float, "critic_loss": float}
        """
        # ── Unpack transitions ────────────────────────────────────────────────
        for key in transitions:
            transitions[key] = torch.FloatTensor(transitions[key])

        r = transitions[f"r_{self.agent_id}"]
        o, u, o_next = [], [], []
        for i in range(self.args.n_agents):
            o.append(transitions[f"o_{i}"])
            u.append(transitions[f"u_{i}"])
            o_next.append(transitions[f"o_next_{i}"])

        # ── Target Q-value (Eq. 10) ───────────────────────────────────────────
        with torch.no_grad():
            u_next = [self.actor_target(o_next[i]) for i in range(self.args.n_agents)]
            q_next = self.critic_target(o_next, u_next)
            target_q = r.unsqueeze(1) + self.args.gamma * q_next

        # ── Critic update ─────────────────────────────────────────────────────
        q_val = self.critic(o, u)
        critic_loss = nn.functional.mse_loss(q_val, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # ── Actor update (Eq. 11) ─────────────────────────────────────────────
        u_new = list(u)   # copy; only replace current agent's action
        u_new[self.agent_id] = self.actor(o[self.agent_id])
        actor_loss = -self.critic(o, u_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # ── Save local models ─────────────────────────────────────────────────
        self._save_local_models()

        # ── Soft target update ────────────────────────────────────────────────
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        self.train_step += 1
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def federated_personalize(self, episode: int) -> None:
        """
        Execute federated aggregation + personalized interpolation (Eq. 13).

        Called by Runner every τ_agg episodes when training_mode ∈ {1, 2}.

        Parameters
        ----------
        episode : int  current training episode k
        """
        args = self.args
        scenario = args.scenario_name

        # ── Aggregate global model(s) ─────────────────────────────────────────
        if not args.lightweight:
            # Full PF-MADDPG: aggregate both Actor and Critic
            fed_average.fed_actor(args.save_dir, scenario, args.n_agents)
            global_actor_path = os.path.join(args.save_dir, scenario, "global", "fl_actor.pth")
            self.actor = fed_average.personalized_interpolation(
                self.actor, global_actor_path, self.alpha
            )

        # Lightweight PF-MADDPG (default): aggregate Critic only (Section IV-C)
        fed_average.fed_critic(args.save_dir, scenario, args.n_agents)
        global_critic_path = os.path.join(args.save_dir, scenario, "global", "fl_critic.pth")
        self.critic = fed_average.personalized_interpolation(
            self.critic, global_critic_path, self.alpha
        )

        # Soft-update target networks with personalized model
        self._soft_update(self.critic_target, self.critic)
        if not args.lightweight:
            self._soft_update(self.actor_target, self.actor)

    def update_alpha(self, episode_return: float) -> None:
        """
        Adaptive weight adjustment (Algorithm 2).

        Uses EMA-smoothed return to perform perturb-and-observe hill-climbing
        over the personalization weight α.

        Parameters
        ----------
        episode_return : float  raw episodic return R_k
        """
        args = self.args

        # EMA smoothing (Eq. 20)
        if self._ema_return is None:
            self._ema_return = episode_return
        else:
            self._ema_return = args.ema_beta * self._ema_return + (1.0 - args.ema_beta) * episode_return

        r_smooth = self._ema_return
        eps = args.ema_threshold

        # Perturb-and-observe direction update (Algorithm 2, lines 5-12)
        if r_smooth >= self._best_ema + eps:
            # Improvement → maintain direction
            self._best_ema = r_smooth
        elif r_smooth < self._best_ema - eps:
            # Degradation → reverse direction
            self._alpha_dir *= -1
        # else: within tolerance → conservative (keep direction)

        # Weight update and clipping (Algorithm 2, lines 13-15)
        self.alpha = float(np.clip(self.alpha + self._alpha_dir * args.alpha_step, 0.0, 1.0))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """τ-weighted soft update of target network (τ from args)."""
        tau = self.args.tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)

    def _save_local_models(self) -> None:
        """Persist local Actor and Critic weights for federated aggregation."""
        path = os.path.join(
            self.args.save_dir, self.args.scenario_name, f"agent_{self.agent_id}"
        )
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor, os.path.join(path, "actor.pth"))
        torch.save(self.critic, os.path.join(path, "critic.pth"))

    def _local_model_path(self, net_name: str) -> str:
        return os.path.join(
            self.args.save_dir, self.args.scenario_name,
            f"agent_{self.agent_id}", f"{net_name}.pth"
        )