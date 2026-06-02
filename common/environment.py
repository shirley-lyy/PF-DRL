"""
Multi-agent Gym environment for UAV-assisted IoV.

Implements the MDP defined in Section III-A of PF-DRL:
  - Observation space  (Eq. 7)
  - Action space       (Eq. 8)
  - Reward function    (Eq. 9)
  - Channel model      (Eqs. 1-3)
  - Energy model       (Eqs. 4-5)
"""

import math
import numpy as np
import gym
from gym import spaces

from common.world import World, Agent


class MultiAgentEnv(gym.Env):
    """
    Centralized training / decentralized execution environment.

    Observations per agent  : [own_pos | user_rel_pos... | other_agent_rel_pos...]
    Actions per agent       : 5-dim discrete-encoded continuous velocity (x+, x-, y+, y-, no-op)
    Reward                  : shared cooperative throughput − individual energy − safety penalties
    """

    metadata = {"render.modes": []}

    def __init__(self, world: World, reset_callback=None,
                 info_callback=None, done_callback=None, args=None):
        self.world = world
        self.args = args
        self.agents = world.agents
        self.n = len(self.agents)

        self.reset_callback = reset_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.time = 0

        # Store previous positions for energy computation (avoid file I/O)
        self._prev_pos = [np.zeros(world.dim_p) for _ in range(self.n)]

        # Build observation / action spaces
        self.observation_space = []
        self.action_space = []
        for agent in self.agents:
            obs_dim = self._obs_dim()
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            )
            # 5-dim: [no-op, +x, -x, +y, -y]
            self.action_space.append(spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32))

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self):
        """Reset episode: restore initial UAV positions and randomize users."""
        self.time = 0

        # Fixed initial UAV positions at the four corners (Section V-A)
        init_positions = [
            np.array([-1.0, -1.0]),
            np.array([-1.0,  1.0]),
            np.array([ 1.0, -1.0]),
            np.array([ 1.0,  1.0]),
        ]
        for i, agent in enumerate(self.agents):
            agent.state.p_pos = init_positions[i].copy()
            agent.state.p_vel = np.zeros(self.world.dim_p)
            agent.state.c = np.zeros(self.world.dim_c)

        self._prev_pos = [a.state.p_pos.copy() for a in self.agents]

        # Reproducible user and obstacle placement (Section V-A, seed 42)
        np.random.seed(42)
        for user in self.world.users:
            user.state.p_pos = np.random.uniform(-0.8, 0.8, self.world.dim_p)
            user.state.p_vel = np.zeros(self.world.dim_p)
        for landmark in self.world.landmarks:
            landmark.state.p_pos = np.random.uniform(-0.8, 0.8, self.world.dim_p)
            landmark.state.p_vel = np.zeros(self.world.dim_p)

        return [self._get_obs(agent) for agent in self.agents]

    def step(self, action_n):
        """
        Execute one time slot.

        Parameters
        ----------
        action_n : list of np.ndarray, shape (5,) per agent

        Returns
        -------
        obs_n, reward_n, done_n, info_n
        """
        # Record positions *before* movement for energy computation
        for i, agent in enumerate(self.agents):
            self._prev_pos[i] = agent.state.p_pos.copy()

        # Translate discrete-encoded actions → continuous velocity commands
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)

        # Physics step
        self.world.step()
        self.time += 1

        obs_n, reward_n, done_n, info_n = [], [], [], {"n": []}
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(i, agent))
            done_n.append(self._get_done(agent))
            info_n["n"].append(self._get_info(agent))

        # Shared cooperative reward: average across agents (Eq. 9 global term)
        mean_reward = float(np.mean(reward_n))
        reward_n = [mean_reward] * self.n

        return obs_n, reward_n, done_n, info_n

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs_dim(self) -> int:
        """Dimension of o_n(t) = [P_n | {U_m - P_n} | {P_k - P_n}_{k≠n}] (Eq. 7)."""
        return (
            self.world.dim_p                          # own position
            + len(self.world.users) * self.world.dim_p    # relative user positions
            + (self.n - 1) * self.world.dim_p            # relative neighbor positions
        )

    def _get_obs(self, agent: Agent) -> np.ndarray:
        """Build observation vector for one agent (Eq. 7)."""
        # Relative positions of all vehicular users
        user_rel = [u.state.p_pos - agent.state.p_pos for u in self.world.users]
        # Relative positions of other UAV agents
        other_rel = [
            other.state.p_pos - agent.state.p_pos
            for other in self.agents if other is not agent
        ]
        return np.concatenate([agent.state.p_pos] + user_rel + other_rel).astype(np.float32)

    # ── Reward (Eq. 9) ────────────────────────────────────────────────────────

    def _get_reward(self, agent_idx: int, agent: Agent) -> float:
        """
        r_n(t) = (1/N) Σ_i R_i(t) / R_norm
                 − λ1 * (E^f_n + E^c_n) / E_norm
                 − λ2 * C_n(t)
        """
        args = self.args

        # ── Global cooperative throughput term ────────────────────────────────
        # Aggregate throughput of ALL UAVs (averaged), normalized
        total_throughput = sum(self._uav_throughput(a) for a in self.agents)
        r_throughput = (total_throughput / self.n) / args.r_norm

        # ── Local energy cost ─────────────────────────────────────────────────
        displacement = np.linalg.norm(agent.state.p_pos - self._prev_pos[agent_idx])
        # velocity magnitude estimate: Δpos / Δt
        velocity = displacement / self.world.dt

        ef = self._propulsion_energy(velocity)
        agent_throughput = self._uav_throughput(agent)
        ec = self._computing_energy(agent_throughput)
        r_energy = (ef + ec) / args.e_norm

        # ── Safety penalty C_n(t) ─────────────────────────────────────────────
        penalty = 0.0
        # UAV–UAV collision (constraint 6b)
        for other in self.agents:
            if other is agent:
                continue
            if self._is_collision(agent, other):
                penalty += args.penalty
        # UAV–obstacle collision (constraint 6c)
        for landmark in self.world.landmarks:
            if self._is_collision(agent, landmark):
                penalty += args.penalty
        # Boundary violation (constraint 6d)
        if self._outside_boundary(agent):
            penalty += args.penalty

        reward = r_throughput - args.lambda1 * r_energy - args.lambda2 * (penalty / args.penalty if penalty > 0 else 0)
        return float(reward)

    # ── Channel model (Eqs. 1-3) ──────────────────────────────────────────────

    def _a2g_rate(self, dist_2d: float, h_uav: float = 1.0) -> float:
        """
        Uplink data rate R_{m,n}(t) from one VU to the serving UAV.

        Parameters
        ----------
        dist_2d : float  horizontal distance between UAV and VU (normalized)
        h_uav   : float  UAV altitude (normalized)
        """
        # Environment constants (urban, Section V-A)
        alpha_env, beta_env = 9.61, 0.16
        eta_los, eta_nlos = 1.0, 20.0        # additional path loss (dB)
        fc = 2.4e8                            # carrier frequency (Hz)
        bandwidth = 1e6                       # 1 MHz per UAV (Section V-A)
        pu_dBm = -10.0                        # transmit power (dBm)
        sigma_dBm = -110.0                    # noise power (dBm)

        pu_dB = pu_dBm - 30.0               # dBm → dB(W)
        sigma_dB = sigma_dBm - 30.0

        # LoS probability (Eq. 1)
        elev_angle = math.atan2(h_uav, max(dist_2d, 1e-6))
        p_los = 1.0 / (1.0 + alpha_env * math.exp(-beta_env * (math.degrees(elev_angle) - alpha_env)))

        # Euclidean 3-D distance
        d3d = math.sqrt(dist_2d ** 2 + h_uav ** 2)

        # Average path loss (Eq. 2)
        pl_fs = 20.0 * math.log10(max(4 * math.pi * fc * d3d / 3e8, 1e-10))
        pl = pl_fs + p_los * eta_los + (1.0 - p_los) * eta_nlos

        # SNR and rate (Eq. 3)
        snr_dB = pu_dB - pl - sigma_dB
        snr_linear = 10.0 ** (snr_dB / 10.0)
        rate = bandwidth * math.log2(1.0 + snr_linear)
        return rate

    def _uav_throughput(self, agent: Agent) -> float:
        """
        R_n(t) = Σ_{m ∈ M_n} R_{m,n}(t)  with equal bandwidth sharing.
        Serves the top-5 nearest VUs (representative subset for computational efficiency).
        """
        distances = [
            np.linalg.norm(agent.state.p_pos - u.state.p_pos)
            for u in self.world.users
        ]
        # Sort and take the K nearest users
        K = min(5, len(self.world.users))
        nearest_dists = sorted(distances)[:K]
        # Equal bandwidth → each user gets B / K (captured by dividing rate by K)
        total_rate = sum(self._a2g_rate(d) / K for d in nearest_dists)
        return total_rate

    # ── Energy model (Eqs. 4-5) ───────────────────────────────────────────────

    def _propulsion_energy(self, velocity: float) -> float:
        """
        Propulsion power P^f_n (Eq. 4) × δt.

        Aerodynamic constants are set to nominal values; override via args if needed.
        """
        P0 = 1.0   # blade profile power
        Pi = 1.0   # induced power
        Utip = 1.0  # rotor tip speed
        v0 = 1.0    # mean rotor induced velocity (hovering)
        d0 = 1.0    # fuselage drag ratio
        rho = 1.0   # air density
        s = 1.0     # rotor solidity
        A = 1.0     # rotor disc area

        v = velocity
        term1 = P0 * (1.0 + 3.0 * v ** 2 / Utip ** 2)
        inner = math.sqrt(max(1.0 + v ** 4 / (4.0 * v0 ** 4), 0.0)) - v ** 2 / (2.0 * v0 ** 2)
        term2 = Pi * math.sqrt(max(inner, 0.0))
        term3 = 0.5 * d0 * rho * s * A * v ** 3
        power = term1 + term2 + term3
        return power * self.world.dt

    @staticmethod
    def _computing_energy(throughput: float) -> float:
        """
        Computing energy E^c_n (Eq. 5).

        E^c_n = γ_c * C_n * (Σ R_{m,n})^2 * f_cpu^2 * δt
        Using simplified form: γ_c * f_cpu^2 * R^2 * δt
        """
        gamma_c = 1e-27   # effective switched capacitance
        C_n = 1000.0      # CPU cycles per bit
        f_cpu = 2e9       # CPU frequency (Hz)
        dt = 0.1
        return gamma_c * C_n * (throughput ** 2) * (f_cpu ** 2) * dt

    # ── Action ────────────────────────────────────────────────────────────────

    def _set_action(self, action: np.ndarray, agent: Agent) -> None:
        """
        Map 5-dim action → 2-D velocity command.

        action[0]: no-op
        action[1]: +x
        action[2]: -x
        action[3]: +y
        action[4]: -y
        """
        if not agent.movable:
            return
        agent.action_u = np.zeros(self.world.dim_p)
        agent.action_u[0] = action[1] - action[2]
        agent.action_u[1] = action[3] - action[4]

        # Soft boundary enforcement (constraint 6d): reflect velocity at boundary
        bound = 1.0
        if agent.state.p_pos[0] > bound:
            agent.action_u[0] = min(agent.action_u[0], -5.0)
        if agent.state.p_pos[0] < -bound:
            agent.action_u[0] = max(agent.action_u[0], 5.0)
        if agent.state.p_pos[1] > bound:
            agent.action_u[1] = min(agent.action_u[1], -5.0)
        if agent.state.p_pos[1] < -bound:
            agent.action_u[1] = max(agent.action_u[1], 5.0)

    # ── Safety checks ─────────────────────────────────────────────────────────

    def _is_collision(self, entity_a, entity_b) -> bool:
        dist = np.linalg.norm(entity_a.state.p_pos - entity_b.state.p_pos)
        return dist < (entity_a.size + entity_b.size)

    def _outside_boundary(self, agent: Agent) -> bool:
        bound = 1.5
        return bool(np.any(np.abs(agent.state.p_pos) > bound))

    # ── Info / Done ───────────────────────────────────────────────────────────

    def _get_info(self, agent: Agent) -> dict:
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def _get_done(self, agent: Agent) -> bool:
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)