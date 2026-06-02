"""
World physics engine for UAV-assisted IoV simulation.
Entities: UAV agents, vehicular users, static obstacles (landmarks).
"""

import numpy as np


# ── State containers ──────────────────────────────────────────────────────────

class EntityState:
    """2-D kinematic state (position + velocity) and communication channel."""

    def __init__(self):
        self.p_pos: np.ndarray = None   # [x, y]
        self.p_vel: np.ndarray = None   # [vx, vy]
        self.c: np.ndarray = None       # communication state


# ── Entity base classes ───────────────────────────────────────────────────────

class Entity:
    """Abstract physical entity in the world."""

    def __init__(self):
        self.name: str = ""
        self.size: float = 0.05
        self.collide: bool = False
        self.movable: bool = False
        self.density: float = 25.0
        self.color = None
        self.max_speed: float = None
        self.state = EntityState()


class Landmark(Entity):
    """Static cylindrical obstacle (e.g., high-rise building)."""

    def __init__(self):
        super().__init__()
        self.collide = True
        self.movable = False
        self.size = 0.10   # obstacle radius in normalized units


class User(Entity):
    """Ground vehicular user (VU)."""

    def __init__(self):
        super().__init__()
        self.collide = False
        self.movable = False
        self.size = 0.05


class Agent(Entity):
    """UAV agent acting as an Aerial Edge Node (AEN)."""

    def __init__(self):
        super().__init__()
        self.movable = True
        self.silent = True
        self.collide = True
        self.max_speed = 0.05          # maps to V_max in normalized coordinates
        self.size = 0.05
        self.action_u: np.ndarray = None   # 2-D velocity command [vx, vy]
        # Per-UAV hardware profile (heterogeneity support, Section V-E5)
        self.v_max_scale: float = 1.0      # relative speed capability
        self.beamwidth: float = 90.0       # antenna half-power beamwidth θ (degrees)


# ── World ─────────────────────────────────────────────────────────────────────

class World:
    """
    Multi-agent physical world.

    Entities are laid out as: agents + users + landmarks.
    Only agents are movable; users and landmarks are static.
    """

    def __init__(self):
        self.agents: list[Agent] = []
        self.users: list[User] = []
        self.landmarks: list[Landmark] = []

        self.dim_c: int = 2       # communication channel dimension
        self.dim_p: int = 2       # position / velocity dimension
        self.dim_color: int = 3

        # Physics parameters
        self.dt: float = 0.1
        self.damping: float = 0.25
        self.contact_force: float = 1e2
        self.contact_margin: float = 1e-3

        self.collaborative: bool = True

    # ── Entity accessors ──────────────────────────────────────────────────────

    @property
    def entities(self) -> list:
        return self.agents + self.users + self.landmarks

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the world by one time slot δt."""
        p_force = [None] * len(self.entities)
        p_force = self._apply_action_force(p_force)
        p_force = self._apply_environment_force(p_force)
        self._integrate_state(p_force)

    # ── Internal physics ──────────────────────────────────────────────────────

    def _apply_action_force(self, p_force: list) -> list:
        for i, agent in enumerate(self.agents):
            if agent.movable:
                p_force[i] = agent.action_u
        return p_force

    def _apply_environment_force(self, p_force: list) -> list:
        """Soft-contact collision response between all entity pairs."""
        entities = self.entities
        for a, entity_a in enumerate(entities):
            for b, entity_b in enumerate(entities):
                if b <= a:
                    continue
                f_a, f_b = self._get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    p_force[a] = f_a if p_force[a] is None else p_force[a] + f_a
                if f_b is not None:
                    p_force[b] = f_b if p_force[b] is None else p_force[b] + f_b
        return p_force

    def _integrate_state(self, p_force: list) -> None:
        """Euler integration with velocity damping and speed clipping."""
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            # Damping
            entity.state.p_vel = entity.state.p_vel * (1.0 - self.damping)
            # Apply force
            if p_force[i] is not None:
                entity.state.p_vel += p_force[i] * self.dt
            # Speed limit (V_max constraint 6f)
            if entity.max_speed is not None:
                speed = np.linalg.norm(entity.state.p_vel)
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
            # Position update: P_n(t+1) = P_n(t) + a_n(t) * δt
            entity.state.p_pos += entity.state.p_vel * self.dt

    def _get_collision_force(self, entity_a: Entity, entity_b: Entity):
        """Soft-contact repulsion force (used for UAV–obstacle and UAV–UAV)."""
        if not (entity_a.collide and entity_b.collide):
            return None, None
        if entity_a is entity_b:
            return None, None

        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = entity_a.size + entity_b.size

        # Soft penetration via log-sum-exp
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / (dist + 1e-8) * penetration

        f_a = +force if entity_a.movable else None
        f_b = -force if entity_b.movable else None
        return f_a, f_b