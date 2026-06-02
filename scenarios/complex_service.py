"""
IoV scenario: N UAVs, M vehicular users, K static obstacles.

Matches Section V-A simulation setup:
  - 4 UAVs initially at corners of [-1, 1]^2 (normalized)
  - 45 vehicular users randomly placed in [-0.8, 0.8]^2
  - 3 cylindrical static obstacles
"""

import numpy as np
from common.world import World, Agent, User, Landmark


class Scenario:
    """Factory for the UAV-assisted IoV world."""

    def make_world(self) -> World:
        world = World()
        world.collaborative = True

        n_agents = 4
        n_users = 45
        n_landmarks = 3

        world.agents = [Agent() for _ in range(n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"uav_{i}"

        world.users = [User() for _ in range(n_users)]
        for i, user in enumerate(world.users):
            user.name = f"vehicle_{i}"

        world.landmarks = [Landmark() for _ in range(n_landmarks)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"obstacle_{i}"

        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        """Reset agent positions and randomize user/obstacle placement."""
        init_positions = [
            np.array([-1.0, -1.0]),
            np.array([-1.0,  1.0]),
            np.array([ 1.0, -1.0]),
            np.array([ 1.0,  1.0]),
        ]
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = init_positions[i].copy()
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        np.random.seed(42)   # reproducible deployment (Section V-A)
        for user in world.users:
            user.state.p_pos = np.random.uniform(-0.8, 0.8, world.dim_p)
            user.state.p_vel = np.zeros(world.dim_p)

        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-0.8, 0.8, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)