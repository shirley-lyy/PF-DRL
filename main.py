"""
Entry point for PF-DRL training / evaluation.

Usage
-----
# Train with standard MADDPG (mode 0):
python main.py --training-mode 0

# Train with Lightweight PF-MADDPG, fixed α=0.5 (mode 1):
python main.py --training-mode 1 --alpha 0.5 --lightweight

# Train with Adaptive PF-MADDPG (mode 2):
python main.py --training-mode 2 --lightweight

# Evaluate a saved model:
python main.py --evaluate --model-dir ./model/complex_service
"""

import os
import sys

# Ensure project root is on sys.path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

from common.arguments import get_args
from common.environment import MultiAgentEnv
from runner import Runner
import scenarios.complex_service as scenario_module


def make_env(args):
    """Instantiate the IoV environment and attach observation/action shapes to args."""
    scenario = scenario_module.Scenario()
    world = scenario.make_world()

    env = MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        args=args,
    )

    # Attach shapes needed by network constructors
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    args.action_shape = [5] * args.n_agents   # 5-dim per agent

    return env, args


def main():
    args = get_args()

    # Validate training mode
    if args.training_mode not in {0, 1, 2}:
        raise ValueError(f"--training-mode must be 0, 1, or 2; got {args.training_mode}")

    env, args = make_env(args)

    print("=" * 60)
    print("PF-DRL: Personalized Federated Deep Reinforcement Learning")
    print("        for Multi-UAV Trajectory Optimization in IoV")
    print("=" * 60)
    print(f"  Scenario    : {args.scenario_name}")
    print(f"  UAVs (N)    : {args.n_agents}")
    print(f"  Users (M)   : {args.n_users}")
    print(f"  Obstacles   : {args.n_landmarks}")
    print(f"  Mode        : {['MADDPG', 'PF-MADDPG (fixed α)', 'PF-MADDPG (adaptive α)'][args.training_mode]}")
    print(f"  Lightweight : {args.lightweight} (Critic-only aggregation)")
    print(f"  Agg. interval τ_agg : {args.agg_interval} episodes")
    print(f"  Obs dims    : {args.obs_shape}")
    print("=" * 60)

    runner = Runner(args, env)
    runner.run()


if __name__ == "__main__":
    main()