import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="PF-DRL: Personalized Federated Deep Reinforcement Learning for Multi-UAV Trajectory Optimization in IoV"
    )

    # ── Environment ──────────────────────────────────────────────────────────
    parser.add_argument("--scenario-name", type=str, default="complex_service",
                        help="Name of the scenario script")
    parser.add_argument("--n-agents", type=int, default=4,
                        help="Number of UAV agents (N)")
    parser.add_argument("--n-users", type=int, default=45,
                        help="Number of vehicular users (M)")
    parser.add_argument("--n-landmarks", type=int, default=3,
                        help="Number of static obstacles")
    parser.add_argument("--max-episode-len", type=int, default=50,
                        help="Maximum time slots per episode (T)")
    parser.add_argument("--n-episodes", type=int, default=40,
                        help="Total number of training episodes (N_max)")
    parser.add_argument("--high-action", type=float, default=1.0,
                        help="Maximum action amplitude (V_max scaling)")

    # ── Core Training Parameters ──────────────────────────────────────────────
    parser.add_argument("--lr-actor", type=float, default=1e-4,
                        help="Learning rate of Actor network")
    parser.add_argument("--lr-critic", type=float, default=1e-3,
                        help="Learning rate of Critic network")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor (γ)")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Soft target update coefficient (τ)")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
                        help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for training")

    # ── Exploration ───────────────────────────────────────────────────────────
    parser.add_argument("--noise-rate", type=float, default=0.2,
                        help="Ornstein-Uhlenbeck noise initial magnitude (σ)")
    parser.add_argument("--noise-theta", type=float, default=0.15,
                        help="Ornstein-Uhlenbeck noise mean-reversion rate (θ)")
    parser.add_argument("--noise-decay-episodes", type=int, default=10,
                        help="Episodes over which exploration noise linearly decays to zero")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon-greedy exploration probability")

    # ── Federated Learning ────────────────────────────────────────────────────
    parser.add_argument("--training-mode", type=int, default=0,
                        choices=[0, 1, 2],
                        help="Training mode: 0=MADDPG, 1=PF-MADDPG (fixed α), 2=PF-MADDPG (adaptive α)")
    parser.add_argument("--agg-interval", type=int, default=5,
                        help="Federated aggregation interval in episodes (τ_agg)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Personalization weight α ∈ [0,1]: 1=local-only, 0=global-only")
    parser.add_argument("--alpha-step", type=float, default=0.003,
                        help="Step size δ_α for adaptive weight adjustment (Algorithm 2)")
    parser.add_argument("--ema-beta", type=float, default=0.9,
                        help="EMA smoothing factor β_R for adaptive weight adjustment")
    parser.add_argument("--ema-threshold", type=float, default=1e-3,
                        help="Tolerance threshold ε_R to avoid reactive α changes")
    parser.add_argument("--lightweight", action="store_true", default=True,
                        help="Lightweight mode: aggregate Critic only (reduces comm. by ~50%%)")

    # ── Reward Weights ────────────────────────────────────────────────────────
    parser.add_argument("--lambda1", type=float, default=0.1,
                        help="Energy cost weight λ_1 in reward function")
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="Safety penalty weight λ_2 in reward function")
    parser.add_argument("--penalty", type=float, default=10.0,
                        help="Collision/boundary violation penalty magnitude Ψ")
    parser.add_argument("--r-norm", type=float, default=1e7,
                        help="Throughput normalization constant R_norm (bps)")
    parser.add_argument("--e-norm", type=float, default=100.0,
                        help="Energy normalization constant E_norm (J)")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model-dir", type=str, default="",
                        help="Directory to load pre-trained models (empty = train from scratch)")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for training logs and reward curves")

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="Run in evaluation mode (no training)")
    parser.add_argument("--evaluate-rate", type=int, default=1,
                        help="Evaluate every N episodes during training")
    parser.add_argument("--evaluate-episodes", type=int, default=5,
                        help="Number of evaluation episodes to average over")

    args = parser.parse_args()
    return args