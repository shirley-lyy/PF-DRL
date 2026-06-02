"""
Federated averaging utilities for PF-MADDPG.

Implements Eq. (12): Θ_global = Σ_n ρ_n Θ_n  (uniform weights → FedAvg).

Two functions are provided:
  fed_actor()  – aggregate Actor networks (Full PF-MADDPG)
  fed_critic() – aggregate Critic networks (Lightweight PF-MADDPG and Full)
"""

import os
import collections
import torch


def _fedavg(model_paths: list[str], save_path: str) -> None:
    """
    Load models from `model_paths`, compute uniform FedAvg, save to `save_path`.

    Parameters
    ----------
    model_paths : list of str  paths to local model files
    save_path   : str          path to write the aggregated global model
    """
    models = [torch.load(p, map_location="cpu") for p in model_paths]
    state_dicts = [m.state_dict() for m in models]
    keys = list(state_dicts[0].keys())
    n = len(models)

    avg_state = collections.OrderedDict()
    for key in keys:
        avg_state[key] = sum(sd[key] for sd in state_dicts) / n

    # Use the first model as template and overwrite weights
    global_model = models[0]
    global_model.load_state_dict(avg_state)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(global_model, save_path)


def fed_actor(save_dir: str, scenario_name: str, n_agents: int) -> None:
    """Aggregate local Actor networks → global Actor (Eq. 12, Θ = θ)."""
    local_paths = [
        os.path.join(save_dir, scenario_name, f"agent_{i}", "actor.pth")
        for i in range(n_agents)
    ]
    global_path = os.path.join(save_dir, scenario_name, "global", "fl_actor.pth")
    _fedavg(local_paths, global_path)


def fed_critic(save_dir: str, scenario_name: str, n_agents: int) -> None:
    """Aggregate local Critic networks → global Critic (Eq. 12, Θ = µ)."""
    local_paths = [
        os.path.join(save_dir, scenario_name, f"agent_{i}", "critic.pth")
        for i in range(n_agents)
    ]
    global_path = os.path.join(save_dir, scenario_name, "global", "fl_critic.pth")
    _fedavg(local_paths, global_path)


def personalized_interpolation(
    local_model: torch.nn.Module,
    global_model_path: str,
    alpha: float,
) -> torch.nn.Module:
    """
    Personalized model update (Eq. 13):
        Θ^new_n = α * Θ^local_n + (1-α) * Θ_global

    Parameters
    ----------
    local_model       : torch.nn.Module  the agent's current local model
    global_model_path : str              path to the aggregated global model file
    alpha             : float            personalization weight ∈ [0, 1]

    Returns
    -------
    local_model with updated weights (in-place)
    """
    global_model = torch.load(global_model_path, map_location="cpu")
    local_sd = local_model.state_dict()
    global_sd = global_model.state_dict()

    new_sd = collections.OrderedDict()
    for key in local_sd:
        new_sd[key] = alpha * local_sd[key] + (1.0 - alpha) * global_sd[key]

    local_model.load_state_dict(new_sd)
    return local_model