import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _floating_tensors(state_dict):
    return [v for v in state_dict.values() if torch.is_floating_point(v) or torch.is_complex(v)]


def state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    values = _floating_tensors(state_dict)
    if not values:
        return torch.empty(0)
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in values])


def flatten_delta(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    values = _floating_tensors(delta)
    if not values:
        return torch.empty(0)
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in values])


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def subtract_state_dicts(new_state, old_state):
    out = {}
    for key in new_state:
        value = new_state[key]
        if torch.is_floating_point(value) or torch.is_complex(value):
            out[key] = value.detach().clone() - old_state[key].detach().clone()
        else:
            out[key] = torch.zeros_like(value)
    return out


def add_delta_to_state(base_state, delta, scale=1.0):
    out = {}
    for key, base in base_state.items():
        if torch.is_floating_point(base) or torch.is_complex(base):
            out[key] = base.detach().clone() + float(scale) * delta[key].detach().clone().to(base.dtype)
        else:
            out[key] = base.detach().clone()
    return out


def average_state_dicts(deltas, weights):
    out = {}
    for key, first in deltas[0].items():
        if torch.is_floating_point(first) or torch.is_complex(first):
            agg = torch.zeros_like(first)
            for delta, weight in zip(deltas, weights):
                agg = agg + delta[key].to(agg.dtype) * float(weight)
            out[key] = agg
        else:
            out[key] = torch.zeros_like(first)
    return out


def apply_state_dict(model, state_dict):
    model.load_state_dict(state_dict, strict=True)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def entropy_from_weights(weights: List[float]) -> float:
    eps = 1e-12
    values = np.asarray(weights, dtype=np.float64)
    values = np.clip(values, eps, 1.0)
    return float(-(values * np.log(values)).sum())


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    x = x.float()
    y = y.float()
    denom = (x.norm() * y.norm()).item() + 1e-12
    return float(torch.dot(x, y).item() / denom)


def project_simplex(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D")
    if scores.size == 0:
        return scores
    if np.isclose(scores.sum(), 1.0) and np.all(scores >= 0):
        return scores
    u = np.sort(scores)[::-1]
    cssv = np.cumsum(u)
    candidates = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(candidates) == 0:
        return np.ones_like(scores) / len(scores)
    rho = candidates[-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    weights = np.maximum(scores - theta, 0)
    if weights.sum() <= 0:
        return np.ones_like(scores) / len(scores)
    return weights / weights.sum()
