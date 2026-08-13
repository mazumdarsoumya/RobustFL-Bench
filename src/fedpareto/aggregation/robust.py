import numpy as np
import torch

from fedpareto.utils import flatten_delta


def _stack_param(client_results, key):
    return torch.stack([client.delta[key].detach().cpu().float() for client in client_results], dim=0)


def _is_float(value):
    return torch.is_floating_point(value) or torch.is_complex(value)


def coord_median_aggregate(client_results):
    out = {}
    for key, first in client_results[0].delta.items():
        if _is_float(first):
            stacked = _stack_param(client_results, key)
            out[key] = stacked.median(dim=0).values.to(first.device, dtype=first.dtype)
        else:
            out[key] = torch.zeros_like(first)
    n = len(client_results)
    return out, [1.0 / n] * n, {"method": "coord_median"}


def trimmed_mean_aggregate(client_results, trim_ratio=0.2):
    out = {}
    n = len(client_results)
    trim_k = int(float(trim_ratio) * n)
    for key, first in client_results[0].delta.items():
        if _is_float(first):
            stacked = _stack_param(client_results, key)
            values, _ = torch.sort(stacked, dim=0)
            trimmed = values[trim_k:n - trim_k] if n - 2 * trim_k > 0 else values
            out[key] = trimmed.mean(dim=0).to(first.device, dtype=first.dtype)
        else:
            out[key] = torch.zeros_like(first)
    return out, [1.0 / n] * n, {"method": "trimmed_mean", "trim_ratio": float(trim_ratio)}


def krum_aggregate(client_results, byzantine_count=None):
    vectors = [flatten_delta(client.delta) for client in client_results]
    n = len(vectors)
    f = int(byzantine_count) if byzantine_count is not None else max(0, (n - 3) // 2)
    f = min(f, max(0, n - 3))
    scores = []
    for i in range(n):
        distances = [torch.norm(vectors[i] - vectors[j]).item() ** 2 for j in range(n) if i != j]
        distances.sort()
        scores.append(sum(distances[:max(1, n - f - 2)]))
    winner = int(np.argmin(scores))
    out = {key: value.detach().clone() for key, value in client_results[winner].delta.items()}
    weights = [0.0] * n
    weights[winner] = 1.0
    return out, weights, {"method": "krum", "winner_client": client_results[winner].client_id, "scores": scores}
