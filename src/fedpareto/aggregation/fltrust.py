import numpy as np
import torch

from fedpareto.utils import average_state_dicts, cosine_similarity, flatten_delta


def fltrust_aggregate(client_results, root_delta):
    root_vec = flatten_delta(root_delta)
    trust_scores = []
    normalized_deltas = []
    root_norm = root_vec.norm().item() + 1e-12

    for client in client_results:
        vec = flatten_delta(client.delta)
        cosine = max(0.0, cosine_similarity(vec, root_vec))
        trust_scores.append(cosine)
        vec_norm = vec.norm().item() + 1e-12
        scale = root_norm / vec_norm
        normalized = {}
        for key, value in client.delta.items():
            if torch.is_floating_point(value) or torch.is_complex(value):
                normalized[key] = value * scale
            else:
                normalized[key] = torch.zeros_like(value)
        normalized_deltas.append(normalized)

    scores = np.asarray(trust_scores, dtype=np.float64)
    weights = np.ones_like(scores) / len(scores) if scores.sum() <= 0 else scores / scores.sum()
    aggregate = average_state_dicts(normalized_deltas, weights)
    return aggregate, weights.tolist(), {"method": "fltrust", "trust_scores": trust_scores}
