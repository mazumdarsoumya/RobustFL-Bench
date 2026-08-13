import numpy as np
from fedpareto.utils import average_state_dicts

def fedavg_aggregate(client_results):
    sample_counts = np.array([c.samples for c in client_results], dtype=np.float64)
    weights = sample_counts / sample_counts.sum()
    delta = average_state_dicts([c.delta for c in client_results], weights)
    return delta, weights.tolist(), {"method": "fedavg"}
