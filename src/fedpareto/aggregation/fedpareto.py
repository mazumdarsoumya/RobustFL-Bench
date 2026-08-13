import numpy as np
from fedpareto.metrics import expected_calibration_error
from fedpareto.utils import average_state_dicts, cosine_similarity, flatten_delta, project_simplex

def pareto_ranks(points: np.ndarray):
    n = len(points)
    ranks = np.ones(n, dtype=np.int64)
    for i in range(n):
        dominated_by = 0
        for j in range(n):
            if i == j:
                continue
            ge = np.all(points[j] >= points[i])
            gt = np.any(points[j] > points[i])
            if ge and gt:
                dominated_by += 1
        ranks[i] = 1 + dominated_by
    return ranks

def _safe_norm(x):
    x = np.asarray(x, dtype=np.float64)
    if np.allclose(x.max(), x.min()):
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def compute_anchor_scores(client_results, cfg, temporal_reliability, benign_reference_vec):
    wcfg = cfg["fedpareto"]["objective_weights"]
    fairness_strength = float(cfg["fedpareto"]["fairness_strength"])
    temperature = float(cfg["fedpareto"]["temperature"])
    pareto_bonus = float(cfg["fedpareto"]["pareto_bonus"])
    trust_penalty = float(cfg["fedpareto"]["trust_penalty"])
    entropy_reg = float(cfg["fedpareto"]["entropy_reg"])

    acc_gains = []
    cal_gains = []
    fairness_scores = []
    robust_scores = []
    client_ids = []
    diagnostics = {}

    for c in client_results:
        probs = c.anchor_summary["probs"]
        labels = c.anchor_summary["labels"]
        preds = probs.argmax(dim=1)
        acc = float((preds == labels).float().mean().item())
        ece = expected_calibration_error(probs, labels, n_bins=cfg["evaluation"]["ece_bins"])
        acc_gains.append(acc)
        cal_gains.append(-ece)
        local_acc = float(c.local_metrics["accuracy"])
        fairness_scores.append(1.0 - local_acc)
        vec = flatten_delta(c.delta)
        benign = max(0.0, cosine_similarity(vec, benign_reference_vec))
        temp_rel = temporal_reliability.get(c.client_id, 0.5)
        robust_scores.append(0.5 * benign + 0.5 * temp_rel)
        client_ids.append(c.client_id)

    acc_gains = _safe_norm(acc_gains)
    cal_gains = _safe_norm(cal_gains)
    fairness_scores = _safe_norm(fairness_scores)
    robust_scores = _safe_norm(robust_scores)

    objective_matrix = np.stack([acc_gains, cal_gains, fairness_scores, robust_scores], axis=1)
    ranks = pareto_ranks(objective_matrix)

    scalar = (
        wcfg["accuracy"] * acc_gains
        + wcfg["calibration"] * cal_gains
        + wcfg["fairness"] * fairness_strength * fairness_scores
        + wcfg["robustness"] * robust_scores
    )
    scalar += pareto_bonus * (1.0 / ranks)
    scalar -= trust_penalty * (1.0 - robust_scores)

    weights = np.exp(temperature * scalar)
    weights = project_simplex(weights)
    if entropy_reg > 0:
        uniform = np.ones_like(weights) / len(weights)
        weights = (1 - entropy_reg) * weights + entropy_reg * uniform
        weights = weights / weights.sum()

    diagnostics = {
        "client_ids": client_ids,
        "acc_scores": acc_gains.tolist(),
        "cal_scores": cal_gains.tolist(),
        "fairness_scores": fairness_scores.tolist(),
        "robust_scores": robust_scores.tolist(),
        "pareto_ranks": ranks.tolist(),
        "scalar_scores": scalar.tolist(),
    }
    return weights, diagnostics

def fedpareto_aggregate(client_results, cfg, temporal_reliability, benign_reference_vec):
    weights, diagnostics = compute_anchor_scores(
        client_results, cfg, temporal_reliability, benign_reference_vec
    )
    agg = average_state_dicts([c.delta for c in client_results], weights)
    diagnostics["method"] = "fedpareto"
    diagnostics["weights"] = weights.tolist()
    return agg, weights.tolist(), diagnostics
