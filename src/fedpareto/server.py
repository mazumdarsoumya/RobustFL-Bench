from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from fedpareto.aggregation import (
    fedavg_aggregate,
    coord_median_aggregate,
    trimmed_mean_aggregate,
    krum_aggregate,
    fltrust_aggregate,
    fedpareto_aggregate,
)
from fedpareto.utils import (
    add_delta_to_state,
    clone_state_dict,
    flatten_delta,
    entropy_from_weights,
)

@dataclass
class ServerStepOutput:
    global_state: Dict[str, torch.Tensor]
    weights: List[float]
    diagnostics: Dict
    weight_entropy: float

class FedServer:
    def __init__(self, model_fn, device, cfg):
        self.model_fn = model_fn
        self.device = device
        self.cfg = cfg
        self.temporal_reliability = defaultdict(lambda: 0.5)
        self.global_model = self.model_fn().to(self.device)
        self.best_metric = -1.0

    def get_state(self):
        return clone_state_dict(self.global_model.state_dict())

    def apply_aggregated_delta(self, delta):
        new_state = add_delta_to_state(self.get_state(), delta, scale=1.0)
        self.global_model.load_state_dict(new_state, strict=True)

    def train_root_update(self, root_loader):
        model = self.model_fn().to(self.device)
        model.load_state_dict(self.get_state(), strict=True)
        model.train()
        opt = torch.optim.SGD(
            model.parameters(),
            lr=self.cfg["federated"]["lr"],
            momentum=self.cfg["federated"]["momentum"],
            weight_decay=self.cfg["federated"]["weight_decay"],
        )
        epochs = int(self.cfg["anchor"].get("server_root_epochs", 1))
        base_state = clone_state_dict(model.state_dict())
        for _ in range(epochs):
            for x, y in root_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
        new_state = clone_state_dict(model.state_dict())
        return {k: new_state[k] - base_state[k] for k in new_state.keys()}

    def update_temporal_reliability(self, client_results, weights, diagnostics):
        momentum = float(self.cfg["fedpareto"]["reliability_momentum"])
        robust_scores = diagnostics.get("robust_scores", [0.5] * len(client_results))
        for c, w, rs in zip(client_results, weights, robust_scores):
            prev = self.temporal_reliability[c.client_id]
            signal = 0.5 * float(w) + 0.5 * float(rs)
            self.temporal_reliability[c.client_id] = momentum * prev + (1 - momentum) * signal

    def step(self, client_results, root_loader=None):
        method = self.cfg["method"]["name"].lower()
        benign_reference_vec = None
        diagnostics = {}

        if method == "fedavg":
            delta, weights, diagnostics = fedavg_aggregate(client_results)
        elif method == "coord_median":
            delta, weights, diagnostics = coord_median_aggregate(client_results)
        elif method == "trimmed_mean":
            trim_ratio = float(self.cfg["method"].get("trim_ratio", 0.2))
            delta, weights, diagnostics = trimmed_mean_aggregate(client_results, trim_ratio=trim_ratio)
        elif method == "krum":
            byz = int(round(self.cfg["attack"].get("malicious_fraction", 0.0) * len(client_results)))
            delta, weights, diagnostics = krum_aggregate(client_results, byzantine_count=byz)
        elif method == "fltrust":
            if root_loader is None:
                raise ValueError("FLTrust requires a root_loader.")
            root_delta = self.train_root_update(root_loader)
            delta, weights, diagnostics = fltrust_aggregate(client_results, root_delta)
        elif method == "fedpareto":
            root_delta = self.train_root_update(root_loader) if root_loader is not None else client_results[0].delta
            benign_reference_vec = flatten_delta(root_delta)
            delta, weights, diagnostics = fedpareto_aggregate(
                client_results=client_results,
                cfg=self.cfg,
                temporal_reliability=self.temporal_reliability,
                benign_reference_vec=benign_reference_vec,
            )
            self.update_temporal_reliability(client_results, weights, diagnostics)
        else:
            raise ValueError(f"Unsupported method: {method}")

        self.apply_aggregated_delta(delta)
        ent = entropy_from_weights(weights)
        return ServerStepOutput(
            global_state=self.get_state(),
            weights=weights,
            diagnostics=diagnostics,
            weight_entropy=ent,
        )
