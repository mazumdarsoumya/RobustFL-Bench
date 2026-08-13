import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from fedpareto.client import LocalClient
from fedpareto.datasets import build_federated_data, build_loaders
from fedpareto.metrics import evaluate_model, worst_client_accuracy, attack_success_rate
from fedpareto.models import get_model
from fedpareto.server import FedServer
from fedpareto.utils import get_device, save_json, set_seed

class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg["seed"])
        self.device = get_device(cfg.get("device", "cuda"))
        self.bundle = build_federated_data(cfg)
        (
            self.client_train_loaders,
            self.client_eval_loaders,
            self.anchor_loader,
            self.root_loader,
            self.test_loader,
        ) = build_loaders(self.bundle, cfg)

        self.model_fn = lambda: get_model(
            cfg["model"]["name"],
            cfg["dataset"]["name"],
            self.bundle.num_classes,
            cfg["dataset"].get("image_size"),
        )
        self.server = FedServer(self.model_fn, self.device, cfg)
        self.run_dir = Path(cfg["output"]["root_dir"]) / cfg["experiment_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_rows = []
        self.weight_rows = []
        self.best_accuracy = -1.0

        import yaml
        with open(self.run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def select_clients(self, rnd):
        num_clients = self.cfg["partition"]["num_clients"]
        m = self.cfg["federated"]["clients_per_round"]
        rng = np.random.default_rng(self.cfg["seed"] + rnd)
        return sorted(rng.choice(num_clients, size=m, replace=False).tolist())

    def malicious_clients(self):
        frac = float(self.cfg["attack"].get("malicious_fraction", 0.0))
        num_clients = self.cfg["partition"]["num_clients"]
        count = int(round(frac * num_clients))
        rng = np.random.default_rng(self.cfg["seed"] + 999)
        return set(rng.choice(num_clients, size=count, replace=False).tolist()) if count > 0 else set()

    def run(self):
        malicious_set = self.malicious_clients()
        total_start = time.time()

        for rnd in trange(1, self.cfg["federated"]["rounds"] + 1, desc="Rounds"):
            round_start = time.time()
            selected = self.select_clients(rnd)
            global_state = self.server.get_state()

            client_results = []
            for cid in selected:
                client = LocalClient(
                    client_id=cid,
                    model_fn=self.model_fn,
                    train_loader=self.client_train_loaders[cid],
                    eval_loader=self.client_eval_loaders[cid],
                    device=self.device,
                    cfg=self.cfg,
                    malicious=(cid in malicious_set),
                )
                result = client.train(global_state=global_state, anchor_loader=self.anchor_loader)
                client_results.append(result)

            step_output = self.server.step(client_results, root_loader=self.root_loader)
            test_out = evaluate_model(
                self.server.global_model,
                self.test_loader,
                self.device,
                ece_bins=self.cfg["evaluation"]["ece_bins"],
            )
            worst_acc, per_client_metrics = worst_client_accuracy(
                self.server.global_model,
                self.client_eval_loaders,
                self.device,
                ece_bins=self.cfg["evaluation"]["ece_bins"],
            )

            asr = 0.0
            if self.cfg["attack"]["name"] == "badnets":
                asr = attack_success_rate(
                    self.server.global_model,
                    self.test_loader,
                    self.device,
                    target_label=int(self.cfg["attack"].get("target_label", 0)),
                    patch_value=float(self.cfg["attack"].get("patch_value", 1.0)),
                    trigger_size=int(self.cfg["attack"].get("trigger_size", 4)),
                )

            row = {
                "round": rnd,
                "test_loss": test_out["loss"],
                "test_accuracy": test_out["accuracy"],
                "test_ece": test_out["ece"],
                "worst_client_accuracy": worst_acc,
                "attack_success_rate": asr,
                "weight_entropy": step_output.weight_entropy,
                "round_runtime_sec": time.time() - round_start,
                "num_selected_clients": len(selected),
            }
            self.metrics_rows.append(row)

            weight_row = {
                "round": rnd,
                "selected_client_ids": selected,
                "weights": step_output.weights,
                "diagnostics": step_output.diagnostics,
            }
            self.weight_rows.append(weight_row)

            if test_out["accuracy"] > self.best_accuracy:
                self.best_accuracy = test_out["accuracy"]
                torch.save(self.server.global_model.state_dict(), self.run_dir / "best_model.pt")

        total_runtime = time.time() - total_start
        torch.save(self.server.global_model.state_dict(), self.run_dir / "last_model.pt")
        df = pd.DataFrame(self.metrics_rows)
        df.to_csv(self.run_dir / "metrics_round.csv", index=False)
        save_json(self.weight_rows, self.run_dir / "client_weights_round.json")

        summary = {
            "experiment_name": self.cfg["experiment_name"],
            "method": self.cfg["method"]["name"],
            "attack": self.cfg["attack"]["name"],
            "final_test_accuracy": float(df["test_accuracy"].iloc[-1]),
            "best_test_accuracy": float(df["test_accuracy"].max()),
            "final_test_ece": float(df["test_ece"].iloc[-1]),
            "final_worst_client_accuracy": float(df["worst_client_accuracy"].iloc[-1]),
            "final_attack_success_rate": float(df["attack_success_rate"].iloc[-1]),
            "total_runtime_sec": total_runtime,
        }
        save_json(summary, self.run_dir / "summary.json")
        print(f"Run finished. Outputs saved to: {self.run_dir}")
