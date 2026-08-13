from typing import Dict

import torch


def apply_model_attack(delta: Dict[str, torch.Tensor], attack_cfg):
    name = attack_cfg["name"].lower()
    if name == "none":
        return delta
    if name == "sign_flip":
        scale = float(attack_cfg.get("scale", 3.0))
        return {k: (-scale * v if torch.is_floating_point(v) or torch.is_complex(v) else v.detach().clone()) for k, v in delta.items()}
    if name == "gaussian":
        std = float(attack_cfg.get("std", 0.5))
        out = {}
        for key, value in delta.items():
            if torch.is_floating_point(value) or torch.is_complex(value):
                out[key] = value + torch.randn_like(value) * std
            else:
                out[key] = value.detach().clone()
        return out
    return delta


def maybe_poison_batch(x, y, attack_cfg, malicious: bool):
    if not malicious or attack_cfg["name"].lower() != "badnets":
        return x, y
    poison_fraction = float(attack_cfg.get("poison_fraction", 0.30))
    target_label = int(attack_cfg.get("target_label", 0))
    trigger_size = max(1, int(attack_cfg.get("trigger_size", 4)))
    patch_value = float(attack_cfg.get("patch_value", 1.0))
    poison_count = min(x.size(0), max(1, int(poison_fraction * x.size(0))))
    x = x.clone()
    y = y.clone()
    x[:poison_count, :, -trigger_size:, -trigger_size:] = patch_value
    y[:poison_count] = target_label
    return x, y
