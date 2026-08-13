import torch
import torch.nn.functional as F


@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for index in range(n_bins):
        low, high = boundaries[index], boundaries[index + 1]
        in_bin = (confidences > low) & (confidences <= high)
        proportion = in_bin.float().mean()
        if proportion.item() > 0:
            accuracy = accuracies[in_bin].float().mean()
            confidence = confidences[in_bin].mean()
            ece += torch.abs(confidence - accuracy) * proportion
    return float(ece.item())


@torch.no_grad()
def evaluate_model(model, loader, device, ece_bins=15):
    model.eval()
    total = 0
    correct = 0
    losses = []
    probabilities = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        probs = torch.softmax(logits, dim=1)
        total += y.size(0)
        correct += probs.argmax(dim=1).eq(y).sum().item()
        losses.append(loss.item() * y.size(0))
        probabilities.append(probs)
        labels.append(y)
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "ece": 0.0}
    probs_all = torch.cat(probabilities, dim=0)
    labels_all = torch.cat(labels, dim=0)
    return {
        "loss": sum(losses) / total,
        "accuracy": correct / total,
        "ece": expected_calibration_error(probs_all, labels_all, n_bins=ece_bins),
    }


@torch.no_grad()
def worst_client_accuracy(model, client_eval_loaders, device, ece_bins=15):
    accuracies = []
    metrics = {}
    for client_id, loader in client_eval_loaders.items():
        result = evaluate_model(model, loader, device, ece_bins=ece_bins)
        accuracies.append(result["accuracy"])
        metrics[client_id] = result
    return (min(accuracies) if accuracies else 0.0), metrics


@torch.no_grad()
def attack_success_rate(model, loader, device, target_label=0, patch_value=1.0, trigger_size=4):
    model.eval()
    total = 0
    success = 0
    trigger_size = max(1, int(trigger_size))
    for x, y in loader:
        x = x.clone()
        x[:, :, -trigger_size:, -trigger_size:] = float(patch_value)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        eligible = y.ne(int(target_label))
        total += eligible.sum().item()
        success += (preds.eq(int(target_label)) & eligible).sum().item()
    return success / max(1, total)
