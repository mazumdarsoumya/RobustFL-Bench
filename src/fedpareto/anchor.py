import torch

@torch.no_grad()
def summarize_on_anchor(model, anchor_loader, device):
    model.eval()
    logits_list = []
    probs_list = []
    labels_list = []
    for x, y in anchor_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        logits_list.append(logits.cpu())
        probs_list.append(probs.cpu())
        labels_list.append(y.cpu())
    return {
        "logits": torch.cat(logits_list, dim=0),
        "probs": torch.cat(probs_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
    }
