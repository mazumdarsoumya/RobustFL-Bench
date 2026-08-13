from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from fedpareto.partition import dirichlet_partition, pathological_partition


_DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "gtsrb": ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
}


@dataclass
class FederatedDataBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    anchor_dataset: Dataset
    root_dataset: Dataset
    client_train_subsets: Dict[int, Dataset]
    client_eval_subsets: Dict[int, Dataset]
    num_classes: int


def _channels(name: str) -> int:
    return 1 if name in {"mnist", "fashionmnist"} else 3


def _num_classes(name: str) -> int:
    values = {
        "mnist": 10,
        "fashionmnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "gtsrb": 43,
    }
    if name not in values:
        raise ValueError(f"Unknown dataset: {name}")
    return values[name]


def get_transforms(name: str, normalize: bool = True, image_size: int | None = None, augment: bool = False, train: bool = True):
    from torchvision import transforms

    name = name.lower()
    if name not in _DATASET_STATS:
        raise ValueError(f"Unknown dataset: {name}")

    size = int(image_size or (28 if _channels(name) == 1 else 32))
    tfms = []
    if train and augment:
        if name in {"cifar10", "cifar100", "svhn"}:
            tfms.extend([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()])
        elif name == "gtsrb":
            tfms.extend([transforms.Resize((size, size)), transforms.RandomRotation(10)])
        else:
            tfms.append(transforms.RandomAffine(degrees=8, translate=(0.05, 0.05)))
    else:
        tfms.append(transforms.Resize((size, size)))

    tfms.append(transforms.ToTensor())
    if normalize:
        mean, std = _DATASET_STATS[name]
        tfms.append(transforms.Normalize(mean, std))
    return transforms.Compose(tfms)


def _build_dataset(name: str, root: Path, train: bool, transform):
    from torchvision import datasets as tv_datasets

    if name == "mnist":
        return tv_datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if name == "fashionmnist":
        return tv_datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    if name == "cifar10":
        return tv_datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if name == "cifar100":
        return tv_datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    if name == "svhn":
        return tv_datasets.SVHN(root=root, split="train" if train else "test", download=True, transform=transform)
    if name == "gtsrb":
        return tv_datasets.GTSRB(root=root, split="train" if train else "test", download=True, transform=transform)
    raise ValueError(f"Unknown dataset: {name}")


def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        values = dataset.targets
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "_samples"):
        return np.asarray([label for _, label in dataset._samples])
    raise AttributeError("Dataset has no accessible target labels")


def build_federated_data(cfg) -> FederatedDataBundle:
    name = cfg["dataset"]["name"].lower()
    root = Path(cfg["dataset"]["root"])
    normalize = bool(cfg["dataset"].get("normalize", True))
    augment = bool(cfg["dataset"].get("augment", False))
    image_size = cfg["dataset"].get("image_size")

    train_transform = get_transforms(name, normalize, image_size, augment, True)
    test_transform = get_transforms(name, normalize, image_size, False, False)
    train_dataset = _build_dataset(name, root, True, train_transform)
    test_dataset = _build_dataset(name, root, False, test_transform)

    targets = _targets(train_dataset)
    num_clients = int(cfg["partition"]["num_clients"])
    part_type = cfg["partition"]["type"].lower()

    if part_type == "dirichlet":
        client_indices = dirichlet_partition(targets, num_clients, float(cfg["partition"]["dirichlet_alpha"]), int(cfg["seed"]))
    elif part_type == "pathological":
        client_indices = pathological_partition(targets, num_clients, int(cfg["partition"]["pathological_classes_per_client"]), int(cfg["seed"]))
    else:
        raise ValueError(f"Unsupported partition type: {part_type}")

    rng = np.random.default_rng(int(cfg["seed"]))
    anchor_size = min(int(cfg["anchor"]["size"]), len(train_dataset))
    all_indices = np.arange(len(train_dataset))
    rng.shuffle(all_indices)
    anchor_idx = all_indices[:anchor_size].tolist()
    root_size = min(anchor_size, 256, max(0, len(train_dataset) - anchor_size))
    root_idx = all_indices[anchor_size:anchor_size + root_size].tolist()

    anchor_dataset = Subset(train_dataset, anchor_idx)
    root_dataset = Subset(train_dataset, root_idx)
    reserved = set(anchor_idx) | set(root_idx)
    client_train_subsets = {}
    client_eval_subsets = {}

    for cid, idxs in enumerate(client_indices):
        filtered = [int(i) for i in idxs if int(i) not in reserved]
        if not filtered:
            filtered = [int(i) for i in idxs]
        if len(filtered) == 1:
            filtered = filtered * 2
        cut = max(1, int(0.8 * len(filtered)))
        train_ids = filtered[:cut]
        eval_ids = filtered[cut:] or filtered[-1:]
        client_train_subsets[cid] = Subset(train_dataset, train_ids)
        client_eval_subsets[cid] = Subset(train_dataset, eval_ids)

    return FederatedDataBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        anchor_dataset=anchor_dataset,
        root_dataset=root_dataset,
        client_train_subsets=client_train_subsets,
        client_eval_subsets=client_eval_subsets,
        num_classes=_num_classes(name),
    )


def build_loaders(bundle: FederatedDataBundle, cfg):
    batch_size = int(cfg["federated"]["batch_size"])
    eval_batch_size = int(cfg["evaluation"]["batch_size"])
    workers = int(cfg.get("data_loader", {}).get("num_workers", 2))
    pin_memory = bool(cfg.get("data_loader", {}).get("pin_memory", True))

    client_train_loaders = {
        cid: DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
        for cid, ds in bundle.client_train_subsets.items()
    }
    client_eval_loaders = {
        cid: DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
        for cid, ds in bundle.client_eval_subsets.items()
    }
    anchor_loader = DataLoader(bundle.anchor_dataset, batch_size=int(cfg["anchor"]["batch_size"]), shuffle=False, num_workers=workers, pin_memory=pin_memory)
    root_loader = DataLoader(bundle.root_dataset, batch_size=int(cfg["anchor"]["batch_size"]), shuffle=True, num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(bundle.test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    return client_train_loaders, client_eval_loaders, anchor_loader, root_loader, test_loader
