from collections import defaultdict
import numpy as np

def dirichlet_partition(labels, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, split_points)
        for cid, part in enumerate(splits):
            client_indices[cid].extend(part.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices

def pathological_partition(labels, num_clients, classes_per_client, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per_class_indices = {c: np.where(labels == c)[0].tolist() for c in classes}
    for c in classes:
        rng.shuffle(per_class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    assignments = [rng.choice(classes, size=classes_per_client, replace=False) for _ in range(num_clients)]

    buckets = defaultdict(list)
    for cid, chosen in enumerate(assignments):
        for cls in chosen:
            buckets[cls].append(cid)

    for cls, idxs in per_class_indices.items():
        owners = buckets[cls]
        if not owners:
            owners = list(range(num_clients))
        chunks = np.array_split(np.array(idxs), len(owners))
        for cid, chunk in zip(owners, chunks):
            client_indices[cid].extend(chunk.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices
