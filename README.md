# RobustFL-Bench

**Robust federated learning benchmarking with FedPARETO, classic aggregators, heterogeneous clients, and poisoning attacks.**

RobustFL-Bench is the public research implementation used to study federated aggregation under statistical heterogeneity and adversarial clients. The repository includes the FedPARETO aggregation method, comparison baselines, experiment configurations, and the scripts needed to run and summarize experiments. It does not include datasets, checkpoints, detailed training logs, or restricted result files. In other words: the code is here, the giant pile of GPU exhaust is not.

## What is included

- Aggregation methods: FedAvg, coordinate-wise median, trimmed mean, Krum, FLTrust, and FedPARETO
- Datasets: MNIST, FashionMNIST, CIFAR-10, CIFAR-100, SVHN, and GTSRB
- Models: SimpleCNN, MLP, ResNet-18, MobileNetV3-Small, ShuffleNetV2, and EfficientNet-B0
- Attacks: sign-flip, Gaussian model poisoning, and BadNets-style backdoor poisoning
- Non-IID partitioning: Dirichlet and pathological class-skew partitions
- Metrics: global accuracy, expected calibration error, worst-client accuracy, attack success rate, aggregation-weight entropy, and runtime
- Utilities for multi-seed runs, ablations, plots, LaTeX tables, and mean/std aggregation

## Installation

```bash
conda env create -f environment.yml
conda activate fedpareto
pip install -e .
```

A regular virtual environment also works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run an experiment

```bash
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
```

Run the baseline set:

```bash
bash scripts/run_all_baselines.sh
```

Run multiple seeds:

```bash
python scripts/run_multi_seed.py --config configs/experiments/mnist_fedpareto.yaml --seeds 1 2 3
```

## Study configurations

Canonical hand-written experiment files live in `configs/experiments/`. Configurations retained from the study bundles are in `configs/study/retained/`. The attack-suite manifest is in `configs/study/attack_suite_manifest.json` and records the benchmark combinations used by the attack harness without publishing restricted result content.

## Repository layout

```text
configs/
  experiments/
  study/
scripts/
src/fedpareto/
  aggregation/
docs/
LICENSE
NOTICE
CITATION.cff
```

Run outputs are written under `runs/` by default and are intentionally ignored by Git. This is not Git being shy; it is the repository refusing to become a storage unit.

## Result utilities

Create plots from a completed run:

```bash
python scripts/make_plots.py --run_dir runs/<experiment_name>
```

## Data and artifact availability

Datasets are obtained from their original distribution sources through the dataset loaders and remain subject to their original licenses. They are not redistributed in this repository.

Detailed training logs, checkpoints, and key result files are not openly distributed because of institutional data-sharing restrictions. They may be accessed upon reasonable request to the corresponding author at `reachme@soumyamazumdar.com`.

## Reproducibility notes

Random seeds are controlled from YAML configuration files. Each run writes a configuration snapshot alongside generated metrics when local output is enabled. Dataset downloads, experiment outputs, model checkpoints, caches, and local IDE state are excluded from version control.

Federated learning distributes the training. Unfortunately, it does not distribute responsibility for checking the config file.

## Citation

Citation metadata is provided in `CITATION.cff`. The repository and software copyright are held by **Soumya Mazumdar**.

## License

Copyright © 2026 Soumya Mazumdar. Released under the MIT License. See `LICENSE` and `NOTICE`.