#!/usr/bin/env bash
set -e

python scripts/run_experiment.py --config configs/experiments/mnist_fedavg.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_trimmed_mean.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_krum.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fltrust.yaml
python scripts/run_experiment.py --config configs/experiments/mnist_fedpareto.yaml
