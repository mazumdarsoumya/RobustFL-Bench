# Project structure notes

## Main command
`python scripts/run_experiment.py --config <path-to-yaml>`

## What gets saved
Each run writes to `runs/<experiment_name>/`:

- `metrics_round.csv`
- `client_weights_round.json`
- `best_model.pt`
- `last_model.pt`
- `summary.json`
- `plots/` after `make_plots.py`

## Recommended experiment order
1. `mnist_fedavg.yaml`
2. `mnist_fedpareto.yaml`
3. `mnist_fedpareto_signflip.yaml`
4. `mnist_fedpareto_badnets.yaml`
5. `run_ablation_suite.py`
