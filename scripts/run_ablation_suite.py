import argparse
import copy

from fedpareto.config import load_yaml
from fedpareto.experiments import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', required=True)
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    base_name = base["experiment_name"]

    variants = []

    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = f"{base_name}_ablation_no_calibration"
    cfg["fedpareto"]["objective_weights"]["calibration"] = 0.0
    variants.append(cfg)

    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = f"{base_name}_ablation_no_fairness"
    cfg["fedpareto"]["objective_weights"]["fairness"] = 0.0
    variants.append(cfg)

    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = f"{base_name}_ablation_no_robustness"
    cfg["fedpareto"]["objective_weights"]["robustness"] = 0.0
    variants.append(cfg)

    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = f"{base_name}_ablation_no_pareto_bonus"
    cfg["fedpareto"]["pareto_bonus"] = 0.0
    variants.append(cfg)

    for cfg in variants:
        runner = ExperimentRunner(cfg)
        runner.run()

if __name__ == "__main__":
    main()
