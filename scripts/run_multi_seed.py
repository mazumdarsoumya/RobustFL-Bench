import argparse
import copy
from pathlib import Path

import yaml

from fedpareto.config import load_yaml
from fedpareto.experiments import ExperimentRunner


def build_cfg_for_seed(base_cfg: dict, seed: int, suffix_template: str = "_seed{seed}") -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)

    base_name = cfg.get("experiment_name", "experiment")
    if "_seed" in base_name:
        base_name = base_name.split("_seed")[0]
    cfg["experiment_name"] = f"{base_name}{suffix_template.format(seed=seed)}"
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run the same experiment config for multiple seeds."
    )
    parser.add_argument("--config", required=True, type=str, help="Base YAML config path")
    parser.add_argument(
        "--seeds",
        required=True,
        nargs="+",
        type=int,
        help="List of seeds, e.g. --seeds 1 2 3",
    )
    parser.add_argument(
        "--write_derived_configs",
        action="store_true",
        help="If set, save derived per-seed YAML configs under runs/<exp>/derived_configs/",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)

    for seed in args.seeds:
        cfg = build_cfg_for_seed(base_cfg, seed)
        exp_name = cfg["experiment_name"]
        print(f"\\n=== Running seed {seed}: {exp_name} ===")

        if args.write_derived_configs:
            root_dir = Path(cfg["output"]["root_dir"]) / exp_name / "derived_configs"
            root_dir.mkdir(parents=True, exist_ok=True)
            with open(root_dir / f"{exp_name}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

        runner = ExperimentRunner(cfg)
        runner.run()


if __name__ == "__main__":
    main()
