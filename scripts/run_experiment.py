import argparse
from fedpareto.config import load_yaml
from fedpareto.experiments import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    runner = ExperimentRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
