from pathlib import Path
import yaml

def load_yaml(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
