import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_plot(df, x, y, path, ylabel):
    plt.figure(figsize=(6, 4))
    plt.plot(df[x], df[y], marker='o')
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "metrics_round.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_plot(df, "round", "test_accuracy", plot_dir / "accuracy_vs_round.png", "Test Accuracy")
    save_plot(df, "round", "test_ece", plot_dir / "ece_vs_round.png", "ECE")
    save_plot(df, "round", "worst_client_accuracy", plot_dir / "worst_client_vs_round.png", "Worst Client Accuracy")
    if "attack_success_rate" in df.columns:
        save_plot(df, "round", "attack_success_rate", plot_dir / "asr_vs_round.png", "Attack Success Rate")
    if "weight_entropy" in df.columns:
        save_plot(df, "round", "weight_entropy", plot_dir / "weight_entropy_vs_round.png", "Weight Entropy")

    print(f"Saved plots to: {plot_dir}")

if __name__ == "__main__":
    main()
