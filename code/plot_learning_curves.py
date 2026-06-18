import argparse
from pathlib import Path

import matplotlib

# Use non-interactive backend.
# This prevents Tkinter/init.tcl errors on Windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(values, window=20):
    return values.rolling(window=window, min_periods=1).mean()


def plot_rl_curve(log_path, output_path, title, metric="episode_return", window=20):
    df = pd.read_csv(log_path)

    if "episode" not in df.columns:
        raise ValueError(f"{log_path} does not contain an episode column.")

    if metric not in df.columns:
        raise ValueError(f"{log_path} does not contain metric column: {metric}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = df["episode"]
    y = df[metric]
    y_smooth = moving_average(y, window=window)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, alpha=0.35, label="Raw")
    plt.plot(x, y_smooth, linewidth=2, label=f"Moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def plot_bc_loss(log_path, output_path, title):
    df = pd.read_csv(log_path)

    required = ["epoch", "train_loss", "val_loss"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{log_path} does not contain column: {col}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def plot_bc_accuracy(log_path, output_path, title):
    df = pd.read_csv(log_path)

    required = ["epoch", "train_accuracy", "val_accuracy"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{log_path} does not contain column: {col}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(df["epoch"], df["train_accuracy"], label="Train accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    rl_logs = [
        {
            "log": "logs/dqn_seed0.csv",
            "output": "figures/dqn_episode_return.png",
            "title": "DQN Learning Curve",
        },
        {
            "log": "logs/double_dqn_seed0.csv",
            "output": "figures/double_dqn_episode_return.png",
            "title": "Double DQN Learning Curve",
        },
        {
            "log": "logs/dueling_dqn_seed0.csv",
            "output": "figures/dueling_dqn_episode_return.png",
            "title": "Dueling DQN Learning Curve",
        },
    ]

    for item in rl_logs:
        log_path = Path(item["log"])

        if log_path.exists():
            plot_rl_curve(
                log_path=log_path,
                output_path=item["output"],
                title=item["title"],
                metric="episode_return",
                window=args.window,
            )
        else:
            print(f"Skipped missing file: {log_path}")

    bc_logs = [
        {
            "log": "logs/bc_training_t055.csv",
            "loss_output": "figures/bc_seed0_loss.png",
            "acc_output": "figures/bc_seed0_accuracy.png",
            "loss_title": "BC Seed 0 Loss Curve",
            "acc_title": "BC Seed 0 Accuracy Curve",
        },
        {
            "log": "logs/bc_training_t055_seed1.csv",
            "loss_output": "figures/bc_seed1_loss.png",
            "acc_output": "figures/bc_seed1_accuracy.png",
            "loss_title": "BC Seed 1 Loss Curve",
            "acc_title": "BC Seed 1 Accuracy Curve",
        },
        {
            "log": "logs/bc_training_t055_seed2.csv",
            "loss_output": "figures/bc_seed2_loss.png",
            "acc_output": "figures/bc_seed2_accuracy.png",
            "loss_title": "BC Seed 2 Loss Curve",
            "acc_title": "BC Seed 2 Accuracy Curve",
        },
    ]

    for item in bc_logs:
        log_path = Path(item["log"])

        if log_path.exists():
            plot_bc_loss(
                log_path=log_path,
                output_path=item["loss_output"],
                title=item["loss_title"],
            )

            plot_bc_accuracy(
                log_path=log_path,
                output_path=item["acc_output"],
                title=item["acc_title"],
            )
        else:
            print(f"Skipped missing file: {log_path}")


if __name__ == "__main__":
    main()