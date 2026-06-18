import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


class BCNetwork(nn.Module):
    """
    Behavioral cloning policy network.

    Input:
        flattened observation vector

    Output:
        logits over discrete actions
    """

    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(n, batch_size, shuffle=True):
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def accuracy_from_logits(logits, actions, masks):
    masked_logits = logits.masked_fill(~masks, -1e9)
    preds = masked_logits.argmax(dim=1)
    return float((preds == actions).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    dataset = np.load(cfg["dataset_path"])

    states = dataset["states"].astype(np.float32)
    actions = dataset["actions"].astype(np.int64)
    action_masks = dataset["action_masks"].astype(np.bool_)

    n_samples = states.shape[0]
    state_dim = states.shape[1]
    n_actions = action_masks.shape[1]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split = int(0.8 * n_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BCNetwork(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
    )

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])

    log_path = Path(cfg["log_path"])
    save_path = Path(cfg["save_path"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()

        best_val_accuracy = -1.0

        for epoch in range(1, epochs + 1):
            model.train()

            train_losses = []
            train_accuracies = []

            for batch_idx in make_batches(len(train_idx), batch_size, shuffle=True):
                idx = train_idx[batch_idx]

                batch_states = torch.as_tensor(
                    states[idx],
                    dtype=torch.float32,
                    device=device,
                )
                batch_actions = torch.as_tensor(
                    actions[idx],
                    dtype=torch.long,
                    device=device,
                )
                batch_masks = torch.as_tensor(
                    action_masks[idx],
                    dtype=torch.bool,
                    device=device,
                )

                logits = model(batch_states)

                # Force invalid actions to have very low logits.
                masked_logits = logits.masked_fill(~batch_masks, -1e9)

                loss = nn.functional.cross_entropy(masked_logits, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                train_losses.append(float(loss.item()))
                train_accuracies.append(
                    accuracy_from_logits(logits, batch_actions, batch_masks)
                )

            model.eval()

            val_losses = []
            val_accuracies = []

            with torch.no_grad():
                for batch_idx in make_batches(len(val_idx), batch_size, shuffle=False):
                    idx = val_idx[batch_idx]

                    batch_states = torch.as_tensor(
                        states[idx],
                        dtype=torch.float32,
                        device=device,
                    )
                    batch_actions = torch.as_tensor(
                        actions[idx],
                        dtype=torch.long,
                        device=device,
                    )
                    batch_masks = torch.as_tensor(
                        action_masks[idx],
                        dtype=torch.bool,
                        device=device,
                    )

                    logits = model(batch_states)
                    masked_logits = logits.masked_fill(~batch_masks, -1e9)

                    loss = nn.functional.cross_entropy(masked_logits, batch_actions)

                    val_losses.append(float(loss.item()))
                    val_accuracies.append(
                        accuracy_from_logits(logits, batch_actions, batch_masks)
                    )

            train_loss = float(np.mean(train_losses))
            train_acc = float(np.mean(train_accuracies))
            val_loss = float(np.mean(val_losses))
            val_acc = float(np.mean(val_accuracies))

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )
            log_file.flush()

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc

                torch.save(
                    {
                        "state_dim": state_dim,
                        "n_actions": n_actions,
                        "hidden_dim": int(cfg["hidden_dim"]),
                        "model_state_dict": model.state_dict(),
                    },
                    save_path,
                )

    print(f"Saved best BC policy to {save_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()