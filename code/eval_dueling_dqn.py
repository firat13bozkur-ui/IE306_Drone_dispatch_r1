import argparse
import json
from pathlib import Path

import torch
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from obs_utils import flatten_obs
from train_dueling_dqn import DuelingDQNAgent


class DuelingDQNPolicy:
    """
    Evaluation wrapper for trained Dueling DQN model.
    """

    def __init__(self, model_path, hidden_dim=256, device="cpu"):
        self.device = torch.device(device)

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.agent = DuelingDQNAgent(
            state_dim=checkpoint["state_dim"],
            n_actions=checkpoint["n_actions"],
            hidden_dim=hidden_dim,
            device=self.device,
            prefer_assignment=checkpoint.get("prefer_assignment", True),
            charge_threshold=checkpoint.get("charge_threshold", 0.0),
        )

        self.agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.agent.q_network.eval()

    def act(self, obs):
        state = flatten_obs(obs)
        return self.agent.act(obs=obs, state=state, epsilon=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dueling_dqn.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = Config.from_yaml(cfg["eval_config"])

    model_path = cfg["save_path"]
    hidden_dim = int(cfg["hidden_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    policy = DuelingDQNPolicy(
        model_path=model_path,
        hidden_dim=hidden_dim,
        device=device,
    )

    results = evaluate(policy, env_cfg, seeds)

    print(json.dumps(results["mean"], indent=2))

    output_path = Path(cfg.get("eval_output_path", "logs/dueling_dqn_eval.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()