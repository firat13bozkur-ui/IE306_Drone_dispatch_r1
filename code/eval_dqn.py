import argparse
import json
from pathlib import Path

import torch
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from obs_utils import flatten_obs
from dqn_agent import DQNAgent


class DQNPolicy:
    """
    Evaluation wrapper for a trained DQN model.

    It implements the required policy interface:
        act(obs) -> action
    """

    def __init__(self, model_path, hidden_dim=256, device="cpu"):
        self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.agent = DQNAgent(
            state_dim=checkpoint["state_dim"],
            n_actions=checkpoint["n_actions"],
            hidden_dim=hidden_dim,
            device=self.device,
        )

        self.agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.agent.q_network.eval()

    def act(self, obs):
        state = flatten_obs(obs)
        return self.agent.act(obs=obs, state=state, epsilon=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", default="logs/dqn_eval_seed0_1_2.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    env_cfg = Config.from_yaml(train_cfg["eval_config"])
    model_path = train_cfg["save_path"]
    hidden_dim = int(train_cfg["hidden_dim"])

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    policy = DQNPolicy(
        model_path=model_path,
        hidden_dim=hidden_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = evaluate(policy, env_cfg, seeds)

    print(json.dumps(results["mean"], indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()