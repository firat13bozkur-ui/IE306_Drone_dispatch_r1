import argparse
import json
from pathlib import Path

import torch
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from obs_utils import flatten_obs, masked_argmax
from dqn_agent import build_battery_aware_action_mask
from train_bc_policy import BCNetwork


class BCPolicy:
    """
    Evaluation wrapper for behavioral cloning policy.

    The network predicts action logits, but during deployment we apply
    the same high-level safety/assignment mask used by the teacher:
    - charge low-battery drones first
    - otherwise prefer assignment actions
    """

    def __init__(
        self,
        model_path,
        charge_threshold=0.55,
        prefer_assignment=True,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.charge_threshold = charge_threshold
        self.prefer_assignment = prefer_assignment

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model = BCNetwork(
            state_dim=checkpoint["state_dim"],
            n_actions=checkpoint["n_actions"],
            hidden_dim=checkpoint.get("hidden_dim", 256),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, obs):
        state = flatten_obs(obs)

        action_mask = build_battery_aware_action_mask(
            obs=obs,
            charge_threshold=self.charge_threshold,
            prefer_assignment=self.prefer_assignment,
        )

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            logits = self.model(state_tensor)
            logits = logits.squeeze(0).cpu().numpy()

        return masked_argmax(logits, action_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = Config.from_yaml(cfg["eval_config"])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = BCPolicy(
        model_path=cfg["save_path"],
        charge_threshold=float(cfg.get("teacher_charge_threshold", 0.55)),
        prefer_assignment=True,
        device=device,
    )

    results = evaluate(policy, env_cfg, seeds)

    print(json.dumps(results["mean"], indent=2))

    output_path = Path(cfg["eval_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()