import argparse
import json
from pathlib import Path

import torch
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from obs_utils import flatten_obs, get_action_mask, masked_argmax
from dqn_agent import build_battery_aware_action_mask
from train_bc_policy import BCNetwork


class BCPolicy:
    """
    Evaluation wrapper for behavioral cloning policy.

    mask_mode options:
    - raw: use only the simulator's original valid action mask
    - teacher: use teacher-style action mask
    """

    def __init__(
        self,
        model_path,
        mask_mode="teacher",
        charge_threshold=0.56,
        prefer_assignment=True,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.mask_mode = mask_mode
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

    def _action_mask(self, obs):
        if self.mask_mode == "raw":
            return get_action_mask(obs)

        if self.mask_mode == "teacher":
            return build_battery_aware_action_mask(
                obs=obs,
                charge_threshold=self.charge_threshold,
                prefer_assignment=self.prefer_assignment,
            )

        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def act(self, obs):
        state = flatten_obs(obs)
        action_mask = self._action_mask(obs)

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
    parser.add_argument("--mask-mode", choices=["raw", "teacher"], default="teacher")
    parser.add_argument("--charge-threshold", type=float, default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = Config.from_yaml(cfg["eval_config"])

    seeds = [
        int(s.strip())
        for s in args.seeds.split(",")
        if s.strip()
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.charge_threshold is not None:
        charge_threshold = args.charge_threshold
    else:
        charge_threshold = float(cfg.get("teacher_charge_threshold", 0.55))

    policy = BCPolicy(
        model_path=cfg["save_path"],
        mask_mode=args.mask_mode,
        charge_threshold=charge_threshold,
        prefer_assignment=True,
        device=device,
    )

    results = evaluate(policy, env_cfg, seeds)

    print(json.dumps(results["mean"], indent=2))

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = Path(cfg["eval_output_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()