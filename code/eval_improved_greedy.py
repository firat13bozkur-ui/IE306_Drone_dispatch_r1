import argparse
import json
from pathlib import Path

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from improved_greedy_policy import ImprovedGreedyPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--charge-threshold", type=float, default=0.50)
    parser.add_argument(
        "--output",
        default="logs/improved_greedy_eval_seed0_1_2.json",
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    policy = ImprovedGreedyPolicy(
        cfg=cfg,
        charge_threshold=args.charge_threshold,
    )

    results = evaluate(policy, cfg, seeds)

    print(json.dumps(results["mean"], indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()