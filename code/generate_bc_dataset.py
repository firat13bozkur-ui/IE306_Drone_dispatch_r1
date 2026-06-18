import argparse
from pathlib import Path

import numpy as np
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv

from obs_utils import flatten_obs, get_action_mask
from improved_greedy_policy import ImprovedGreedyPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_bc = yaml.safe_load(f)

    env_cfg = Config.from_yaml(cfg_bc["eval_config"])
    env = DroneDispatchEnv(env_cfg)

    seed = int(cfg_bc.get("seed", 0))
    n_episodes = int(cfg_bc["dataset_episodes"])
    threshold = float(cfg_bc["teacher_charge_threshold"])
    dataset_path = Path(cfg_bc["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = ImprovedGreedyPolicy(
        cfg=env_cfg,
        charge_threshold=threshold,
    )

    states = []
    actions = []
    action_masks = []
    episode_returns = []
    delivered_counts = []
    dropped_counts = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)

        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            state = flatten_obs(obs)
            mask = get_action_mask(obs)
            action = teacher.act(obs)

            states.append(state)
            actions.append(action)
            action_masks.append(mask)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            steps += 1
            done = terminated or truncated

        episode_returns.append(ep_return)
        delivered_counts.append(env.stats.get("delivered", 0))
        dropped_counts.append(env.stats.get("dropped", 0))

        print(
            f"episode={ep} "
            f"return={ep_return:.2f} "
            f"steps={steps} "
            f"delivered={env.stats.get('delivered', 0)} "
            f"dropped={env.stats.get('dropped', 0)}"
        )

    states = np.stack(states).astype(np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    action_masks = np.stack(action_masks).astype(np.bool_)

    np.savez_compressed(
        dataset_path,
        states=states,
        actions=actions,
        action_masks=action_masks,
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        delivered_counts=np.asarray(delivered_counts, dtype=np.float32),
        dropped_counts=np.asarray(dropped_counts, dtype=np.float32),
        teacher_charge_threshold=np.asarray([threshold], dtype=np.float32),
    )

    print(f"Saved BC dataset to {dataset_path}")
    print(f"Number of samples: {len(actions)}")
    print(f"State dim: {states.shape[1]}")
    print(f"Number of actions: {action_masks.shape[1]}")
    print(f"Mean teacher return: {float(np.mean(episode_returns)):.2f}")
    print(f"Mean delivered: {float(np.mean(delivered_counts)):.2f}")
    print(f"Mean dropped: {float(np.mean(dropped_counts)):.2f}")


if __name__ == "__main__":
    main()