import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.distributions import Categorical

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.evaluate import evaluate

from actor_critic_agent import MaskedActorCriticAgent
from obs_utils import flatten_obs, get_action_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(config_path, seed):
    env_cfg = Config.from_yaml(config_path)
    env = DroneDispatchEnv(env_cfg)
    obs, info = env.reset(seed=seed)
    return env, obs


def evaluate_policy(agent, config_path, seeds):
    env_cfg = Config.from_yaml(config_path)
    eval_results = evaluate(agent, env_cfg, seeds=seeds)

    summary = dict(eval_results["mean"])
    summary["seeds"] = list(seeds)
    summary["per_seed"] = eval_results["per_seed"]
    return summary


def train_actor_critic(
    config_path,
    total_episodes,
    seed,
    gamma,
    lr,
    hidden_dim,
    value_coef,
    entropy_coef,
    device,
    log_path,
    weight_path,
):
    set_seed(seed)

    env, obs = make_env(config_path, seed)

    state_dim = len(flatten_obs(obs))
    n_actions = int(env.action_space.n)

    agent = MaskedActorCriticAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        device=device,
    )

    optimizer = torch.optim.Adam(agent.network.parameters(), lr=lr)

    episode_logs = []

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + episode)

        log_probs = []
        values = []
        rewards = []
        entropies = []

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            state = flatten_obs(obs)
            action_mask = get_action_mask(obs)

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0)

            logits, value = agent.network(state_tensor)
            logits = logits.squeeze(0)
            value = value.squeeze(0)

            masked_logits = agent.masked_logits(logits, action_mask)
            dist = Categorical(logits=masked_logits)

            action = int(dist.sample().item())
            log_prob = dist.log_prob(torch.as_tensor(action, device=agent.device))
            entropy = dist.entropy()

            next_obs, reward, terminated, truncated, info = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)

            episode_return += float(reward)
            obs = next_obs

        returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.append(running_return)
        returns.reverse()

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=agent.device,
        )

        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        advantages = returns_tensor - values_tensor.detach()

        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        entropy_loss = -entropies_tensor.mean()

        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
        optimizer.step()

        metrics = info.get("metrics", {})
        log_row = {
            "episode": episode,
            "episode_return": float(episode_return),
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropies_tensor.mean().item()),
            "cost_per_order": float(metrics.get("cost_per_order", np.nan)),
            "n_delivered": float(metrics.get("n_delivered", np.nan)),
            "n_dropped": float(metrics.get("n_dropped", np.nan)),
        }
        episode_logs.append(log_row)

        if episode == 1 or episode % 10 == 0:
            print(
                f"episode={episode:04d} "
                f"return={episode_return:.2f} "
                f"cost_per_order={log_row['cost_per_order']:.3f} "
                f"delivered={log_row['n_delivered']:.0f}"
            )

    env.close()

    os.makedirs(Path(log_path).parent, exist_ok=True)
    os.makedirs(Path(weight_path).parent, exist_ok=True)

    agent.save(weight_path)

    eval_summary = evaluate_policy(agent, config_path, seeds=[0, 1, 2])

    output = {
        "method": "masked_actor_critic",
        "config_path": config_path,
        "train_seed": seed,
        "total_episodes": total_episodes,
        "hyperparameters": {
            "gamma": gamma,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
        },
        "final_evaluation": eval_summary,
        "training_log": episode_logs,
        "weight_path": weight_path,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("Saved weights:", weight_path)
    print("Saved log:", log_path)
    print("Final evaluation:")
    print(json.dumps(eval_summary, indent=2))

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log-path",
        default="logs/role_b_actor_critic_summary.json",
    )
    parser.add_argument(
        "--weight-path",
        default="weights/role_b_actor_critic.pt",
    )
    args = parser.parse_args()

    train_actor_critic(
        config_path=args.config,
        total_episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=args.device,
        log_path=args.log_path,
        weight_path=args.weight_path,
    )


if __name__ == "__main__":
    main()
