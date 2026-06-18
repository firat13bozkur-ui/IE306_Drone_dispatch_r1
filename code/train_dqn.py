import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv

from obs_utils import flatten_obs
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent, QNetwork


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_epsilon(step, eps_start, eps_end, decay_steps):
    if step >= decay_steps:
        return eps_end

    frac = step / decay_steps
    return eps_start + frac * (eps_end - eps_start)


def train_step(
    agent,
    target_network,
    replay_buffer,
    optimizer,
    batch_size,
    gamma,
    device,
):
    states, actions, rewards, next_states, dones, next_action_masks = replay_buffer.sample(
        batch_size
    )

    states = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    next_action_masks = torch.as_tensor(
        next_action_masks, dtype=torch.bool, device=device
    )

    q_values = agent.q_network(states)
    chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states)

        # Invalid next actions should not be selected in the Bellman target.
        next_q_values = next_q_values.masked_fill(~next_action_masks, -1e9)

        max_next_q_values = next_q_values.max(dim=1).values
        target_q_values = rewards + gamma * (1.0 - dones) * max_next_q_values

    loss = nn.functional.smooth_l1_loss(chosen_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_train = yaml.safe_load(f)

    seed = int(cfg_train.get("seed", 0))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_config_path = cfg_train["eval_config"]
    env_cfg = Config.from_yaml(env_config_path)
    env = DroneDispatchEnv(env_cfg)

    obs, info = env.reset(seed=seed)
    state = flatten_obs(obs)
    state_dim = state.shape[0]
    n_actions = env.action_space.n

    total_episodes = int(args.episodes or cfg_train["total_episodes"])

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=int(cfg_train["hidden_dim"]),
        device=device,
        prefer_assignment=True,
        charge_threshold=0.30,
    )

    target_network = QNetwork(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=int(cfg_train["hidden_dim"]),
    ).to(device)

    target_network.load_state_dict(agent.q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(
        agent.q_network.parameters(),
        lr=float(cfg_train["learning_rate"]),
    )

    replay_buffer = ReplayBuffer(capacity=int(cfg_train["replay_size"]))

    gamma = float(cfg_train["gamma"])
    batch_size = int(cfg_train["batch_size"])
    min_replay_size = int(cfg_train["min_replay_size"])
    train_freq = int(cfg_train["train_freq"])
    target_update_freq = int(cfg_train["target_update_freq"])

    epsilon_start = float(cfg_train["epsilon_start"])
    epsilon_end = float(cfg_train["epsilon_end"])
    epsilon_decay_steps = int(cfg_train["epsilon_decay_steps"])

    save_path = Path(cfg_train["save_path"])
    log_path = Path(cfg_train["log_path"])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "episode",
                "episode_return",
                "episode_steps",
                "epsilon",
                "loss",
                "buffer_size",
                "delivered",
                "dropped",
                "energy",
                "depletion_events",
            ],
        )
        writer.writeheader()

        for episode in range(total_episodes):
            obs, info = env.reset(seed=seed + episode)
            state = flatten_obs(obs)

            done = False
            episode_return = 0.0
            episode_steps = 0
            losses = []

            while not done:
                epsilon = linear_epsilon(
                    global_step,
                    epsilon_start,
                    epsilon_end,
                    epsilon_decay_steps,
                )

                action = agent.act(obs=obs, state=state, epsilon=epsilon)

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = flatten_obs(next_obs)
                done = terminated or truncated

                # Important:
                # Use the same decision mask in training targets as the policy uses
                # during action selection.
                next_action_mask = agent.decision_mask(next_obs)

                replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_action_mask=next_action_mask,
                )

                state = next_state
                obs = next_obs

                episode_return += reward
                episode_steps += 1
                global_step += 1

                if (
                    len(replay_buffer) >= min_replay_size
                    and global_step % train_freq == 0
                ):
                    loss = train_step(
                        agent=agent,
                        target_network=target_network,
                        replay_buffer=replay_buffer,
                        optimizer=optimizer,
                        batch_size=batch_size,
                        gamma=gamma,
                        device=device,
                    )
                    losses.append(loss)

                if global_step % target_update_freq == 0:
                    target_network.load_state_dict(agent.q_network.state_dict())

            mean_loss = float(np.mean(losses)) if losses else 0.0

            writer.writerow(
                {
                    "episode": episode,
                    "episode_return": episode_return,
                    "episode_steps": episode_steps,
                    "epsilon": epsilon,
                    "loss": mean_loss,
                    "buffer_size": len(replay_buffer),
                    "delivered": env.stats.get("delivered", 0),
                    "dropped": env.stats.get("dropped", 0),
                    "energy": env.stats.get("energy", 0.0),
                    "depletion_events": env.stats.get("depletion_events", 0),
                }
            )
            log_file.flush()

            print(
                f"episode={episode} "
                f"return={episode_return:.2f} "
                f"steps={episode_steps} "
                f"epsilon={epsilon:.3f} "
                f"loss={mean_loss:.4f} "
                f"delivered={env.stats.get('delivered', 0)} "
                f"dropped={env.stats.get('dropped', 0)} "
                f"depletion_events={env.stats.get('depletion_events', 0)}"
            )

    agent.save(save_path)
    print(f"Saved model to {save_path}")
    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()