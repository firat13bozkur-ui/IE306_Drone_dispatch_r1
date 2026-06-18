import random

import numpy as np
import torch
import torch.nn as nn

from obs_utils import get_action_mask, masked_argmax, sample_valid_action


class QNetwork(nn.Module):
    """
    Standard feed-forward Q-network for DQN.

    Input:
        flattened observation vector

    Output:
        Q-value for each discrete action
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


def build_battery_aware_action_mask(
    obs,
    charge_threshold=0.30,
    prefer_assignment=True,
):
    """
    Build a safer action mask for DQN.

    Logic:
    1. If an idle drone has critically low battery and charge is valid,
       force charge for those low-battery drones.
    2. Otherwise, prefer assignment actions when possible.
    3. When choosing assignments, remove low-battery drones from the assignment set.
    4. If no assignment is available, fall back to the original valid mask.
    """

    raw_mask = get_action_mask(obs)
    mask = np.array(raw_mask, copy=True).astype(np.bool_)

    drones = obs["drones"]
    orders = obs["orders"]

    n_drones = drones.shape[0]
    k_max = orders.shape[0]
    assignment_end = n_drones * k_max
    noop_index = len(mask) - 1

    # Charge action layout:
    # assignment actions: 0 ... n_drones*k_max - 1
    # charge actions: assignment_end ... assignment_end + n_drones - 1
    # noop: last action

    # 1. If any low-battery idle drone can charge, force charging first.
    charge_mask = np.zeros_like(mask, dtype=np.bool_)

    for d in range(n_drones):
        soc = float(drones[d, 2])
        alive = bool(drones[d, 3] > 0.5)
        charge_action = assignment_end + d

        if alive and soc < charge_threshold and mask[charge_action]:
            charge_mask[charge_action] = True

    if np.any(charge_mask):
        return charge_mask

    if not prefer_assignment:
        return mask

    # 2. Prefer assignments, but only for drones with enough battery.
    assignment_mask = np.zeros_like(mask, dtype=np.bool_)

    for d in range(n_drones):
        soc = float(drones[d, 2])
        alive = bool(drones[d, 3] > 0.5)

        if not alive:
            continue

        if soc < charge_threshold:
            continue

        start = d * k_max
        end = start + k_max
        assignment_mask[start:end] = mask[start:end]

    if np.any(assignment_mask[:assignment_end]):
        return assignment_mask

    # 3. If no safe assignment exists, fall back to valid actions.
    if np.any(mask):
        return mask

    # Defensive fallback.
    fallback = np.zeros_like(mask, dtype=np.bool_)
    fallback[noop_index] = True
    return fallback


class DQNAgent:
    """
    DQN agent with epsilon-greedy action selection and action masking.
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        hidden_dim=256,
        device="cpu",
        prefer_assignment=True,
        charge_threshold=0.30,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.prefer_assignment = prefer_assignment
        self.charge_threshold = charge_threshold

        self.q_network = QNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def decision_mask(self, obs):
        """
        Return the action mask actually used by the DQN policy.
        This is also used for next-state target calculation.
        """

        return build_battery_aware_action_mask(
            obs=obs,
            charge_threshold=self.charge_threshold,
            prefer_assignment=self.prefer_assignment,
        )

    def act(self, obs, state, epsilon=0.0):
        """
        Select one valid action using epsilon-greedy over the decision mask.
        """

        action_mask = self.decision_mask(obs)

        if random.random() < epsilon:
            return sample_valid_action(action_mask)

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.q_network(state_tensor)
            q_values = q_values.squeeze(0).cpu().numpy()

        return masked_argmax(q_values, action_mask)

    def save(self, path):
        """
        Save Q-network weights.
        """

        torch.save(
            {
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "prefer_assignment": self.prefer_assignment,
                "charge_threshold": self.charge_threshold,
                "model_state_dict": self.q_network.state_dict(),
            },
            path,
        )

    def load(self, path):
        """
        Load Q-network weights.
        """

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.prefer_assignment = checkpoint.get("prefer_assignment", True)
        self.charge_threshold = checkpoint.get("charge_threshold", 0.30)
        self.q_network.eval()