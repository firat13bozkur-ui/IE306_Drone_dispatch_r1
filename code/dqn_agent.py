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
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        self.q_network = QNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def act(self, obs, state, epsilon=0.0):
        """
        Select one valid action.

        obs:
            original environment observation dictionary.
            Used only for action_mask.

        state:
            flattened observation vector.
            Used as neural network input.

        epsilon:
            probability of random valid action.
        """

        action_mask = get_action_mask(obs)

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
                "model_state_dict": self.q_network.state_dict(),
            },
            path,
        )

    def load(self, path):
        """
        Load Q-network weights.
        """

        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.q_network.eval()