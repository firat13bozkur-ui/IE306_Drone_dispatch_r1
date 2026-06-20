import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_utils import flatten_obs, get_action_mask, masked_argmax


class ActorCriticNetwork(nn.Module):
    """
    Shared MLP with a policy head and a value head.
    Policy head chooses the action.
    Value head estimates V(s).
    """

    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class MaskedActorCriticAgent:
    """
    Actor-Critic agent for the discrete drone dispatch action space.

    Important:
    Invalid actions are always masked before action selection.
    """

    def __init__(self, state_dim, n_actions, hidden_dim=256, device="cpu"):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_dim=int(hidden_dim),
        ).to(self.device)

    def masked_logits(self, logits, action_mask):
        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=logits.device,
        )

        if mask_tensor.dim() == 1 and logits.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return logits.masked_fill(~mask_tensor, -1e9)

    def act(self, obs, deterministic=False):
        state = flatten_obs(obs)
        action_mask = get_action_mask(obs)

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            logits, _ = self.network(state_tensor)
            logits = logits.squeeze(0)
            masked = self.masked_logits(logits, action_mask)

            if deterministic:
                return masked_argmax(masked.cpu().numpy(), action_mask)

            dist = Categorical(logits=masked)
            return int(dist.sample().item())

    def save(self, path):
        torch.save(
            {
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "model_state_dict": self.network.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
