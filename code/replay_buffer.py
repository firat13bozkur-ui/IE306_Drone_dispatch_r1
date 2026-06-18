import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Replay buffer for DQN.

    Stores transitions:
    state, action, reward, next_state, done, next_action_mask

    next_action_mask is needed because the DQN target should not select
    invalid actions in the next state.
    """

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")

        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done, next_action_mask):
        """
        Add one transition to the replay buffer.
        """

        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                np.asarray(next_action_mask, dtype=np.bool_),
            )
        )

    def sample(self, batch_size):
        """
        Sample a random batch from the replay buffer.

        Returns:
        states: [batch_size, state_dim]
        actions: [batch_size]
        rewards: [batch_size]
        next_states: [batch_size, state_dim]
        dones: [batch_size]
        next_action_masks: [batch_size, n_actions]
        """

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer size={len(self.buffer)}."
            )

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)

        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
            np.stack(next_action_masks).astype(np.bool_),
        )