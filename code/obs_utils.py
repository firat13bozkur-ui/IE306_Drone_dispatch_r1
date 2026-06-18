import numpy as np


def flatten_obs(obs):
    """
    Convert the environment observation dictionary into a single flat float32 vector.

    Expected obs keys:
    - drones: [n_drones, drone_features]
    - orders: [k_max, order_features]
    - grid: [height, width]
    - time: [1]
    - action_mask: [n_actions]

    The action_mask is not included in the neural network input.
    It is used only during action selection.
    """

    drones = np.asarray(obs["drones"], dtype=np.float32).flatten()
    orders = np.asarray(obs["orders"], dtype=np.float32).flatten()
    grid = np.asarray(obs["grid"], dtype=np.float32).flatten()
    time = np.asarray(obs["time"], dtype=np.float32).flatten()

    state = np.concatenate([drones, orders, grid, time], axis=0)

    return state.astype(np.float32)


def get_action_mask(obs):
    """
    Return the action mask as a boolean numpy array.
    True means the action is valid.
    False means the action is invalid.
    """

    mask = np.asarray(obs["action_mask"], dtype=np.bool_)
    return mask


def masked_argmax(q_values, action_mask):
    """
    Select the valid action with the highest Q-value.

    q_values: numpy array with shape [n_actions]
    action_mask: boolean numpy array with shape [n_actions]
    """

    masked_q = np.array(q_values, copy=True)
    masked_q[~action_mask] = -1e9

    return int(np.argmax(masked_q))


def sample_valid_action(action_mask, rng=None):
    """
    Randomly sample one valid action from the action mask.
    Used for epsilon-greedy exploration.
    """

    if rng is None:
        rng = np.random.default_rng()

    valid_actions = np.flatnonzero(action_mask)

    if len(valid_actions) == 0:
        raise ValueError("No valid actions available in action_mask.")

    return int(rng.choice(valid_actions))