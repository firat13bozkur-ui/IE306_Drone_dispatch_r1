import numpy as np


def flatten_obs(obs):
    """
    Convert the environment observation dictionary into a normalized flat vector.

    We do not include action_mask in the neural network input.
    The mask is used only during action selection.
    """

    drones = np.asarray(obs["drones"], dtype=np.float32).copy()
    orders = np.asarray(obs["orders"], dtype=np.float32).copy()
    grid = np.asarray(obs["grid"], dtype=np.float32).copy()
    time = np.asarray(obs["time"], dtype=np.float32).copy()

    height, width = grid.shape
    coord_scale_x = max(float(height - 1), 1.0)
    coord_scale_y = max(float(width - 1), 1.0)

    # Drone features:
    # columns 0,1: x,y
    # column 2: state of charge already in [0,1]
    # column 3: alive flag already in {0,1}
    # remaining columns: one-hot/status/binary features
    drones[:, 0] = drones[:, 0] / coord_scale_x
    drones[:, 1] = drones[:, 1] / coord_scale_y

    # Order features:
    # columns 0,1: origin x,y
    # columns 2,3: destination x,y
    # column 4: order age
    orders[:, 0] = orders[:, 0] / coord_scale_x
    orders[:, 1] = orders[:, 1] / coord_scale_y
    orders[:, 2] = orders[:, 2] / coord_scale_x
    orders[:, 3] = orders[:, 3] / coord_scale_y

    # Standard SLA is 60 in eval_standard.yaml.
    # Clip prevents extreme values from dominating the neural network input.
    orders[:, 4] = np.clip(orders[:, 4] / 60.0, 0.0, 2.0)

    # Grid cell codes are 0,1,2,3.
    grid = grid / 3.0

    state = np.concatenate(
        [
            drones.flatten(),
            orders.flatten(),
            grid.flatten(),
            time.flatten(),
        ],
        axis=0,
    )

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