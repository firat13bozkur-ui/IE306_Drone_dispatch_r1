import numpy as np

from drone_dispatch_env.world import Router


class RoutedDistanceCache:
    """
    No-fly-aware routed distance cache.

    It uses BFS distance fields, similar to the official baselines.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._grid_key = None
        self._router = None
        self._fields = {}

    def _ensure_router(self, grid):
        grid = np.asarray(grid)
        key = grid.tobytes()

        if key != self._grid_key:
            self._grid_key = key
            self._router = Router(grid, self.cfg.neighborhood)
            self._fields = {}

    def dist(self, grid, source, target):
        self._ensure_router(grid)

        source = (int(source[0]), int(source[1]))
        target = (int(target[0]), int(target[1]))

        if source not in self._fields:
            self._fields[source] = self._router.dist_field(source)

        field = self._fields[source]
        d = float(field[target[0], target[1]])

        if np.isfinite(d):
            return d

        # Defensive fallback.
        return abs(source[0] - target[0]) + abs(source[1] - target[1])


class PlanningPolicy:
    """
    Custom dispatch heuristic.

    Difference from greedy_nearest:
    - Uses higher charge threshold.
    - Scores assignment by total route length:
      drone -> pickup + pickup -> dropoff.
    """

    def __init__(
    	self,
    	cfg,
    	charge_threshold=0.50,
    	route_weight=1.0,
    	urgency_weight=0.25,
    	battery_penalty_weight=5.0,
    ):
    	self.cfg = cfg
    	self.charge_threshold = charge_threshold

    	self.route_weight = route_weight
    	self.urgency_weight = urgency_weight
    	self.battery_penalty_weight = battery_penalty_weight

    	self._dist = RoutedDistanceCache(cfg)

    def act(self, obs):
        c = self.cfg
        mask = np.asarray(obs["action_mask"], dtype=np.bool_)

        drones = obs["drones"]
        orders = obs["orders"]
        grid = obs["grid"]

        # 1. Charge low-battery idle drones first.
        low_battery_drones = []

        for d in range(c.n_drones):
            charge_action = c.charge_index(d)

            if mask[charge_action] and drones[d, 2] < self.charge_threshold:
                low_battery_drones.append((float(drones[d, 2]), d))

        if low_battery_drones:
            _, d = min(low_battery_drones, key=lambda x: x[0])
            return c.charge_index(d)

        # 2. Choose assignment with minimum total routed distance.
        best_action = None
        best_score = float("inf")

        for d in range(c.n_drones):
            if drones[d, 2] < self.charge_threshold:
                continue

            for slot in range(c.k_max):
                action = c.assign_index(d, slot)

                if not mask[action]:
                    continue

                drone_pos = (drones[d, 0], drones[d, 1])
                pickup = (orders[slot, 0], orders[slot, 1])
                dropoff = (orders[slot, 2], orders[slot, 3])

                drone_to_pickup = self._dist.dist(grid, drone_pos, pickup)
                pickup_to_dropoff = self._dist.dist(grid, pickup, dropoff)

                route_distance = drone_to_pickup + pickup_to_dropoff

                order_age = float(orders[slot, 4])

                battery_level = float(drones[d, 2])

                battery_penalty = max(
                    0.0,
                    route_distance * c.e_move - battery_level
                )

                score = (
                    self.route_weight * route_distance
                    - self.urgency_weight * order_age
                    + self.battery_penalty_weight * battery_penalty
                )

                if score < best_score:
                    best_score = score
                    best_action = action

                if score < best_score:
                    best_score = score
                    best_action = action

        if best_action is not None:
            return int(best_action)

        # 3. Nothing useful to do.
        return c.noop_index