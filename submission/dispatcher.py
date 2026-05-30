"""
v5 planner: reservation-based, DEADLOCK-FREE BY CONSTRUCTION, multi-stop.

Approach (the proven Flatland-winner family): plan every train as a timed path with
space-time reservations (cell intervals + undirected-edge intervals). If each train
follows its reservation, plans are pairwise conflict-free => no collision and no
deadlock, ever. Trains are scheduled in deadline order; a train waits (in place) until
its next cell's reservation window is free.

Fixes vs v3 (whose reservations were correct but over-serialized -> trains never
departed -> -5x cancellation -> worse score):
  * NO "reserve target forever". remove_agents_at_target=True, so a finished train
    frees its cell. v3 parked targets for the whole horizon, blocking every other train
    routed through a shared hub. This was the main cause of non-departure.
  * Intermediate stops dwell just ONE step (enough to register STOPPED => "served"),
    instead of holding a shared hub cell for a long earliest_departure window. We accept
    the small early-departure penalty (0.5x) to avoid blocking the hub for everyone.
  * Routes thread all timetable waypoints (topology.multistop_route).

Execution follows the timed plan; a train never drifts on DO_NOTHING. (Malfunction
re-planning is a later addition; the no-malfunction levels exercise the core first.)
"""
import numpy as np
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from submission.topology import Topology

DO, FWD, STOP = RailEnvActions.DO_NOTHING, RailEnvActions.MOVE_FORWARD, RailEnvActions.STOP_MOVING
_MOVES = [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]
DWELL = 1                      # stop steps at an intermediate waypoint (>=1 => "served")


def _kof(a):
    return max(1, int(round(1 / max(a.speed_counter.speed, 1e-9))))


class Reservation:
    def __init__(self):
        self.cell = {}
        self.edge = {}

    def cell_free(self, c, t0, t1):
        for (a, b) in self.cell.get(c, ()):
            if not (t1 < a or t0 > b):
                return False
        return True

    def edge_free(self, u, v, t0, t1):
        for (a, b) in self.edge.get(frozenset((u, v)), ()):
            if not (t1 < a or t0 > b):
                return False
        return True

    def add_cell(self, c, t0, t1):
        self.cell.setdefault(c, []).append((t0, t1))

    def add_edge(self, u, v, t0, t1):
        self.edge.setdefault(frozenset((u, v)), []).append((t0, t1))


class V5Planner:
    def __init__(self, max_wait=800, multistop=False, K=4, use_kpaths=False, abandon_unplaceable=None,
                 replan_interval=0):
        # replan_interval>0: re-dispatch every N steps from trains' CURRENT positions (online /
        # "local dispatcher" style) so the schedule adapts to real congestion instead of trains
        # hopelessly following a stale timeline they've fallen behind (the high-density cascade).
        self.replan_interval = replan_interval
        self.last_replan = -10 ** 9
        # use_kpaths=False: route with our own fast topology descent (~1ms/agent) instead of
        # flatland's get_k_shortest_paths (~3.5s/agent -> ~30min at 532 agents, blows budget).
        self.use_kpaths = use_kpaths
        # abandon_unplaceable: trains that can't complete stay off-map instead of parking
        # mid-map. Helps at high density (avoids jam cascades, ~2x completion -> better vs the
        # 25% abort cliff); hurts at low density (needlessly cancels a train that could have
        # parked & progressed). None => decide by agent count in _plan_all.
        self._abandon_cfg = abandon_unplaceable
        self.abandon_unplaceable = False
        # multistop=False: route DIRECT to target (short paths => high completion).
        # Intermediate stops cost only -50 each, negligible vs the -max_steps floor for
        # not completing, so direct routing maximizes the dominant term (target arrivals).
        # K: number of alternative shortest paths tried per train; the planner commits the
        # one that schedules with least delay given prior reservations -> spreads traffic
        # off congested corridors (the main reason v2's k-paths beat single-route v5).
        self.multistop = multistop
        self.K = K
        # buf: extra following-spacing steps per cell. Tested 0 vs 1 -> no difference at high
        # density (the wall is the prioritized planner, not packing), so keep 0 (max throughput).
        self.buf = 0
        self.topo = None
        self.plans = {}        # h -> [(cell, dir, enter_t, depart_t), ...] or None
        self.ptr = {}
        self.kpaths = {}
        self.ready = False
        self.max_wait = max_wait

    @staticmethod
    def _start(a):
        """Current (pos, dir) if on-map, else initial -- so planning works mid-episode."""
        if a.position is not None:
            return tuple(a.position), int(a.direction)
        return tuple(a.initial_position), int(a.initial_direction)

    def _order(self, env):
        ag = env.agents
        topo = self.topo

        def slack(h):
            a = ag[h]
            la = a.latest_arrival or 10 ** 9
            sp, sd = self._start(a)
            d = topo.distance(sp, sd, a.target)
            d = d if np.isfinite(d) else 10 ** 9
            return la - d
        return sorted(range(env.get_num_agents()), key=slack)

    def _candidate_routes(self, env, h):
        a = env.agents[h]
        if self.multistop:
            route, stops = self.topo.multistop_route(a)
            return [(route, set(stops[:-1]) if stops else set())] if route else []
        sp, sd = self._start(a)
        routes = []
        if self.use_kpaths:                           # flatland Yen's: ~3.5s/agent (slow!)
            try:
                paths = get_k_shortest_paths(env, sp, sd, a.target, k=self.K)
            except Exception:
                paths = None
            routes = [[(tuple(wp.position), int(wp.direction)) for wp in p] for p in paths] if paths else []
        if not routes:                                # fast topology K-routing (~ms/agent)
            routes = self.topo.k_routes(sp, sd, {tuple(a.target)}, K=self.K)
        return [(r, set()) for r in routes]

    def _build(self, route, stop_set, t0, k, res, horizon):
        """Build a timed conflict-free plan for `route` departing at t0 (reads res, no
        mutation). Returns (plan, adds, reached, last_cell, last_enter)."""
        target = route[-1][0]
        plan = [(route[0][0], route[0][1], t0, t0 + k)]
        adds = [("C", route[0][0], t0, t0 + k - 1)]
        prev_cell, prev_enter, prev_depart = route[0][0], t0, t0 + k
        for i in range(1, len(route)):
            cell, d = route[i]
            enter_t = prev_depart
            stuck = False
            while not (res.cell_free(cell, enter_t, enter_t + k - 1)
                       and res.cell_free(prev_cell, prev_enter, enter_t - 1)
                       and res.edge_free(prev_cell, cell, prev_enter, enter_t + k - 1)):
                enter_t += 1
                if enter_t - prev_depart > self.max_wait or enter_t > horizon:
                    stuck = True
                    break
            if stuck:
                break
            depart_t = enter_t + k + (DWELL if i in stop_set else 0)
            adds.append(("C", prev_cell, prev_enter, enter_t - 1))
            adds.append(("E", prev_cell, cell, prev_enter, enter_t + k - 1))
            adds.append(("C", cell, enter_t, depart_t - 1 + self.buf))   # +buf = following spacing
            plan.append((cell, d, enter_t, depart_t))
            prev_cell, prev_enter, prev_depart = cell, enter_t, depart_t
        reached = plan[-1][0] == target and plan[-1][2] <= horizon
        return plan, adds, reached, prev_cell, prev_enter

    def _schedule(self, env, h, res, t_now=0):
        a = env.agents[h]
        cands = self._candidate_routes(env, h)
        if not cands:
            self.plans[h] = None
            return
        k = _kof(a)
        horizon = env._max_episode_steps
        # earliest time this train can occupy its start cell: now if already on-map, else
        # the later of now and its earliest_departure (still off-map at the depot).
        on_map = a.position is not None
        base = t_now if on_map else max(t_now, int(a.waypoints_earliest_departure[0] or 0))

        # Try each route, and for each a few DEPARTURE times. Prefer a complete clean run
        # (depart later if needed) over parking mid-map: a train waiting at the depot blocks
        # nobody, while a parked train jams the corridor and cascades stalls onto others.
        # On-map trains can't delay (they're already moving), so only off-map trains stagger.
        delays = [0, 30, 60, 120, 240, 480, 960] if not on_map else [0]
        best = None  # (reached, -arrival/-progress, plan, adds, parks, last_cell, last_enter)
        for route, stop_set in cands:
            for dl in delays:
                t0 = base + dl
                if t0 > horizon:
                    break
                plan, adds, reached, lc, le = self._build(route, stop_set, t0, k, res, horizon)
                score = (reached, -plan[-1][2] if reached else len(plan) - 10 ** 6)
                cand = (score, plan, adds, not reached, lc, le)
                if best is None or score > best[0]:
                    best = cand
                if reached:
                    break          # earliest departure that completes this route
            if best is not None and best[0][0]:
                break              # a route completed; take it

        _, plan, adds, parks, last_cell, last_enter = best
        if parks and self.abandon_unplaceable and not on_map:
            # Can't complete and still at the depot: keep it OFF-MAP (no plan -> never departs).
            # At high density a train parked mid-map jams a corridor and cascades stalls onto
            # trains that COULD complete; an undeparted train blocks nobody. (On-map trains
            # can't be abandoned -- they're already out there -- so they keep their best plan.)
            self.plans[h] = None
            return
        if parks and last_enter <= horizon:
            adds = adds + [("C", last_cell, last_enter, horizon + k)]
        for it in adds:
            if it[0] == "C":
                res.add_cell(it[1], it[2], it[3])
            else:
                res.add_edge(it[1], it[2], it[3], it[4])
        self.plans[h] = plan
        self.ptr[h] = 0

    def _plan_all(self, env, t_now=0):
        if self.topo is None:
            self.topo = Topology(env)
        # density-adaptive: abandon unplaceable trains off-map only when crowded
        n = env.get_num_agents()
        self.abandon_unplaceable = (n > 60) if self._abandon_cfg is None else self._abandon_cfg
        res = Reservation()
        # Pre-reserve every on-map train's CURRENT cell for this instant so re-planned routes
        # steer around where trains physically are now (a train's own start cell is never
        # free-checked in _build, so this adds no self-conflict).
        for a in env.agents:
            if a.position is not None and a.state != TrainState.DONE:
                res.add_cell(tuple(a.position), t_now, t_now)
        self.plans = {}
        self.ptr = {}
        for h in self._order(env):
            if env.agents[h].state == TrainState.DONE:
                self.plans[h] = None
                continue
            self._schedule(env, h, res, t_now=t_now)
        self.ready = True
        self.last_replan = t_now

    def _action_to(self, env, cell, d, nxt_cell):
        for act in _MOVES:
            r = env.rail.apply_action_independent(RailEnvActions.from_value(act), (cell, d))
            if r is None:
                continue
            (np_, nd_), _ = r
            if tuple(np_) == nxt_cell:
                return act
        return FWD

    def act_many(self, handles, envs):
        env = envs[0]
        ag = env.agents
        t = env._elapsed_steps
        if not self.ready:
            self._plan_all(env, t_now=t)
        elif self.replan_interval > 0 and t - self.last_replan >= self.replan_interval \
                and not all(a.state == TrainState.DONE for a in ag):
            self._plan_all(env, t_now=t)            # online re-dispatch from current positions

        cur_cell = {}
        for h, a in enumerate(ag):
            if a.position is not None and a.state != TrainState.DONE:
                cur_cell[h] = tuple(a.position)
        occupied = {c: h for h, c in cur_cell.items()}

        # each train wants its next planned cell once its scheduled move time is reached
        desired = {}
        for h in handles:
            a = ag[h]
            plan = self.plans.get(h)
            if a.state == TrainState.DONE or not plan:
                continue
            if getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0:
                continue
            if a.position is None:
                if t >= plan[0][2]:                      # earliest_departure reached
                    desired[h] = plan[0][0]
            else:
                cp = cur_cell[h]
                i = self.ptr.get(h, 0)
                while i + 1 < len(plan) and plan[i][0] != cp:
                    i += 1
                self.ptr[h] = i
                if i + 1 < len(plan) and t >= plan[i][3] - _kof(a):   # scheduled to leave by now
                    desired[h] = plan[i + 1][0]

        # chain resolution: grant a move when the next cell is free OR its occupant is also
        # being granted to move out this step -> nose-to-tail chains advance together (the
        # absolute-time rigidity that stalled chains was the high-density failure mode).
        granted = {}
        claimed = {}
        changed = True
        while changed:
            changed = False
            for h in handles:
                if h in granted or h not in desired:
                    continue
                nxt = desired[h]
                if nxt in claimed:
                    continue
                occ = occupied.get(nxt)
                if occ is not None and occ != h and occ not in granted:
                    continue                              # blocked by a train not (yet) vacating
                granted[h] = nxt
                claimed[nxt] = h
                changed = True

        actions = {}
        for h in handles:
            a = ag[h]
            if a.state == TrainState.DONE:
                actions[h] = DO
                continue
            if getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0:
                actions[h] = STOP
                continue
            if h in granted:
                if a.position is None:
                    actions[h] = FWD
                else:
                    d = int(a.direction) if a.direction is not None else int(a.initial_direction)
                    actions[h] = self._action_to(env, cur_cell[h], d, granted[h])
            else:
                actions[h] = STOP if a.position is not None else DO
        return actions


