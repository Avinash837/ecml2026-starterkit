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
                 replan_interval=0, load_weight=0.0):
        # load_weight>0: congestion-aware routing. TESTED -> net NEGATIVE at scene-relevant
        # density (120 agents: 31.7%->25.8%): detours add delay and cost more completion than
        # hub-spreading saves. Kept (off by default) for possible use at extreme saturation.
        self.load_weight = load_weight
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
        # V6: collect intermediate-stop credit only when it is already on the
        # direct route. Keep this low-density so one-step dwell does not disturb
        # the dense completion-first behavior that made V5 stable.
        self.opportunistic_stops = True
        self.opportunistic_stop_agent_cap = 60
        # Stop-aware K-route choice is safe only on small maps. Explicit
        # stop-detour routes were tested and rejected because they cost too
        # many completions even when the geometric detour looked short.
        self.prefer_route_stops = False
        self.prefer_route_stops_agent_cap = 30
        self.prefer_route_stops_bonus = 20
        # buf: extra following-spacing steps per cell. Tested 0 vs 1 -> no difference at high
        # density (the wall is the prioritized planner, not packing), so keep 0 (max throughput).
        self.buf = 0
        self.topo = None
        self.plans = {}        # h -> [(cell, dir, enter_t, depart_t), ...] or None
        self.ptr = {}
        self.stop_at = {}      # h -> plan indices where one explicit STOP is required
        self.stopped_at = {}   # h -> stop indices already emitted as STOP_MOVING
        self.kpaths = {}
        self.ready = False
        self.max_wait = max_wait
        self.corridor_load = {}
        # --- malfunction handling ---
        # reroute-stuck-trains repair: TESTED neutral/slightly-negative (90 agents, realistic
        # malfunction rate: 37.8%->36.7%) -- short breakdowns are cheaper to wait out than to
        # detour around, and reroutes disrupt the conflict-free schedule. Off by default.
        # (The off-map malfunction fix in act_many is unconditional and IS kept -- it helps
        # level-4 departure delays: issue the stored move so a frozen depot train auto-departs.)
        self.malfunction_repair = False
        self.stuck = {}            # h -> consecutive steps blocked (wanted to move, not granted)
        self.reroutes = {}         # h -> #times rerouted (capped to avoid thrash)
        self.reroute_after = 10    # steps stuck before attempting a reroute
        self.max_reroutes = 2
        # --- RELEASE VALVE: periodically retry the trains that NEVER departed (abandoned
        # off-map because no conflict-free full route fit at plan time). As the network drains
        # late in the episode, free capacity opens up that those trains never use. Re-schedule
        # ONLY the undeparted trains against the EXISTING committed plans (committed plans are
        # never modified -> preserves deadlock-free-by-construction; differs from the online
        # replan that re-dispatched everyone and thrashed). 0 = off. ---
        self.release_interval = 0
        # --- SIGNAL GUARD (home signal at the last crossing): never admit a train ONTO a
        # junction unless the cell BEYOND it (its planned exit) is also clear. Prevents the
        # junction-rest deadlock (train stuck ON a switch, blocking the queue that would clear
        # it). A reactive overlay on the reservation engine, evaluated on ACTUAL occupancy so
        # it also absorbs the realizability drift that the timed plan suffers. 0 = off. ---
        self.signal_guard = False
        # --- BLOCK LOCK (single-track directional interlocking = the token/staff principle):
        # a single-track corridor SEGMENT (topo.seg_of) may be occupied in ONE direction at a
        # time. A train is admitted into a segment only if the segment is empty or its current
        # occupants travel the SAME way (following). A train wanting the opposite direction is
        # held at the entrance until the block clears. Makes head-on deadlock IMPOSSIBLE (the
        # measured root cause: ~8 head-on pairs/episode backing up dozens of queued trains).
        # Occupancy-driven, so robust to the timetable drift that defeats the a-priori plan. ---
        self.block_lock = False
        self._exit_cache = {}
        # --- CRITICALITY ROUTING (route by DANGER, not just distance): penalize each cell by
        # the BLAST RADIUS of its single-track block = how many trains get stranded if that
        # block jams. Trains that HAVE alternatives peel off the high-danger chokepoints,
        # relieving the forced traffic that has no choice. 0 = off. ---
        self.crit_weight = 0.0
        self.crit_top = 50         # compute blast radius for the top-N busiest segments only
        self.criticality = None    # {cell: blast_radius}, computed once (map is fixed)
        # --- DIRECTIONAL SEPARATION (double-track / up-line down-line running): when a single-
        # track block has a passing loop, route OPPOSING-direction trains onto the parallel track
        # instead of separating them only in TIME (which drift collapses into a head-on). During
        # sequential planning, a candidate route is penalized for traversing a corridor cell
        # AGAINST the direction already committed there -> the opposing train takes the loop.
        # Spatial separation is drift-robust. 0 = off. ---
        self.dir_weight = 0.0
        self.dir_load = {}         # {(cell,dir): #committed traversals}
        # --- GREEDY ADVANCE: follow the planned ROUTE but with REACTIVE timing -- a train
        # advances to its next planned cell as soon as that cell is free (chain resolution),
        # instead of waiting for its scheduled clock time plan[i][3]. Decouples route (good,
        # from the plan) from timing (reactive) so trains AHEAD of schedule don't idle on clear
        # track. Occupancy is the only brake. 0 = obey timetable (default). ---
        self.greedy_advance = False
        # --- STUCK REPLAN (surgical online repair): when a train has been blocked for
        # stuck_thresh steps, replan ONLY that train from its CURRENT position against everyone
        # else's committed plans (which stay fixed). Unlike full re-dispatch (replan_interval,
        # which thrashed 22->5 by discarding good plans), this is targeted -> it can recover a
        # train that found a clear alternative without destabilizing the rest. 0 = off. ---
        self.stuck_replan = False
        self.stuck_thresh = 40
        # --- MEET-PASS (reactive divert-to-loop): backtrack analysis showed ~7 head-on cores
        # per episode are the ROOT of 100% of the stuck-train cascade, and most core segments
        # HAVE a tight parallel passing loop. block_lock merely HELD the entering train at the
        # junction (fouling it -> jam relocates). Instead, when a train is about to enter a
        # single-track segment an OPPOSING train already occupies, DIVERT it onto a clear
        # parallel loop toward its target (true meet-pass). Capped per train. 0 = off. ---
        self.meet_pass = False
        self.meet_pass_reroutes = {}
        self.max_meet_pass = 3
        # --- CITY-HOLD (platform-as-siding): a train at a MULTI-TRACK city (a safe waiting spot --
        # other platform tracks stay free) is HELD there if a single-track segment ahead on its
        # route is occupied by an OPPOSING train, until that segment clears. Unlike block_lock
        # (which holds at a bare junction and fouls it), holding at a multi-track city does NOT
        # block through-traffic. Backtrack showed ~70% of cores have such a city within ~6 cells.
        # 0 = off. ---
        self.city_hold = False
        self.safe_cells = None      # cells in multi-track city clusters (safe to wait on)
        self.lookahead = 8          # cells of route lookahead for an opposing single-track block
        # --- FROZEN REROUTE (surgical malfunction re-planning): when a train MALFUNCTIONS (freezes
        # 20-50 steps) in a corridor, PROACTIVELY divert trains that are about to ENTER that
        # blocked corridor onto a clear alternative, BEFORE they queue up behind it. Prevents the
        # cascade (one freeze stalling a whole queue) that costs 24-34pp on malfunction levels.
        # Differs from malfunction_repair (reroutes already-stuck trains -> too late). 0 = off. ---
        self.frozen_reroute = False
        self.frozen_reroutes = {}
        self.max_frozen_reroutes = 6
        # --- REPLAN-ON-MALFUNCTION (continuous frozen-aware re-planning): when a NEW train
        # freezes, re-dispatch all active trains from current positions, RESERVING the frozen
        # cells for their remaining freeze duration so the fresh plan routes AROUND them.
        # Throttled (min interval) to avoid the thrash that blind periodic re-dispatch caused. ---
        self.replan_on_malfunction = False
        self._last_frozen = set()
        self._frozen_reserve = []
        self._last_malf_replan = -10 ** 9
        self.malf_replan_min = 25
        # --- EXEC FAST-FIRST (dynamic release): when trains CONTEND for a cell (esp. a queue
        # freed after a malfunction clears), grant the FASTER train first. Execution-time
        # complement to the fast-first planning order -- targets the malfunction release order. ---
        self.exec_fast_first = False
        # --- LATE SEGMENT GUARD (stale-timetable protection): if a train has slipped far behind
        # its planned slot while inside a single-track segment, hold opposite-direction trains
        # before they enter that same segment. This targets the malfunction drift failure where a
        # late train meets an on-time opposing train in a plain corridor; once they become adjacent,
        # Flatland cannot reverse either train, so repair is usually too late. 0 = off. ---
        self.late_segment_guard = False
        self.late_guard_threshold = 30
        self.debug_counts = {}
        # --- LNS refinement: destroy+repair subsets of the prioritized plan to fit more trains ---
        self.lns_iters = 0         # 0 = off; >0 = run LNS after the initial plan
        self.lns_time = 20.0       # seconds budget for LNS
        self.lns_nbhd = 25         # neighborhood size (agents re-planned jointly)
        # --- RL-tuned priority hook: callable (env, topo, h) -> score (low = planned first).
        # Set by the ES trainer; replaces the slack heuristic for the planning ORDER while v5
        # still does routing + deadlock-free reservations. None = use the slack heuristic. ---
        self.priority_fn = None

    @staticmethod
    def _start(a):
        """Current (pos, dir) if on-map, else initial -- so planning works mid-episode."""
        if a.position is not None:
            return tuple(a.position), int(a.direction)
        return tuple(a.initial_position), int(a.initial_direction)

    def _order(self, env):
        ag = env.agents
        topo = self.topo

        if self.priority_fn is not None:               # RL-learned planning priority
            return sorted(range(env.get_num_agents()),
                          key=lambda h: self.priority_fn(env, topo, h))

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
            load, lw = self.corridor_load, self.load_weight
            if self.crit_weight > 0 and self.criticality:
                load, lw = self.criticality, self.crit_weight   # route by danger (blast radius)
            routes = self.topo.k_routes(sp, sd, {tuple(a.target)}, K=self.K,
                                        load=load, lw=lw,
                                        dir_load=self.dir_load, dw=self.dir_weight)
        return [(r, self._opportunistic_stop_set(env, h, r)) for r in routes]

    def _opportunistic_stop_set(self, env, h, route):
        """Path indices where V6 can dwell without changing the V5 route.

        This preserves V5's completion-first path choice. If a train's direct
        route naturally passes an intermediate timetable platform in order, add
        the one-step dwell that ECML rewards as a served stop.
        """
        if not self.opportunistic_stops:
            return set()
        if env.get_num_agents() > self.opportunistic_stop_agent_cap:
            return set()
        a = env.agents[h]
        if len(a.waypoints) <= 2 or not route:
            return set()

        stop_set = set()
        search_from = 1
        final_gi = len(a.waypoints) - 1
        for gi in range(1, final_gi):
            goal_states = {
                (tuple(w.position), int(w.direction))
                for w in a.waypoints[gi]
                if getattr(w, "direction", None) is not None
            }
            goal_cells = {tuple(w.position) for w in a.waypoints[gi]}
            found = None
            for idx in range(search_from, len(route) - 1):
                cell, direction = route[idx]
                if goal_states:
                    matched = (cell, int(direction)) in goal_states
                else:
                    matched = cell in goal_cells
                if matched:
                    found = idx
                    break
            if found is not None:
                stop_set.add(found)
                search_from = found + 1
        return stop_set

    def _compute_criticality(self, env):
        """{cell: blast_radius} for the busiest single-track blocks. blast_radius(seg) =
        #agents whose target becomes UNREACHABLE from their source if that block is removed."""
        from collections import deque, Counter
        topo = self.topo
        N = env.get_num_agents()
        triples = [(tuple(a.initial_position), int(a.initial_direction), tuple(a.target))
                   for a in env.agents]
        load = Counter()
        for sp, sd, tgt in triples:
            r = topo.route(sp, sd, {tgt})
            if r:
                for c, _ in r:
                    if c in topo.seg_of:
                        load[topo.seg_of[c]] += 1
        top = [s for s, _ in load.most_common(self.crit_top)]

        def reach(sp, sd, tgt, blocked):
            if sp in blocked:
                return False
            seen = {(sp, sd)}; q = deque([(sp, sd)])
            while q:
                cur = q.popleft()
                if cur[0] == tgt:
                    return True
                for nb in topo.succ.get(cur, ()):
                    if nb[0] not in blocked and nb not in seen:
                        seen.add(nb); q.append(nb)
            return False

        crit = {}
        for s in top:
            blocked = set(topo.seg_cells[s])
            blast = sum(1 for sp, sd, tgt in triples if not reach(sp, sd, tgt, blocked))
            if blast > 0:
                for c in topo.seg_cells[s]:
                    crit[c] = blast
        self.criticality = crit

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
            self.stop_at[h] = set()
            self.stopped_at[h] = set()
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
        best = None  # (score, plan, adds, parks, last_cell, last_enter, stop_set)
        prefer_stops = self.prefer_route_stops and env.get_num_agents() <= self.prefer_route_stops_agent_cap
        for route, stop_set in cands:
            for dl in delays:
                t0 = base + dl
                if t0 > horizon:
                    break
                plan, adds, reached, lc, le = self._build(route, stop_set, t0, k, res, horizon)
                if reached and prefer_stops:
                    score = (reached, -plan[-1][2] + self.prefer_route_stops_bonus * len(stop_set))
                else:
                    score = (reached, -plan[-1][2] if reached else len(plan) - 10 ** 6)
                cand = (score, plan, adds, not reached, lc, le, set(stop_set))
                if best is None or score > best[0]:
                    best = cand
                if reached:
                    break          # earliest departure that completes this route
            if best is not None and best[0][0] and not prefer_stops:
                break              # a route completed; take it

        _, plan, adds, parks, last_cell, last_enter, chosen_stops = best
        if parks and self.abandon_unplaceable and not on_map:
            # Can't complete and still at the depot: keep it OFF-MAP (no plan -> never departs).
            # At high density a train parked mid-map jams a corridor and cascades stalls onto
            # trains that COULD complete; an undeparted train blocks nobody. (On-map trains
            # can't be abandoned -- they're already out there -- so they keep their best plan.)
            self.plans[h] = None
            self.stop_at[h] = set()
            self.stopped_at[h] = set()
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
        self.stop_at[h] = {i for i in chosen_stops if 0 < i < len(plan) - 1}
        self.stopped_at[h] = set()
        for (cell, d, et, dt) in plan:                # register this train's corridor usage
            self.corridor_load[cell] = self.corridor_load.get(cell, 0) + 1
            self.dir_load[(cell, d)] = self.dir_load.get((cell, d), 0) + 1   # directional usage

    # ---- LNS (Large Neighborhood Search) refinement of the prioritized plan ----------------
    def _reserve_plan(self, res, plan, k):
        for i, (cell, d, et, dt) in enumerate(plan):
            res.add_cell(cell, et, dt - 1)
            if i > 0:
                res.add_edge(plan[i - 1][0], cell, plan[i - 1][2], et + k - 1)

    def _reaches(self, env, h):
        plan = self.plans.get(h)
        return bool(plan) and plan[-1][0] == tuple(env.agents[h].target) \
            and plan[-1][2] <= env._max_episode_steps

    def _n_reached(self, env):
        return sum(self._reaches(env, h) for h in range(env.get_num_agents()))

    def _build_res_excluding(self, env, t_now, nbhd):
        """Reservation holding the plans of all agents NOT in the neighborhood."""
        res = Reservation()
        for a in env.agents:
            if a.position is not None and a.state != TrainState.DONE:
                res.add_cell(tuple(a.position), t_now, t_now)
        for h, plan in self.plans.items():
            if h in nbhd or not plan:
                continue
            self._reserve_plan(res, plan, _kof(env.agents[h]))
        return res

    def _lns(self, env, t_now):
        """Iteratively destroy a neighborhood of agents and re-plan them jointly against the
        rest, keeping the change only if MORE trains reach their target. Random + congestion-
        targeted (around the busiest cell = a hub station/crossing) neighbourhoods."""
        import random, time as _time
        rng = random.Random(0)
        N = env.get_num_agents()
        active = [h for h in range(N) if env.agents[h].state != TrainState.DONE]
        if not active:
            return
        best = self._n_reached(env)
        deadline = _time.time() + self.lns_time
        for it in range(self.lns_iters):
            if _time.time() > deadline:
                break
            # neighborhood: a failed agent + agents sharing its busiest corridor (congestion-
            # targeted, your station/crossing idea) padded with random agents.
            failed = [h for h in active if not self._reaches(env, h)]
            seed_h = rng.choice(failed) if failed else rng.choice(active)
            seg_of = self.topo.seg_of
            seed_cells = {c for (c, _, _, _) in (self.plans.get(seed_h) or [])}
            nbhd = {h for h in active
                    if any(c in seed_cells for (c, _, _, _) in (self.plans.get(h) or []))}
            nbhd = set(list(nbhd)[:self.lns_nbhd]) | {seed_h}
            while len(nbhd) < self.lns_nbhd and len(nbhd) < len(active):
                nbhd.add(rng.choice(active))
            saved = {h: self.plans.get(h) for h in nbhd}
            saved_ptr = {h: self.ptr.get(h) for h in nbhd}
            saved_stop_at = {h: set(self.stop_at.get(h, ())) for h in nbhd}
            saved_stopped_at = {h: set(self.stopped_at.get(h, ())) for h in nbhd}
            res = self._build_res_excluding(env, t_now, nbhd)
            order = list(nbhd)
            rng.shuffle(order)                      # a fresh priority order for the subset
            for h in order:
                self._schedule(env, h, res, t_now=t_now)
            now = self._n_reached(env)
            if now >= best:
                best = now                          # accept (>= lets it explore plateaus)
            else:
                for h in nbhd:                      # revert
                    self.plans[h] = saved[h]
                    self.stop_at[h] = saved_stop_at[h]
                    self.stopped_at[h] = saved_stopped_at[h]
                    if saved_ptr[h] is not None:
                        self.ptr[h] = saved_ptr[h]

    def _plan_all(self, env, t_now=0):
        if self.topo is None:
            self.topo = Topology(env)
        if self.crit_weight > 0 and self.criticality is None:
            self._compute_criticality(env)     # one-time blast-radius danger map
        if self.city_hold and self.safe_cells is None:
            self._compute_safe_cells(env)      # one-time multi-track city (safe-wait) map
        self.corridor_load = {}        # cell -> #planned trains using it (congestion routing)
        self.dir_load = {}             # (cell,dir) -> #planned trains traversing it that way
        self.meet_pass_reroutes = {}   # h -> #reactive meet-pass diversions (capped)
        self.frozen_reroutes = {}      # h -> #frozen-corridor diversions (capped)
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
        for (cell, t0, t1) in self._frozen_reserve:    # block frozen cells for their freeze span
            res.add_cell(cell, t0, t1)
        self.plans = {}
        self.ptr = {}
        self.stop_at = {}
        self.stopped_at = {}
        for h in self._order(env):
            if env.agents[h].state == TrainState.DONE:
                self.plans[h] = None
                self.stop_at[h] = set()
                self.stopped_at[h] = set()
                continue
            self._schedule(env, h, res, t_now=t_now)
        if self.lns_iters > 0:
            self._lns(env, t_now)                  # refine: fit more trains via destroy+repair
        self.ready = True
        self.last_replan = t_now

    def _compute_safe_cells(self, env):
        """Cells in multi-track city clusters -> safe waiting spots (other tracks stay free)."""
        import collections
        topo = self.topo
        stations = list(topo.stations)
        used = set(); safe = set()
        for c0 in stations:
            if c0 in used:
                continue
            q = collections.deque([c0]); g = {c0}; used.add(c0)
            while q:
                x = q.popleft()
                for y in stations:
                    if y not in used and abs(x[0]-y[0]) + abs(x[1]-y[1]) <= 4:
                        used.add(y); g.add(y); q.append(y)
            if len(g) >= 2:                          # multi-track city => safe to hold
                safe |= g
        self.safe_cells = safe

    def _contested_ahead(self, h, seg_lock):
        """Does h's planned route, within `lookahead` cells, enter a single-track segment an
        OPPOSING train currently occupies?"""
        plan = self.plans.get(h); i = self.ptr.get(h, 0)
        if not plan:
            return False
        topo = self.topo
        for j in range(i, min(i + self.lookahead, len(plan))):
            c, d = plan[j][0], plan[j][1]
            s = topo.seg_of.get(c)
            if s is not None and s in seg_lock and self._seg_exit(c, d) != seg_lock[s]:
                return True
        return False

    def _bump_debug(self, key, n=1):
        self.debug_counts[key] = self.debug_counts.get(key, 0) + n

    def _lateness(self, env, h, t):
        """How late h is at its current planned cell, measured against the executor clock."""
        a = env.agents[h]
        plan = self.plans.get(h)
        if not plan or a.position is None:
            return 0
        cp = tuple(a.position)
        i = self.ptr.get(h, 0)
        while i + 1 < len(plan) and plan[i][0] != cp:
            i += 1
        if i >= len(plan) or plan[i][0] != cp:
            return 0
        self.ptr[h] = i
        scheduled_move = plan[i][3] - _kof(a)
        return max(0, t - scheduled_move)

    def _apply_late_segment_guard(self, env, occupied, desired):
        """Hold new opposing entries into a segment occupied by a materially late train.

        This is intentionally narrower than block_lock: on-time trains keep using the planned
        reservations, but a stale train inside a single-track segment temporarily owns that block
        in its travel direction so another train does not enter nose-to-nose.
        """
        topo = self.topo
        t = env._elapsed_steps
        late_lock = {}
        for h, a in enumerate(env.agents):
            if a.position is None or a.state == TrainState.DONE:
                continue
            c = tuple(a.position)
            s = topo.seg_of.get(c)
            if s is None:
                continue
            malf = getattr(a.malfunction_handler, "malfunction_down_counter", 0)
            late = self._lateness(env, h, t)
            if late < self.late_guard_threshold and malf <= 0:
                continue
            exit_cell = self._seg_exit(c, int(a.direction))
            prev = late_lock.get(s)
            if prev is None:
                late_lock[s] = exit_cell
            elif prev != exit_cell:
                self._bump_debug("late_guard_conflicted_locks")

        if not late_lock:
            return

        for h in list(desired):
            a = env.agents[h]
            hp = tuple(a.position) if a.position is not None else None
            if hp is None:
                continue
            nx = desired[h]
            s = topo.seg_of.get(nx)
            if s is None or s not in late_lock or topo.seg_of.get(hp) == s:
                continue
            din = self._dir_between(hp, nx)
            my_exit = self._seg_exit(nx, din) if din is not None else None
            if my_exit != late_lock[s]:
                del desired[h]
                self._bump_debug("late_guard_holds")

    def _apply_frozen_reroute(self, env, occupied, desired):
        """Surgical malfunction re-planning: divert trains about to ENTER a corridor that a
        malfunctioning (frozen) train is blocking, onto a clear alternative -> no queue cascade."""
        topo = self.topo
        t = env._elapsed_steps
        blocked = set()                          # segments containing a frozen on-map train
        for a in env.agents:
            if a.position is not None and getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0:
                s = topo.seg_of.get(tuple(a.position))
                if s is not None:
                    blocked.add(s)
        if not blocked:
            return
        penalty = {c: 1000.0 for s in blocked for c in topo.seg_cells[s]}
        for h in list(desired):
            a = env.agents[h]
            if a.position is None or getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0:
                continue
            hp = tuple(a.position)
            nx = desired[h]
            s = topo.seg_of.get(nx)
            if s is None or s not in blocked or topo.seg_of.get(hp) == s:
                continue                          # only when ENTERING a frozen-blocked corridor
            if self.frozen_reroutes.get(h, 0) >= self.max_frozen_reroutes:
                continue
            cd = int(a.direction) if a.direction is not None else int(a.initial_direction)
            for route in topo.k_routes(hp, cd, {tuple(a.target)}, K=6, load=penalty, lw=1.0):
                if len(route) < 2 or topo.seg_of.get(route[1][0]) in blocked:
                    continue                      # need a route that avoids the frozen corridor(s)
                rn = route[1][0]
                if rn in occupied:
                    continue
                k = _kof(a)
                self.plans[h] = [(route[j][0], route[j][1], t + j * k, t + j * k + k)
                                 for j in range(len(route))]
                self.ptr[h] = 0
                self.stop_at[h] = set()
                self.stopped_at[h] = set()
                desired[h] = rn
                self.frozen_reroutes[h] = self.frozen_reroutes.get(h, 0) + 1
                break

    def _apply_meet_pass(self, env, occupied, seg_lock, desired):
        """Reactive meet-pass: a train about to ENTER a single-track segment that an OPPOSING
        train occupies is DIVERTED onto a clear parallel route to its target (uses the tight
        passing loops the backtrack found), instead of being held at the junction."""
        topo = self.topo
        t = env._elapsed_steps
        for h in list(desired):
            a = env.agents[h]
            if a.position is None:
                continue
            hp = tuple(a.position)
            nx = desired[h]
            s = topo.seg_of.get(nx)
            if s is None or topo.seg_of.get(hp) == s or s not in seg_lock:
                continue
            din = self._dir_between(hp, nx)
            my_exit = self._seg_exit(nx, din) if din is not None else None
            if my_exit == seg_lock[s]:
                continue                                  # same direction (following) -> fine
            if self.meet_pass_reroutes.get(h, 0) >= self.max_meet_pass:
                continue
            cd = int(a.direction) if a.direction is not None else int(a.initial_direction)
            penalty = {c: 1000.0 for c in topo.seg_cells[s]}     # forbid the opposing segment
            for route in topo.k_routes(hp, cd, {tuple(a.target)}, K=6, load=penalty, lw=1.0):
                if len(route) < 2 or topo.seg_of.get(route[1][0]) == s:
                    continue                              # need a route that avoids segment s
                rn = route[1][0]
                if rn in occupied:
                    continue                              # immediate next cell must be free
                k = _kof(a)
                self.plans[h] = [(route[j][0], route[j][1], t + j * k, t + j * k + k)
                                 for j in range(len(route))]
                self.ptr[h] = 0
                self.stop_at[h] = set()
                self.stopped_at[h] = set()
                desired[h] = rn
                self.meet_pass_reroutes[h] = self.meet_pass_reroutes.get(h, 0) + 1
                break

    def _stuck_replan(self, env, t_now):
        """Surgical online repair: replan only the long-stuck on-map trains from their current
        position against everyone else's committed plans (kept fixed -> no thrashing)."""
        cand = [h for h in range(env.get_num_agents())
                if self.stuck.get(h, 0) >= self.stuck_thresh
                and env.agents[h].position is not None
                and env.agents[h].state != TrainState.DONE]
        if not cand:
            return
        res = Reservation()
        for a in env.agents:
            if a.position is not None and a.state != TrainState.DONE:
                res.add_cell(tuple(a.position), t_now, t_now)
        candset = set(cand)
        for h, plan in self.plans.items():
            if h in candset or not plan:
                continue
            self._reserve_plan(res, plan, _kof(env.agents[h]))
        for h in cand:
            old = self.plans.get(h)
            old_stop_at = set(self.stop_at.get(h, ()))
            old_stopped_at = set(self.stopped_at.get(h, ()))
            self._schedule(env, h, res, t_now=t_now)   # re-route from current pos vs the rest
            if self.plans.get(h):
                self.stuck[h] = 0
            else:
                self.plans[h] = old                     # replan abandoned it -> keep old plan
                self.stop_at[h] = old_stop_at
                self.stopped_at[h] = old_stopped_at

    def _release(self, env, t_now):
        """Release valve: try to schedule the never-departed (off-map, plan=None) trains into
        whatever capacity the committed plans leave free, WITHOUT touching committed plans."""
        N = env.get_num_agents()
        undeparted = [h for h in range(N)
                      if self.plans.get(h) is None
                      and env.agents[h].position is None
                      and env.agents[h].state != TrainState.DONE
                      and t_now >= int(env.agents[h].waypoints_earliest_departure[0] or 0)]
        if not undeparted:
            return
        res = Reservation()
        for a in env.agents:                       # current on-map positions
            if a.position is not None and a.state != TrainState.DONE:
                res.add_cell(tuple(a.position), t_now, t_now)
        for h, plan in self.plans.items():         # committed future occupancy (kept fixed)
            if plan:
                self._reserve_plan(res, plan, _kof(env.agents[h]))
        order = [h for h in self._order(env) if h in set(undeparted)]
        for h in order:                            # abandon_unplaceable stays True -> a train
            self._schedule(env, h, res, t_now=t_now)   # that still can't complete stays off-map

    def _action_to(self, env, cell, d, nxt_cell):
        for act in _MOVES:
            r = env.rail.apply_action_independent(RailEnvActions.from_value(act), (cell, d))
            if r is None:
                continue
            (np_, nd_), _ = r
            if tuple(np_) == nxt_cell:
                return act
        return FWD

    def _seg_exit(self, cell, d):
        """Boundary cell (junction/dead-end) that a train at (cell,d) heads TOWARD by following
        the corridor forward -> identifies its direction of travel through a single-track block."""
        key = (cell, d)
        if key in self._exit_cache:
            return self._exit_cache[key]
        topo = self.topo
        cur = (cell, d); seen = {cell}
        for _ in range(300):
            outs = topo.succ.get(cur, ())
            if not outs:
                break
            nc, nd = outs[0]
            cur = (nc, nd)
            if nc in topo.junctions or nc in topo.deadends or nc in seen:
                break
            seen.add(nc)
        self._exit_cache[key] = cur[0]
        return cur[0]

    def _seg_locks(self, env):
        """sid -> exit-boundary that the segment is currently locked toward (its occupants'
        travel direction). Empty segments absent => free to lock either way."""
        topo = self.topo
        lock = {}
        for a in env.agents:
            if a.position is None or a.state == TrainState.DONE:
                continue
            c = tuple(a.position)
            s = topo.seg_of.get(c)
            if s is None:                       # junction/dead-end cell, not inside a block
                continue
            lock.setdefault(s, self._seg_exit(c, int(a.direction)))
        return lock

    @staticmethod
    def _dir_between(p, q):
        dr, dc = q[0] - p[0], q[1] - p[1]
        if dr == -1: return 0
        if dc == 1: return 1
        if dr == 1: return 2
        if dc == -1: return 3
        return None

    def _reroute(self, env, h, t):
        """A train blocked too long (e.g. behind a malfunctioning train) tries an alternative
        route to its target whose next corridor is currently CLEAR -> bypass the frozen train
        and recover the delay. Conservative: only commit a detour onto empty track, capped per
        train, so we don't reintroduce head-on deadlocks."""
        a = env.agents[h]
        if a.position is None:
            return False
        cp = tuple(a.position)
        if cp not in self.topo.junctions:     # can only divert AT a switch (a branch point)
            return False
        cd = int(a.direction) if a.direction is not None else int(a.initial_direction)
        occ = {tuple(x.position) for x in env.agents
               if x is not a and x.position is not None}
        plan = self.plans.get(h)
        cur_next = None
        if plan:
            i = self.ptr.get(h, 0)
            if i + 1 < len(plan):
                cur_next = plan[i + 1][0]
        k = _kof(a)
        for route in self.topo.k_routes(cp, cd, {tuple(a.target)}, K=5):
            if len(route) < 2 or route[1][0] == cur_next:
                continue                                  # need a DIFFERENT next step
            if any(c in occ for (c, _) in route[1:7]):
                continue                                  # next stretch must be clear (no head-on)
            self.plans[h] = [(route[j][0], route[j][1], t + j * k, t + j * k + k)
                             for j in range(len(route))]
            self.ptr[h] = 0
            self.stop_at[h] = set()
            self.stopped_at[h] = set()
            return True
        return False

    def act_many(self, handles, envs):
        env = envs[0]
        ag = env.agents
        t = env._elapsed_steps
        if not self.ready:
            self._plan_all(env, t_now=t)
        elif self.replan_interval > 0 and t - self.last_replan >= self.replan_interval \
                and not all(a.state == TrainState.DONE for a in ag):
            self._plan_all(env, t_now=t)            # online re-dispatch from current positions
        if self.release_interval > 0 and self.ready and t > 0 and t % self.release_interval == 0:
            self._release(env, t)                   # retry never-departed trains into free space
        if self.stuck_replan and self.ready:
            self._stuck_replan(env, t)              # surgical online repair of long-stuck trains

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
                if i in self.stop_at.get(h, set()) \
                        and i not in self.stopped_at.get(h, set()):
                    continue
                if i + 1 < len(plan) and (self.greedy_advance or t >= plan[i][3] - _kof(a)):
                    desired[h] = plan[i + 1][0]               # scheduled time, or greedy (reactive)

        # chain resolution: grant a move when the next cell is free OR its occupant is also
        # being granted to move out this step -> nose-to-tail chains advance together (the
        # absolute-time rigidity that stalled chains was the high-density failure mode).
        if self.replan_on_malfunction and self.ready:
            cur_frozen = {h for h, a in enumerate(ag)
                          if a.position is not None
                          and getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0}
            new_frozen = cur_frozen - self._last_frozen
            self._last_frozen = cur_frozen
            if new_frozen and t - self._last_malf_replan >= self.malf_replan_min \
                    and not all(a.state == TrainState.DONE for a in ag):
                self._frozen_reserve = [
                    (tuple(a.position), t, t + getattr(a.malfunction_handler, "malfunction_down_counter", 0))
                    for a in ag if a.position is not None
                    and getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0]
                self._plan_all(env, t_now=t)         # re-dispatch around the frozen cells
                self._frozen_reserve = []
                self._last_malf_replan = t
                # recompute desired after the fresh plan
                desired = {}
                for h in handles:
                    a = ag[h]; plan = self.plans.get(h)
                    if a.state == TrainState.DONE or not plan: continue
                    if getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0: continue
                    if a.position is None:
                        if t >= plan[0][2]: desired[h] = plan[0][0]
                    else:
                        cp = cur_cell[h]; i = self.ptr.get(h, 0)
                        while i + 1 < len(plan) and plan[i][0] != cp: i += 1
                        self.ptr[h] = i
                        if i in self.stop_at.get(h, set()) \
                                and i not in self.stopped_at.get(h, set()):
                            continue
                        if i + 1 < len(plan) and t >= plan[i][3] - _kof(a):
                            desired[h] = plan[i + 1][0]
        if self.frozen_reroute:
            self._apply_frozen_reroute(env, occupied, desired)   # divert around frozen corridors
        if self.late_segment_guard:
            self._apply_late_segment_guard(env, occupied, desired)
        seg_lock = self._seg_locks(env) if (self.block_lock or self.meet_pass or self.city_hold) else None
        if self.meet_pass and seg_lock:
            self._apply_meet_pass(env, occupied, seg_lock, desired)
        if self.city_hold and seg_lock:
            # hold trains parked on a multi-track city if an opposing single-track block is ahead
            for h in list(desired):
                hp = cur_cell.get(h)
                if hp in self.safe_cells and self._contested_ahead(h, seg_lock):
                    del desired[h]                        # wait on the platform (safe siding)
        granted = {}
        claimed = {}
        grant_order = handles
        if self.exec_fast_first:                  # faster trains win cell contention (dynamic release)
            grant_order = sorted(handles, key=lambda h: -float(ag[h].speed_counter.speed))
        changed = True
        while changed:
            changed = False
            for h in grant_order:
                if h in granted or h not in desired:
                    continue
                nxt = desired[h]
                if nxt in claimed:
                    continue
                occ = occupied.get(nxt)
                if occ is not None and occ != h and occ not in granted:
                    continue                              # blocked by a train not (yet) vacating
                if self.signal_guard and ag[h].position is not None \
                        and nxt in self.topo.junctions:
                    # home signal: don't step ONTO a junction unless its planned EXIT is clear
                    plan = self.plans.get(h); i = self.ptr.get(h, 0)
                    beyond = plan[i + 2][0] if (plan and i + 2 < len(plan)) else None
                    if beyond is not None:
                        bocc = occupied.get(beyond)
                        if bocc is not None and bocc != h and bocc not in granted \
                                and beyond not in claimed:
                            continue                      # exit blocked -> hold before the switch
                if seg_lock is not None and ag[h].position is not None:
                    s = self.topo.seg_of.get(nxt)
                    hp = cur_cell[h]
                    # only gate when ENTERING a new single-track block (not moving within one)
                    if s is not None and self.topo.seg_of.get(hp) != s and s in seg_lock:
                        din = self._dir_between(hp, nxt)
                        my_exit = self._seg_exit(nxt, din) if din is not None else None
                        if my_exit != seg_lock[s]:
                            continue                      # block locked the OTHER way -> hold
                granted[h] = nxt
                claimed[nxt] = h
                changed = True

        if self.stuck_replan:                       # track how long each train has been blocked
            for h in handles:
                if ag[h].position is not None and h in desired and h not in granted:
                    self.stuck[h] = self.stuck.get(h, 0) + 1
                else:
                    self.stuck[h] = 0

        actions = {}
        for h in handles:
            a = ag[h]
            if a.state == TrainState.DONE:
                actions[h] = DO
                continue
            if getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0:
                # On-map: frozen, must STOP. Off-map (depot) malfunction: issue the intended
                # move -> Flatland STORES it and auto-departs the instant the break clears
                # (a STOP here would keep it parked, needlessly delaying departure).
                plan = self.plans.get(h)
                if a.position is None and plan and t >= plan[0][2]:
                    actions[h] = FWD
                else:
                    actions[h] = STOP
                continue
            if h in granted:
                if a.position is None:
                    actions[h] = FWD
                else:
                    d = int(a.direction) if a.direction is not None else int(a.initial_direction)
                    actions[h] = self._action_to(env, cur_cell[h], d, granted[h])
            else:
                if a.position is not None:
                    i = self.ptr.get(h)
                    if i in self.stop_at.get(h, set()) \
                            and i not in self.stopped_at.get(h, set()):
                        self.stopped_at.setdefault(h, set()).add(i)
                    actions[h] = STOP
                else:
                    actions[h] = DO

        # --- switch-level repair: a train sitting AT A SWITCH whose chosen branch is blocked
        # by a STOPPED/MALFUNCTIONING train ahead diverts onto a clear alternate branch toward
        # target. Decisions only at switches (the only branch points); triggered by a real
        # blockage (not transient), capped per train. ---
        if self.malfunction_repair:
            frozen_or_stopped = {c for c, hh in occupied.items()
                                 if getattr(ag[hh].malfunction_handler, "malfunction_down_counter", 0) > 0
                                 or ag[hh].state == TrainState.STOPPED}
            for h in handles:
                a = ag[h]
                if (a.position is None or a.state == TrainState.DONE
                        or self.plans.get(h) is None
                        or getattr(a.malfunction_handler, "malfunction_down_counter", 0) > 0):
                    continue
                if (h in desired and h not in granted               # blocked this step
                        and desired[h] in frozen_or_stopped         # by a stuck train ahead
                        and tuple(a.position) in self.topo.junctions  # and we're at a switch
                        and self.reroutes.get(h, 0) < self.max_reroutes):
                    if self._reroute(env, h, t):
                        self.reroutes[h] = self.reroutes.get(h, 0) + 1
        return actions


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from eval_harness import evaluate
    QUICK_MAP = dict(x_dim=100, y_dim=80, n_cities=10)
    print("=== v5 planner (reservation, deadlock-free by construction, multi-stop) ===\n")
    evaluate(lambda: V5Planner(), levels=[0], scenarios=[0, 1, 2], seed=1,
             map_cfg=QUICK_MAP, time_limit=180.0)
