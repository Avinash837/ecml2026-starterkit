"""
Event-driven interlocking controller on top of bay_graph.BayGraph.

Safety design (no global clock, no timed reservations):

1. BAY SLOTS. A train may stand indefinitely only in a bay (loop sibling,
   stub, or its target sink). Slots are counted against bay capacity and the
   parallel-group keep-one-clear rule (standing members <= n_routes - 1).

2. ATOMIC SECTION CLAIMS. Before leaving a bay (or the depot), a train
   atomically claims its entire path to the NEXT bay: a directional claim on
   every block in between plus a slot at that bay. Opposing claims on a block
   are mutually exclusive; standing trains make a block unclaimable; claimed
   blocks reject new standing slots. Therefore trains in transit own their
   whole path, opposing trains can only meet at bays, and the head-on
   circular wait of per-block locking cannot occur.

3. ROLLING UPGRADE. A train holding a slot at bay j tries, every step, to
   claim section j -> j+1 and a slot at j+1; on success it releases slot j
   and rolls through without stopping. Grants are evaluated in rounds until
   fixpoint, so a slot freed by one grant can satisfy the next in the same
   step (swap patterns resolve instead of gridlocking).

4. PROGRESS. Inside a claimed section only co-directional traffic exists and
   the head train always has free claimed track or its reserved slot ahead,
   so it can advance unless malfunctioning. Malfunction = a held claim;
   everything behind queues by the same rules; nothing is replanned.

Multistop: routes are searched on the directed (cell, heading) graph with
direction-correct goal SETS per stop (any platform alternative of the
station, required heading). The serving platform gets a 1-step STOP_MOVING
dwell (registers TrainState.STOPPED, which is what ECML2026Rewards checks).
"""
from collections import Counter, defaultdict
import heapq
from flatland.envs.rail_env_action import RailEnvActions as A
from flatland.envs.step_utils.states import TrainState

SINK = -1  # virtual infinite bay at the train's target
PSW_SLOT = -2  # sentinel: train is partial-section-waiting (no bay slot held)
OPP = {0: 2, 1: 3, 2: 0, 3: 1}


def node_movement(path, idx):
    """(entry_heading, exit_heading) of the traversal of node cell path[idx]."""
    entry = path[idx][1]
    exit_ = path[idx + 1][1] if idx + 1 < len(path) else entry
    return (entry, exit_)


# ---------------------------------------------------------------- router
class Router:
    BAY_PENALTY = 2  # extra cost per bay-block cell: keep through traffic off parking tracks

    def __init__(self, bay_graph):
        self.bg = bay_graph
        bg = bay_graph
        self._bay_cell = {c for b in bg.blocks if b.is_bay for c in b.chain}
        self._hcache = {}     # frozenset(goals) -> dist field (backward BFS)

    def dist_field(self, goals):
        """True remaining-distance table to a goal SET over (cell, dir) --
        one backward BFS on the fixed map, cached and shared across all
        agents, replans and recoveries that target the same station group.
        Doubles as a perfect admissible A* heuristic (penalties only ADD
        cost, so the unpenalized distance never overestimates)."""
        key = frozenset(goals)
        f = self._hcache.get(key)
        if f is not None:
            return f
        from collections import deque
        pred = self.bg.pred
        dist = {g: 0 for g in goals}
        q = deque(goals)
        while q:
            cur = q.popleft()
            d = dist[cur]
            for prv in pred.get(cur, ()):
                if prv not in dist:
                    dist[prv] = d + 1
                    q.append(prv)
        self._hcache[key] = dist
        return dist

    def _leg(self, start, goals, extra=None):
        """Dijkstra on (cell, dir); goals: set of (cell, dir). Bay cells cost
        1 + BAY_PENALTY unless they are a goal cell; `extra` adds dynamic
        per-cell penalties (occupied corridors, full bays)."""
        succ = self.bg.succ
        goal_cells = {g[0] for g in goals}
        h = self.dist_field(goals)
        if start not in h:
            return None                       # goal unreachable: prune now
        dist, prev = {start: 0}, {}
        pq = [(h[start], start)]
        while pq:
            f, cur = heapq.heappop(pq)
            if cur in goals:
                path = [cur]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return path[::-1]
            d = dist.get(cur, 1 << 30)
            if f - h.get(cur, 0) > d:
                continue
            for nxt in succ.get(cur, ()):
                hn = h.get(nxt)
                if hn is None:
                    continue                  # cannot reach goal from there
                step = 1
                if nxt[0] in self._bay_cell and nxt[0] not in goal_cells:
                    step += self.BAY_PENALTY
                if extra:
                    step += extra.get(nxt[0], 0) + extra.get(nxt, 0)
                nd = d + step
                if nd < dist.get(nxt, 1 << 30):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(pq, (nd + hn, nxt))
        return None

    def route(self, agent):
        """Full multistop route from the agent's origin."""
        start = (tuple(agent.initial_position), int(agent.initial_direction))
        gis = list(range(1, len(agent.waypoints)))
        return self.route_from(agent, start, gis)

    def _goals_for(self, agent, gi):
        goals = set()
        for w in agent.waypoints[gi]:
            pos = tuple(w.position)
            if w.direction is None:
                goals.update((pos, d) for d in range(4))
            else:
                goals.add((pos, int(w.direction)))
        return goals

    @staticmethod
    def _matches_stop(cell, direction, alts):
        for w in alts:
            if tuple(w.position) != tuple(cell):
                continue
            if w.direction is None or int(w.direction) == int(direction):
                return True
        return False

    def route_from(self, agent, start, gis, extra=None):
        return self._route_from_full(agent, start, gis, extra)

    def _route_from_full(self, agent, start, gis, extra=None):
        """Route from `start` through waypoint groups `gis` (ascending,
        ending with the final group). Returns ((path, stops), err)."""
        path, stops = [start], {}
        wps = agent.waypoints
        for gi in gis:
            goals = self._goals_for(agent, gi)
            leg = self._leg(path[-1], goals, extra)
            if leg is None and gi == len(wps) - 1:       # final: any heading
                goals = {(tuple(agent.target), d) for d in range(4)}
                leg = self._leg(path[-1], goals, extra)
            if leg is None:
                return None, f"no leg to stop {gi}"
            path.extend(leg[1:])
            if 1 <= gi < len(wps) - 1:
                stops[len(path) - 1] = gi
        return (path, stops), None

    def route_direct_opportunistic(self, agent, start, gis, extra=None,
                                   collect_stops=True):
        """V5-style direct route with free stop recovery.

        Route only to the final target, then add 1-step dwell stops for
        intermediate waypoints that already appear on that direct path in the
        required order and direction. This keeps dense long-line throughput
        close to direct routing while collecting cheap ECML stop credit.
        """
        if not gis:
            return ([start], {}), None
        final_gi = len(agent.waypoints) - 1
        direct, err = self._route_from_full(agent, start, [final_gi], extra)
        if direct is None:
            return None, err
        path, _ = direct
        stops = {}
        if not collect_stops:
            return (path, stops), None
        search_from = 0
        for gi in [g for g in gis if 1 <= g < final_gi]:
            found = None
            for idx in range(search_from + 1, len(path) - 1):
                cell, direction = path[idx]
                if self._matches_stop(cell, direction, agent.waypoints[gi]):
                    found = idx
                    break
            if found is not None:
                stops[found] = gi
                search_from = found
        return (path, stops), None


# ---------------------------------------------------------------- ledger
class Ledger:
    def __init__(self, bay_graph):
        self.bg = bay_graph
        self.slots = {}      # bid -> [exit_end, set(handles)] standing cohort
        self.transit = {}    # bid -> [end_label, {handle: count}]
        self.phys = {}       # bid -> set(handles) physically inside (any kind)
        self.nodes = {}      # cell -> {handle: [(entry, exit), ...]}
        self.edges = {}      # node-node edge key -> [fwd_flag, {handle: n}]
        # node-node EDGES are directional resources: on junction-dense maps
        # two opposing trains with turning movements can otherwise claim
        # adjacent nodes and stall face-to-face on the edge between them
        # (observed deadlock on env_generator maps), because neither the
        # block rules (no block involved) nor the reversed-movement rule
        # (movements aren't mirror images) forbids it.
        self.slot_of = {}    # handle -> bid of its current slot
        self.t = 0           # controller-maintained clock
        self.last_dir = {}   # bid -> (end, t_freed): direction hysteresis
        self.DIR_COOLDOWN = 0
        self.MAX_STANDING = None   # per-loop-bay standing cap (None = physical)

    # -- bookkeeping helpers
    def slot_holders(self, bid):
        s = self.slots.get(bid)
        return set(s[1]) if s else set()

    def standing_in(self, bid):
        """Handles genuinely PARKED in bid: slot holders, plus anyone
        physically inside who neither transits it nor holds a slot
        elsewhere (a granted train still vacating its old bay is in
        motion, not parked)."""
        t = self.transit.get(bid)
        through = set(t[1]) if t else set()
        parked_phys = {h for h in set(self.phys.get(bid, ())) - through
                       if self.slot_of.get(h, bid) == bid}
        parked_slots = {h for h in self.slot_holders(bid)
                        if self.slot_of.get(h) == bid}
        return parked_slots | parked_phys

    def _group_ok(self, bid, handle):
        """Keep-one-clear is a routing PREFERENCE, not a grant condition.
        Trains in motion are protected by their atomic section claims
        (standing grants are denied wherever transit claims exist), so
        hard-enforcing group clearance defends nobody -- and provably
        deadlocks converging hubs. Always grantable at this layer."""
        return True

    def _block_claimable(self, bid, end, handle):
        others_slots = self.slot_holders(bid) - {handle}
        if others_slots:
            return False                      # parked / reserved by others
        others_phys = set(self.phys.get(bid, ())) - {handle}
        if others_phys:
            # NOTE: co-directional "platooning" past physical occupants was
            # tested and collapses completion (17.8% -> 5.3% on the 320
            # proxy): convoys pile into bays together and saturate
            # stations. Corridor emptiness doubles as admission control.
            return False                      # someone is physically inside
        t = self.transit.get(bid)
        if t is not None and t[1]:
            return t[0] == end
        # empty corridor: hold its last direction for a cooldown so
        # co-directional followers batch through before the flow flips
        ld = self.last_dir.get(bid)
        if ld and ld[0] != end and (self.t - ld[1]) < self.DIR_COOLDOWN:
            return False
        return True

    def _slot_free(self, bid, end, handle):
        if bid == SINK:
            return True
        b = self.bg.blocks[bid]
        if not b.is_bay:
            return False
        if self.transit.get(bid) and self.transit[bid][1]:
            return False                      # block in use as through path
        s = self.slots.get(bid)
        if s and (set(s[1]) - {handle}) and s[0] != end:
            return False                      # opposing standing cohort
        if b.kind == "stub" and (self.slot_holders(bid) - {handle}
                                 or set(self.phys.get(bid, ())) - {handle}):
            return False                      # stubs: one occupant at a time
        occupants = self.slot_holders(bid) | set(self.phys.get(bid, ()))
        occupants.discard(handle)
        cap = b.capacity
        if self.MAX_STANDING is not None and b.kind == "loop":
            cap = min(cap, self.MAX_STANDING)
        return len(occupants) + 1 <= cap and self._group_ok(bid, handle)

    @staticmethod
    def _reversed(m1, m2):
        return m2[0] == OPP[m1[1]] and m2[1] == OPP[m1[0]]

    def _node_claimable(self, cell, mov, handle):
        for other, movs in self.nodes.get(cell, {}).items():
            if other == handle:
                continue
            if any(self._reversed(m, mov) for m in movs):
                return False
        return True

    def _edge_claimable(self, key, fwd, handle):
        e = self.edges.get(key)
        if not e or not e[1]:
            return True
        if e[0] == fwd:
            return True
        return not (set(e[1]) - {handle})   # opposing: only self-overlap ok

    def release_edge(self, handle, key):
        e = self.edges.get(key)
        if e and handle in e[1]:
            e[1][handle] -= 1
            if e[1][handle] <= 0:
                del e[1][handle]
            if not e[1]:
                del self.edges[key]

    # -- atomic section claim: blocks + node movements + edges + bay slot
    def try_claim(self, handle, blocks, node_movs, bay_bid, bay_end,
                  edges=()):
        for bid, end in blocks:
            if not self._block_claimable(bid, end, handle):
                return False
        for cell, mov in node_movs:
            if not self._node_claimable(cell, mov, handle):
                return False
        for key, fwd, _b in edges:
            if not self._edge_claimable(key, fwd, handle):
                return False
        if not self._slot_free(bay_bid, bay_end, handle):
            return False
        for bid, end in blocks:
            t = self.transit.setdefault(bid, [end, {}])
            t[0] = end
            t[1][handle] = t[1].get(handle, 0) + 1
        for cell, mov in node_movs:
            self.nodes.setdefault(cell, {}).setdefault(handle, []).append(mov)
        for key, fwd, _b in edges:
            e = self.edges.setdefault(key, [fwd, {}])
            e[0] = fwd
            e[1][handle] = e[1].get(handle, 0) + 1
        if bay_bid != SINK:
            s = self.slots.setdefault(bay_bid, [bay_end, set()])
            s[0] = bay_end
            s[1].add(handle)
        self.slot_of[handle] = bay_bid
        return True

    def try_claim_psw(self, handle, blocks, node_movs, edges=()):
        """Partial-section claim: reserve the section's blocks/nodes/edges
        DIRECTIONALLY (head-on safe, same as try_claim) but WITHOUT a far bay
        slot. Lets a train advance into and wait on a through-block it owns,
        freeing the bay behind it. Deadlock-freedom is the caller's
        responsibility (it must run the wait-for gate before calling)."""
        for bid, end in blocks:
            if not self._block_claimable(bid, end, handle):
                return False
        for cell, mov in node_movs:
            if not self._node_claimable(cell, mov, handle):
                return False
        for key, fwd, _b in edges:
            if not self._edge_claimable(key, fwd, handle):
                return False
        for bid, end in blocks:
            t = self.transit.setdefault(bid, [end, {}])
            t[0] = end
            t[1][handle] = t[1].get(handle, 0) + 1
        for cell, mov in node_movs:
            self.nodes.setdefault(cell, {}).setdefault(handle, []).append(mov)
        for key, fwd, _b in edges:
            e = self.edges.setdefault(key, [fwd, {}])
            e[0] = fwd
            e[1][handle] = e[1].get(handle, 0) + 1
        return True

    def release_node(self, handle, cell, mov):
        movs = self.nodes.get(cell, {}).get(handle)
        if movs and mov in movs:
            movs.remove(mov)
            if not movs:
                del self.nodes[cell][handle]
                if not self.nodes[cell]:
                    del self.nodes[cell]

    def release_slot(self, handle, bid):
        s = self.slots.get(bid) if bid != SINK else None
        if s:
            s[1].discard(handle)
            if not s[1]:
                del self.slots[bid]
        if self.slot_of.get(handle) == bid:
            del self.slot_of[handle]

    def release_transit(self, handle, bid):
        t = self.transit.get(bid)
        if t and handle in t[1]:
            t[1][handle] -= 1
            if t[1][handle] <= 0:
                del t[1][handle]
            if not t[1]:
                self.last_dir[bid] = (t[0], self.t)
                del self.transit[bid]

    def drop_all(self, handle):
        for bid in list(self.transit):
            self.transit[bid][1].pop(handle, None)
            if not self.transit[bid][1]:
                del self.transit[bid]
        for bid in list(self.slots):
            self.release_slot(handle, bid)
        self.slot_of.pop(handle, None)
        for cell in list(self.nodes):
            self.nodes[cell].pop(handle, None)
            if not self.nodes[cell]:
                del self.nodes[cell]
        for key in list(self.edges):
            self.edges[key][1].pop(handle, None)
            if not self.edges[key][1]:
                del self.edges[key]


# ------------------------------------------------------------- controller
class InterlockingController:
    """Gate-1 controller: safe, conflict-free multistop execution."""

    # dense-traffic heuristics (shortest-first planning, one-way initial
    # reroute, frozen-blocker avoidance) only run on lines with at most
    # this many waypoints. Historic value 3 = short lines only; class
    # attribute so harnesses can sweep it (set before construction).
    MAX_WP_DENSE = 3

    # Experimental partial-section waiting scaffold. Keep disabled until it has
    # a full wait-for cycle check: the conservative gate measured inert, while
    # a relaxed gate regressed L2_s2 badly by stranding trains in sections.
    PSW_ENABLED = False

    # Local two-sided "bridge signals" for hot bidirectional bay pairs. These
    # are detected from the final plan set, then batched by direction so paired
    # loop bays do not fill with opposing cohorts that want to swap.
    CORE_SIGNAL_ENABLED = True
    CORE_SIGNAL_MIN_TRANSITIONS = 12
    CORE_SIGNAL_MIN_EACH_DIR = 4
    CORE_SIGNAL_MIN_LOAD = 1.0
    CORE_SIGNAL_PHASE_MIN = 250
    CORE_SIGNAL_PHASE_MAX = 900
    CORE_SIGNAL_LOW_MIN_TRANSITIONS = 4
    CORE_SIGNAL_LOW_MIN_EACH_DIR = 1
    CORE_SIGNAL_LOW_MIN_LOAD = 0.3
    CORE_SIGNAL_LOW_PHASE_MIN = 100
    CORE_SIGNAL_LOW_PHASE_MAX = 400
    CORE_SIGNAL_DRAIN_BEFORE_FLIP = True
    # Occupancy-gated enforcement: the directional batching only PREVENTS a
    # deadlock when both parallel tracks of the pair fill with opposing
    # cohorts. When the pair has ample room, opposing through-traffic uses the
    # two separate tracks safely (base interlocking's per-block directional
    # claims handle it), so restricting direction there is pure waste
    # (Step-4 measured pair (214,234) at 0% both-full but 97% room-while-denied).
    # Enforce the signal only when occupancy reaches this fraction of capacity.
    # MEASURED OFF: occupancy-gating re-introduces the exact deadlock the signal
    # prevents (L2_s2 100 -> 58, the no-signal baseline). The directional
    # restriction is necessary even with spare room -- "room-while-denied" is
    # the price of deadlock prevention, not reclaimable slack. Kept off.
    CORE_SIGNAL_OCC_GATE = False
    CORE_SIGNAL_ENFORCE_FRAC = 0.6

    def __init__(self, env, bay_graph):
        self.env = env
        self.bg = bay_graph
        self.ledger = Ledger(bay_graph)
        mpd = getattr(env, "malfunction_process_data", None)
        self._mrate = getattr(mpd, "malfunction_rate", 0.0) or 0.0
        self.router = self._make_router(bay_graph)
        # density-gated: one-way designation only pays once parallel
        # corridors are contested (measured: helps at n>=50, noise below)
        default_pen = 5 if len(env.agents) >= 50 else 0
        self.ONEWAY_PEN = getattr(self, "ONEWAY_PEN", default_pen)
        self.plans = {}
        self.unroutable = []
        targets = {}
        for a in env.agents:
            targets.setdefault(tuple(a.target), set()).add(a.handle)
        self.targets = targets
        load = {}                     # bid -> {end: planned traversals}
        self._plan_order = self._planning_order()
        order = self._plan_order
        for a in order:
            extra = self._load_extra(load, a.handle)
            gis = list(range(1, len(a.waypoints)))
            start = (tuple(a.initial_position), int(a.initial_direction))
            r, err = self.router.route_from(a, start, gis, extra)
            if r is None:
                r, err = self.router.route(a)
            if r is None:
                self.unroutable.append((a.handle, err))
                continue
            path, stops = r
            self.plans[a.handle] = self._make_plan(a, path, stops)
            for bid, end in self._path_blocks(path):
                load.setdefault(bid, {}).setdefault(end, 0)
                load[bid][end] += 1
        # ---- route equilibrium iteration (blueprint layer 2) ----
        # measure directional block load over all plans, reroute the worst
        # opposing-flow offenders against the residual load, repeat. Runs
        # pre-departure only; A* + cached fields make it cheap.
        if len(env.agents) >= 50:
            self._equilibrate(iters=2, frac=0.25)
            if env.agents and len(env.agents[0].waypoints) <= self.MAX_WP_DENSE:
                load = {}
                for p in self.plans.values():
                    for bid, end in self._path_blocks(p["path"]):
                        load.setdefault(bid, {}).setdefault(end, 0)
                        load[bid][end] += 1
        # single-sided lines: designate siblings of 2+-member parallel
        # groups as one-way pairs based on planned flow, then bias all
        # later routing. On short dense lines we also reroute the initial
        # plans once with that bias; otherwise the bias arrives too late to
        # prevent the first wave from mixing both directions on the same
        # paired loops.
        self.oneway = {}            # (cell, dir) -> penalty
        for g in bay_graph.groups:
            both = [m for m in g.blocks
                    if bay_graph.blocks[m].dir_ok == {"AB", "BA"}
                    and len(bay_graph.blocks[m].chain) > 0]
            if len(both) < 2:
                continue
            flows = {}
            for m in both:
                ends = load.get(m, {})
                flows[m] = max(ends, key=ends.get) if ends else None
            ends_present = {e for e in flows.values() if e}
            if not ends_present:
                continue
            if len(ends_present) == 1:
                only = next(iter(ends_present))
                other = "A" if only == "B" else "B"
                designation = {both[0]: only}
                for m in both[1:]:
                    designation[m] = other
                    other = only if other != only else other
            else:
                designation = {m: (flows[m] or "A") for m in both}
                if len({designation[m] for m in both}) == 1:
                    designation[both[-1]] = ("A" if designation[both[0]] == "B"
                                             else "B")
            for m, want_end in designation.items():
                b = bay_graph.blocks[m]
                chain = b.chain
                for i, cell in enumerate(chain):
                    nxt = chain[i + 1] if i + 1 < len(chain) else b.ends[1]
                    if nxt is None:
                        continue
                    d_ab = self._cell_heading(cell, nxt)
                    if d_ab is None:
                        continue
                    bad = d_ab if want_end == "A" else OPP[d_ab]
                    self.oneway[(cell, bad)] = self.ONEWAY_PEN
        if (len(env.agents) >= 50 and self.ONEWAY_PEN and env.agents
                and len(env.agents[0].waypoints) <= self.MAX_WP_DENSE):
            self._reroute_initial_with_oneway()
        self.state = {h: dict(ptr=0, anchor=0, slot=None, claims=[],
                              edges_held=[], frontier=-1, served=set(),
                              departed=False, stuck=0, pending=[], replans=0,
                              last_bay=None, last_bay_ptr=0, midstuck=0)
                      for h in self.plans}
        self.REPLAN_AFTER = 15
        dense = len(env.agents) >= 50
        self.RECOVER_EVERY = 8 if dense else 20   # patience at low density
        self.FROZEN_RECOVER_EVERY = 4 if dense else 8
        self.RECOVER_BUDGET = 3
        self.FROZEN_REPAIR_ENABLED = self._mrate >= (1.0 / 180.0)
        self.t = 0
        # depot governor: population cap on trains in transit (None disables).
        # Released most-urgent-first; a train near its last feasible departure
        # bypasses the cap. Density-tuned; see WORKAROUNDS.md S7.
        n_ag = len(env.agents)
        self.transit_cap = (None if n_ag < 50 else
                            30 if n_ag < 150 else
                            75 if n_ag < 200 else 60)
        self.FORCE_MARGIN = 25
        # speed-aware dispatch knobs (WORKAROUNDS.md S7). SLOW_CAP: a
        # 0.25-speed train holds every cell 4x longer, so cap how many are in
        # transit; they still depart near deadline via the last_moment bypass.
        self.SLOW_CAP = None if n_ag < 150 else 20
        self.SLOW_SPEED = 0.26      # speed threshold counted by SLOW_CAP
        self.SPEED_TIEBREAK = 0.9   # fast-first weight in the pressure sort
        self.PRESSURE_TIME_UNITS = False  # rank by time not cells (WORSE; off)
        self.FROZEN_TIME_AWARE = False  # malfunction-clock detour pricing (off)
        self.FROZEN_MIN_DOWN = 0
        self.SPAWN_CLEAR = False    # bay-span spawn vacancy (measured inert)
        # sitter-shuffle: a converted spawn-sitter steps one cell along its own
        # depot block so the next sharer can spawn-convert too (WORKAROUNDS S7).
        self.SITTER_SHUFFLE = True
        self._spawn_cells_now = frozenset()
        # DIR_COOLDOWN: hold a freed corridor's direction so co-directional
        # followers batch through before the flow flips (density-gated).
        self.ledger.DIR_COOLDOWN = 10 if len(env.agents) >= 50 else 0
        self.SIT_WINDOW = 60        # endgame cancellation-conversion window

    @staticmethod
    def _cell_heading(frm, to):
        dr, dc = to[0] - frm[0], to[1] - frm[1]
        m = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
        return m.get((dr, dc))

    def _make_router(self, bay_graph):
        return Router(bay_graph)

    def _grant_order(self, pressure):
        return sorted(self.state, key=pressure, reverse=True)

    def _load_extra(self, load, exclude_handle):
        extra = {}
        for bid, ends in load.items():
            b = self.bg.blocks[bid]
            same = max(ends.values()) if ends else 0
            opp = sum(ends.values()) - same
            pen = 3 * opp + 0.5 * same
            if pen:
                for c in b.chain:
                    extra[c] = extra.get(c, 0) + pen
        for tcell, hs in self.targets.items():
            if hs - {exclude_handle}:
                extra[tcell] = extra.get(tcell, 0) + 30
        return extra

    def _initial_oneway_extra(self, load, exclude_handle):
        extra = dict(self.oneway)
        for bid, ends in load.items():
            b = self.bg.blocks[bid]
            same = max(ends.values()) if ends else 0
            opp = sum(ends.values()) - same
            pen = 4 * opp + 0.5 * same
            if pen:
                for c in b.chain:
                    extra[c] = extra.get(c, 0) + pen
        for tcell, hs in self.targets.items():
            if hs - {exclude_handle}:
                extra[tcell] = extra.get(tcell, 0) + 30
        return extra

    def _reroute_initial_with_oneway(self):
        """Second pre-departure pass after paired-loop directions are known."""
        old = self.plans
        new, load = {}, {}
        for a in self._plan_order:
            h = a.handle
            if h not in old:
                continue
            start = (tuple(a.initial_position), int(a.initial_direction))
            gis = list(range(1, len(a.waypoints)))
            r, _ = self.router.route_from(
                a, start, gis, self._initial_oneway_extra(load, h))
            if r is not None and len(r[0]) <= int(1.8 * len(old[h]["path"])):
                new[h] = self._make_plan(a, r[0], r[1])
            else:
                new[h] = old[h]
            for bid, end in self._path_blocks(new[h]["path"]):
                load.setdefault(bid, {}).setdefault(end, 0)
                load[bid][end] += 1
        self.plans = new

    def _planning_order(self):
        agents = list(self.env.agents)
        if not (len(agents) >= 50 and agents and
                len(agents[0].waypoints) <= self.MAX_WP_DENSE):
            return sorted(agents, key=lambda a: a.earliest_departure)
        rough_len = {}
        for a in agents:
            start = (tuple(a.initial_position), int(a.initial_direction))
            gis = list(range(1, len(a.waypoints)))
            r, _ = self.router.route_from(a, start, gis)
            rough_len[a.handle] = len(r[0]) - 1 if r is not None else 1 << 30

        def key(a):
            length = rough_len.get(a.handle, 1 << 30)
            slack = a.latest_arrival - a.earliest_departure - length
            return (length, slack, a.earliest_departure, a.handle)

        return sorted(agents, key=key)

    def _equilibrate(self, iters=2, frac=0.25):
        env = self.env
        for _ in range(iters):
            load, per_train = {}, {}
            for h, p in self.plans.items():
                tb = self._path_blocks(p["path"])
                per_train[h] = tb
                for bid, end in tb:
                    load.setdefault(bid, {}).setdefault(end, 0)
                    load[bid][end] += 1
            # contention: how much opposing flow each train's route crosses
            def contention(h):
                c = 0
                for bid, end in per_train[h]:
                    ends = load.get(bid, {})
                    c += sum(v for e, v in ends.items() if e != end)
                return c
            ranked = sorted(self.plans, key=contention, reverse=True)
            worst = [h for h in ranked[:max(1, int(len(ranked) * frac))]
                     if contention(h) > 0]
            if not worst:
                break
            moved = 0
            for h in worst:
                for bid, end in per_train[h]:
                    load[bid][end] -= 1
                a = env.agents[h]
                extra = self._load_extra(load, h)
                start = (tuple(a.initial_position),
                         int(a.initial_direction))
                gis = list(range(1, len(a.waypoints)))
                r, _ = self.router.route_from(a, start, gis, extra)
                if r is not None and len(r[0]) <= 2 * len(
                        self.plans[h]["path"]):
                    self.plans[h] = self._make_plan(a, r[0], r[1])
                    moved += 1
                for bid, end in self._path_blocks(self.plans[h]["path"]):
                    load.setdefault(bid, {}).setdefault(end, 0)
                    load[bid][end] += 1
            if moved == 0:
                break

    def _path_blocks(self, path):
        """[(bid, end)] traversals of a path, deduped consecutively."""
        out, seen = [], None
        for i, (cell, _) in enumerate(path):
            r = self.bg.resource_of(cell)
            if r[0] != "block":
                seen = None
                continue
            if r[1] == seen:
                continue
            seen = r[1]
            out.append((r[1], self._traverse_end(r[1], path, i)))
        return out

    # -- plan: anchors (bay visits + sink) and sections between them
    def _make_plan(self, agent, path, stops):
        bg = self.bg
        res = []
        for cell, _ in path:
            res.append(bg.resource_of(cell))
        anchors = []                      # (path_idx_first, path_idx_last, bid)
        i = 0
        while i < len(path):
            kind, rid = res[i]
            if kind == "block" and bg.blocks[rid].is_bay:
                j = i
                while j + 1 < len(path) and res[j + 1] == ("block", rid):
                    j += 1
                anchors.append((i, j, rid))
                i = j + 1
            else:
                i += 1
        anchors = [(i, j, bid,
                    self._traverse_end(bid, path, i)) for i, j, bid in anchors]
        anchors.append((len(path) - 1, len(path) - 1, SINK, None))
        sections = []   # per anchor k: (blocks, node_movs, edges) to anchor k
        starts = [-1] + [a[1] for a in anchors[:-1]]
        for k in range(len(anchors)):
            lo = starts[k] + 1
            hi = anchors[k][0] if anchors[k][2] != SINK else len(path)
            blocks, node_movs, edges, seen = [], [], [], None
            for idx in range(lo, hi):
                kind, rid = res[idx]
                if kind == "node":
                    node_movs.append((rid, node_movement(path, idx)))
                    # node-node edge: a directional resource (head-on guard)
                    if idx > 0 and res[idx - 1][0] == "node":
                        a_cell, b_cell = res[idx - 1][1], rid
                        key = (a_cell, b_cell) if a_cell <= b_cell \
                            else (b_cell, a_cell)
                        edges.append((key, a_cell <= b_cell, b_cell))
                    seen = None
                    continue
                if rid == seen:
                    continue
                seen = rid
                blocks.append((rid, self._traverse_end(rid, path, idx)))
            sections.append((blocks, node_movs, edges))
        return dict(path=path, res=res, stops=stops,
                    anchors=anchors, sections=sections)

    def _traverse_end(self, bid, path, idx):
        b = self.bg.blocks[bid]
        chain = b.chain
        ci = chain.index(path[idx][0])
        cs = set(chain)
        # balloon loop: both ends attach to the SAME junction, so exit-node
        # identity cannot tell directions apart -- two OPPOSING traversals
        # would both label "B", the ledger would treat them co-directional,
        # and they meet head-on inside the loop (observed: L0_s0 deadlock
        # h1 vs h7). Chain-index movement is the only safe label here.
        if b.ends[0] is not None and b.ends[0] == b.ends[1]:
            if idx + 1 < len(path) and path[idx + 1][0] in cs:
                return "B" if chain.index(path[idx + 1][0]) > ci else "A"
            return "B" if ci == 0 else "A"
        # find next path cell outside the block to decide which end we exit
        j = idx
        while j < len(path) and path[j][0] in cs:
            j += 1
        if j < len(path):
            nxt = path[j][0]
            if nxt == b.ends[1]:
                return "B"
            if nxt == b.ends[0]:
                return "A"
        # fall back on chain index movement
        if idx + 1 < len(path) and path[idx + 1][0] in cs:
            return "B" if chain.index(path[idx + 1][0]) > ci else "A"
        return "B"

    # -- per-step main entry -------------------------------------------------
    def act(self):
        env, bg, led = self.env, self.bg, self.ledger
        occ = {}
        for a in env.agents:
            if a.position is not None:
                occ[tuple(a.position)] = a.handle

        # 1. sync pointers, physical presence, releases, DONE cleanup
        led.phys = {}
        for h, st in self.state.items():
            a = env.agents[h]
            if a.state == TrainState.DONE:
                if st["slot"] is not None or st["claims"] or st["pending"] \
                        or st["edges_held"]:
                    led.drop_all(h)
                    st["slot"], st["claims"], st["pending"] = None, [], []
                    st["edges_held"] = []
                continue
            if a.position is None:
                continue
            p = self.plans[h]
            pos = tuple(a.position)
            ptr = st["ptr"]
            while ptr < len(p["path"]) and p["path"][ptr][0] != pos:
                ptr += 1
            if ptr >= len(p["path"]):                  # should not happen
                ptr = st["ptr"]
            if ptr != st["ptr"]:
                self._on_advance(h, st["ptr"], ptr)
            st["ptr"] = ptr
            kind, rid = p["res"][ptr]
            if kind == "block":
                led.phys.setdefault(rid, set()).add(h)
                if st["slot"] == rid and bg.blocks[rid].is_bay:
                    st["last_bay"], st["last_bay_ptr"] = rid, ptr

        if self.SPAWN_CLEAR or self.SITTER_SHUFFLE:
            self._spawn_cells_now = {
                tuple(env.agents[hh].initial_position)
                for hh, ss in self.state.items()
                if not ss["departed"]
                and env.agents[hh].state != TrainState.DONE}

        # 2. grant rounds until fixpoint, highest score-pressure first:
        # a train's pressure = projected steps past latest_arrival. The
        # train bleeding the most points wins contested resources.
        self.t += 1
        led.t = self.t

        def pressure(h):
            a = env.agents[h]
            p = self.plans[h]
            remaining = len(p["path"]) - 1 - self.state[h]["ptr"]
            if self.PRESSURE_TIME_UNITS:
                remaining = remaining / (a.speed_counter.speed or 1.0)
            # fast-first tiebreak: within similar urgency, release faster
            # trains into a corridor before slower ones (never trap a fast
            # train behind a crawler it could have preceded)
            return (self.t + remaining - a.latest_arrival
                    + self.SPEED_TIEBREAK * a.speed_counter.speed)

        # NOTE: shortest-journey-first depot release was tested and is
        # WORSE (44/320 vs 57/320, norm -0.019): long-route trains need
        # early release to have any chance; short ones complete anyway.
        order = self._grant_order(pressure)
        self._recover_left = self.RECOVER_BUDGET
        # transit-cap census: total on-map population (parked + moving).
        # Measured: counting only corridor users floods the map (60 movers
        # + ~100 parked) and collapses completion 17.8% -> 3.4% on the
        # 320-agent proxy. The population cap is load-bearing. Hoisted out
        # of the grant loop (was O(n^2) per round) and updated on grants.
        in_corridor = sum(
            1 for hh, ss in self.state.items()
            if ss["departed"] and env.agents[hh].state != TrainState.DONE)
        slow_in_transit = sum(
            1 for hh, ss in self.state.items()
            if ss["departed"] and env.agents[hh].state != TrainState.DONE
            and env.agents[hh].speed_counter.speed <= self.SLOW_SPEED)
        changed = True
        while changed:
            changed = False
            for h in order:
                st = self.state[h]
                a = env.agents[h]
                if a.state == TrainState.DONE:
                    continue
                p = self.plans[h]
                if not st["departed"]:
                    if a.state != TrainState.READY_TO_DEPART:
                        continue
                    if tuple(a.initial_position) in occ:
                        continue
                    if not self._depart_worthwhile(h):
                        continue
                    sp = len(p["path"]) - 1
                    last_moment = self.env._max_episode_steps - sp \
                        - self.FORCE_MARGIN
                    if self.transit_cap is not None:
                        if in_corridor >= self.transit_cap \
                                and self.t < last_moment:
                            continue
                    is_slow = a.speed_counter.speed <= self.SLOW_SPEED
                    if self.SLOW_CAP is not None and is_slow \
                            and slow_in_transit >= self.SLOW_CAP \
                            and self.t < last_moment:
                        continue   # corridor-hog quota; deadline bypasses
                    if self._claim_to_anchor(h, 0):
                        st["departed"] = True
                        in_corridor += 1
                        slow_in_transit += 1 if is_slow else 0
                        changed = True
                    elif self._sit_pays(h):
                        # last-resort: enter the map unclaimed and sit at
                        # the spawn cell (frontier -1 blocks all movement;
                        # occ shields it). If traffic drains it may still
                        # claim out via the slot-None branch next steps.
                        st["departed"] = True
                        in_corridor += 1
                        slow_in_transit += 1 if is_slow else 0
                        changed = True
                    continue
                k = st["anchor"]
                if st["slot"] == PSW_SLOT:           # partial-section waiting
                    if self._psw_upgrade(h):
                        st["stuck"] = 0
                        changed = True
                    continue
                if st["slot"] is None and st["departed"]:
                    if self._claim_to_anchor(h, k):
                        st["stuck"] = 0
                        changed = True
                    continue
                if k + 1 < len(p["anchors"]) and st["slot"] is not None:
                    span = p["anchors"][k]
                    if span[0] <= st["ptr"] <= span[1] and \
                            self._front_of_queue(h, st["ptr"], span[1], occ):
                        if self._claim_to_anchor(h, k + 1):
                            if p["anchors"][k + 1][2] != span[2]:
                                st["pending"].append(span[2])
                            st["anchor"] = k + 1
                            st["stuck"] = 0
                            st["replans"] = 0
                            changed = True
                        elif self.PSW_ENABLED and self._psw_advance(h, k):
                            st["stuck"] = 0     # advanced into section; freed bay
                            changed = True
                        else:
                            st["stuck"] += 1
                            # cause-aware cadence: parked blockers are static
                            # (recover fast); flowing traffic drains (wait)
                            blocks, node_movs, _edges = p["sections"][k + 1]
                            static = any(
                                led.standing_in(bid) - {h} for bid, _ in blocks)
                            bay_n = p["anchors"][k + 1][2]
                            frozen = (
                                self.FROZEN_REPAIR_ENABLED and
                                len(env.agents[h].waypoints) <= self.MAX_WP_DENSE
                                and self._section_has_frozen_blocker(
                                    h, blocks, node_movs, bay_n)
                            )
                            if not static:
                                if bay_n != SINK and not led._slot_free(
                                        bay_n, p["anchors"][k + 1][3], h):
                                    static = bool(
                                        led.slot_holders(bay_n) - {h})
                            cadence = (self.FROZEN_RECOVER_EVERY if frozen
                                       else self.RECOVER_EVERY if static
                                       else self.RECOVER_EVERY * 4)
                            if st["stuck"] >= cadence \
                                    and self._recover_left > 0:
                                self._recover_left -= 1
                                st["stuck"] = 0          # flat cadence: retry
                                if frozen:
                                    moved = (
                                        self._replan_from_bay(h, avoid_frozen=True)
                                        or self._shunt(h, avoid_frozen=True)
                                        or self._replan_from_bay(h)
                                    )
                                else:
                                    mover = (self._shunt if st["replans"] % 2 == 1
                                             else self._replan_from_bay)
                                    moved = (mover(h)
                                             or (mover is self._shunt
                                                 and self._replan_from_bay(h)))
                                if (moved or (st["replans"] >= 3
                                              and self._consider_drop_stops(h))):
                                    st["replans"] += 1
                                    changed = True
                                else:
                                    st["replans"] += 1   # rotate strategy

        # 3. emit actions
        actions = {}
        for h in range(len(env.agents)):
            a = env.agents[h]
            if h not in self.plans or a.state == TrainState.DONE:
                actions[h] = A.DO_NOTHING
                continue
            st, p = self.state[h], self.plans[h]
            if a.position is None:
                actions[h] = (A.MOVE_FORWARD if st["departed"] else A.DO_NOTHING)
                continue
            if a.malfunction_handler.malfunction_down_counter > 0:
                actions[h] = A.DO_NOTHING
                continue
            ptr = st["ptr"]
            kind_here, rid_here = p["res"][ptr]
            on_bay = kind_here == "block" and self.bg.blocks[rid_here].is_bay
            if (not on_bay and ptr + 1 < len(p["path"])
                    and ptr + 1 > st["frontier"]):
                st["midstuck"] = st.get("midstuck", 0) + 1
                if st["midstuck"] >= self.REPLAN_AFTER:
                    if self._retreat_to_bay(h):
                        st["midstuck"] = 0
            else:
                st["midstuck"] = 0
            # mandatory 1-step dwell at the serving platform
            if ptr in p["stops"] and p["stops"][ptr] not in st["served"]:
                if a.state == TrainState.STOPPED:
                    st["served"].add(p["stops"][ptr])
                else:
                    actions[h] = A.STOP_MOVING
                    continue
            if ptr + 1 >= len(p["path"]):
                actions[h] = A.STOP_MOVING
                continue
            if (self.SITTER_SHUFFLE and st["frontier"] == -1
                    and not st["claims"] and st["slot"] is None):
                mv = self._sitter_shuffle_action(h, ptr, occ)
                if mv is not None:
                    actions[h] = mv
                    continue
            nxt_cell, nxt_dir = p["path"][ptr + 1]
            if not self._may_step(h, ptr, occ, nxt_cell):
                actions[h] = A.STOP_MOVING
                continue
            actions[h] = self._action_to(p["path"][ptr], nxt_dir)
        return actions

    def _sitter_shuffle_action(self, h, ptr, occ):
        """A converted spawn-sitter vacates a shared depot cell: one cell
        along its own block, only if that cell is free, not a node, not
        another spawn, and the block carries no transit claims."""
        p = self.plans[h]
        pos = p["path"][ptr][0]
        if pos not in self._spawn_cells_now:
            return None                  # nobody needs this cell
        if ptr + 1 >= len(p["path"]):
            return None
        nxt_cell, nxt_dir = p["path"][ptr + 1]
        if nxt_cell in occ or nxt_cell in self._spawn_cells_now:
            return None
        kind, rid = self.bg.resource_of(nxt_cell)
        if kind != "block" or self.bg.block_of.get(pos) != rid:
            return None                  # stay within the depot block
        t = self.ledger.transit.get(rid)
        if t and t[1]:
            return None                  # block in use as a through path
        return self._action_to(p["path"][ptr], nxt_dir)

    def _frozen_resources(self, h):
        """Blocks/cells held by other malfunctioning trains, mapped to the
        REMAINING downtime of the longest-down blocker (the malfunction
        clock is exact, so wait-vs-detour can be priced in time units)."""
        frozen_blocks, frozen_cells = {}, {}
        for a in self.env.agents:
            if a.handle == h:
                continue
            down = a.malfunction_handler.malfunction_down_counter
            if down <= 0:
                continue
            st = self.state.get(a.handle)
            if st:
                for bid in st.get("claims", ()):
                    frozen_blocks[bid] = max(frozen_blocks.get(bid, 0), down)
                if st.get("slot") not in (None, SINK):
                    bid = st["slot"]
                    frozen_blocks[bid] = max(frozen_blocks.get(bid, 0), down)
            if a.position is None:
                continue
            cell = tuple(a.position)
            frozen_cells[cell] = max(frozen_cells.get(cell, 0), down)
            kind, rid = self.bg.resource_of(cell)
            if kind == "block":
                frozen_blocks[rid] = max(frozen_blocks.get(rid, 0), down)
        return frozen_blocks, frozen_cells

    def _section_has_frozen_blocker(self, h, blocks, node_movs, bay_bid):
        """A blocker counts only if it stays down longer than
        FROZEN_MIN_DOWN: replanning around a train that recovers in a few
        steps wastes recovery budget (waiting is cheaper than any detour)."""
        frozen_blocks, frozen_cells = self._frozen_resources(h)
        thr = self.FROZEN_MIN_DOWN
        if any(frozen_blocks.get(bid, 0) > thr for bid, _ in blocks):
            return True
        if bay_bid != SINK and frozen_blocks.get(bay_bid, 0) > thr:
            return True
        return any(frozen_cells.get(cell, 0) > thr for cell, _ in node_movs)

    def _frozen_extra(self, h, weight=250):
        """Detour pricing around malfunctioning trains. Time-aware mode
        prices a frozen cell at ~2x the blocker's remaining downtime
        (wait cost + queue re-formation), so short outages keep the direct
        route (wait) and long outages force the detour."""
        frozen_blocks, frozen_cells = self._frozen_resources(h)
        extra = {}

        def w(down):
            if not self.FROZEN_TIME_AWARE:
                return weight
            return max(25, min(2 * down, 400))

        for bid, down in frozen_blocks.items():
            for c in self.bg.blocks[bid].chain:
                extra[c] = max(extra.get(c, 0), w(down))
        for c, down in frozen_cells.items():
            extra[c] = max(extra.get(c, 0), w(down))
        return extra

    def _dyn_extra(self, h, scale=1, avoid_frozen=False):
        led, bg = self.ledger, self.bg
        extra = {}
        for g in bg.groups:
            used = [m for m in g.blocks if led.standing_in(m) - {h}]
            if len(used) >= g.max_standing_blocks:
                for m in g.blocks:
                    if m not in used:
                        for c in bg.blocks[m].chain:
                            extra[c] = extra.get(c, 0) + 10
        for b in bg.blocks:
            standers = led.standing_in(b.bid) - {h}
            full = b.is_bay and len(led.slot_holders(b.bid) |
                                    set(led.phys.get(b.bid, ())) - {h}) \
                >= b.capacity
            if standers or full:
                for c in b.chain:
                    extra[c] = extra.get(c, 0) + 25
        for tcell, hs in self.targets.items():
            if hs - {h}:
                extra[tcell] = extra.get(tcell, 0) + 30
        out = {c: v * scale for c, v in extra.items()}
        if avoid_frozen:
            for c, v in self._frozen_extra(h).items():
                out[c] = out.get(c, 0) + v
        out.update(self.oneway)     # one-way designations: unscaled bias
        return out

    def _detour_ok(self, h, new_len):
        old = max(1, len(self.plans[h]["path"]) - 1 - self.state[h]["ptr"])
        return new_len <= max(old + 25, int(old * 1.35))

    def _remaining_gis(self, h):
        a, p, st = self.env.agents[h], self.plans[h], self.state[h]
        n = len(a.waypoints)
        gis = [gi for gi in range(1, n - 1)
               if gi in set(p["stops"].values()) and gi not in st["served"]]
        gis.append(n - 1)
        return gis

    def _bay_replan_ok(self, h):
        """Common guards: parked in own bay, holding nothing forward."""
        st, a, bg = self.state[h], self.env.agents[h], self.bg
        bid = st["slot"]
        if bid in (None, SINK) or a.position is None:
            return None
        if bg.block_of.get(tuple(a.position)) != bid:
            return None
        if st["claims"] or st["pending"] or st["edges_held"]:
            return None
        return bid

    def _install(self, h, path, stops):
        st, led = self.state[h], self.ledger
        bid = st["slot"]
        a = self.env.agents[h]
        new_plan = self._make_plan(a, path, stops)
        anchors = new_plan["anchors"]
        if not anchors or anchors[0][2] != bid:
            return False
        new_end = anchors[0][3]
        s = led.slots.get(bid)
        if s and s[0] != new_end:
            if set(s[1]) - {h}:
                return False
            s[0] = new_end
        self.plans[h] = new_plan
        st["ptr"], st["anchor"] = 0, 0
        st["frontier"] = anchors[0][1]
        return True

    # -- shunt: pull off into the nearest free bay purely to clear the way,
    # then continue the journey from there. The move that makes opposing
    # traffic able to time-share a single artery.
    def _shunt(self, h, avoid_frozen=False):
        bid = self._bay_replan_ok(h)
        if bid is None:
            return False
        a, led, bg = self.env.agents[h], self.ledger, self.bg
        st = self.state[h]
        extra = self._dyn_extra(h, 1 + st["replans"], avoid_frozen=avoid_frozen)
        foreign_targets = {t for t, hs in self.targets.items() if hs - {h}}
        goals = set()
        for b in bg.blocks:
            if not b.is_bay or b.bid == bid or b.kind == "stub":
                continue
            if len(led.standing_in(b.bid) - {h}) >= b.capacity:
                continue
            if led.transit.get(b.bid) and led.transit[b.bid][1]:
                continue
            if set(b.chain) & foreign_targets:
                continue        # never park on someone's destination
            if b.group is not None:
                g = bg.groups[b.group]
                used = [m for m in g.blocks if led.standing_in(m) - {h}]
                if len(used) >= g.max_standing_blocks:
                    continue    # never shunt into an already-parked station
            for c in b.chain:
                for d in range(4):
                    goals.add((c, d))
        if not goals:
            return False
        pos = (tuple(a.position), int(a.direction))
        leg = self.router._leg(pos, goals, extra)
        if leg is None:
            return False
        r, _ = self.router.route_from(a, leg[-1], self._remaining_gis(h), extra)
        if r is None:
            return False
        rpath, rstops = r
        path = leg + rpath[1:]
        if avoid_frozen and not self._detour_ok(h, len(path) - 1):
            return False
        off = len(leg) - 1
        stops = {i + off: gi for i, gi in rstops.items()}
        return self._install(h, path, stops)

    # -- retreat: a train stalled on through-track between bays reverses to
    # the last bay it departed, where it can legally stand and then re-route
    # or shunt. The path back is the path it just travelled, so it is clear
    # of standing trains by construction; we re-claim it in reverse and let
    # the normal grant loop walk it home.
    def _retreat_to_bay(self, h):
        st, p, led, bg = self.state[h], self.plans[h], self.ledger, self.bg
        a = self.env.agents[h]
        if st["last_bay"] is None or a.position is None:
            return False
        if st["claims"] or st["pending"] or st["edges_held"]:
            return False
        bay = st["last_bay"]
        back_ptr = st["last_bay_ptr"]
        ptr = st["ptr"]
        if back_ptr >= ptr:
            return False
        # build the reversed sub-path from current cell back into the bay,
        # with flipped headings; verify it is a legal directed walk.
        sub = p["path"][back_ptr:ptr + 1][::-1]
        rev = []
        for i, (cell, _) in enumerate(sub):
            if i + 1 < len(sub):
                nxt = sub[i + 1][0]
                d = self._heading(cell, nxt)
                if d is None:
                    return False
                rev.append((cell, d))
            else:
                rev.append((cell, sub[i - 1][1] if rev else 0))
        # the bay cell we re-enter and the remaining journey from it
        r, _ = self.router.route_from(a, rev[-1], self._remaining_gis(h))
        if r is None:
            return False
        fwd, fstops = r
        path = rev + fwd[1:]
        off = len(rev) - 1
        stops = {i + off: gi for i, gi in fstops.items()}
        new_plan = self._make_plan(a, path, stops)
        # must start where the train physically is
        if new_plan["path"][0][0] != tuple(a.position):
            return False
        old_slot = st["slot"]
        if old_slot is not None:
            led.release_slot(h, old_slot)
        self.plans[h] = new_plan
        st["ptr"], st["anchor"] = 0, 0
        st["slot"] = None
        st["frontier"] = -1
        st["departed"] = True
        st["replans"] += 1
        return True

    def _sp(self, start, goals):
        """Shortest-path length: O(1) lookup in the cached distance field."""
        return self.router.dist_field(goals).get(start)

    # ---- economics: when is departing worth it at all? -------------------
    def _depart_worthwhile(self, h):
        """Hold a train in the depot only when departing is provably worse
        than cancelling. All quantities in TIME steps, not cells: the
        cancellation penalty is 5 x shortest-path travel TIME (cells/speed),
        so for a 0.25-speed train the old cell-based estimate was 4x low.
        not-reached costs max(100, end-delay) + 50 per unserved stop."""
        a = self.env.agents[h]
        p = self.plans[h]
        speed = a.speed_counter.speed or 1.0
        sp_now = (len(p["path"]) - 1) / speed
        max_steps = self.env._max_episode_steps
        if self.t + sp_now <= max_steps:
            return True                       # can still make it: go
        cancel_pen = min(5 * sp_now, max_steps)
        n_stops = max(0, len(a.waypoints) - 2)
        fail_pen = min(max(100, max_steps + sp_now - a.latest_arrival)
                       + 50 * n_stops, max_steps)
        return fail_pen < cancel_pen

    def _sit_pays(self, h):
        """Endgame cancellation conversion: merely ENTERING the map turns
        the cancellation penalty (5 x shortest-path time, huge for slow
        trains) into target-not-reached (delay-based, no 5x factor).
        Measured on the 320 proxy: ~1,370 points saved per slow train,
        but a LOSS for fast short trains -- so gate on the actual margin.
        Only fires in the last SIT_WINDOW steps so an unclaimed squatter
        cannot block productive traffic for long."""
        if self.t < self.env._max_episode_steps - self.SIT_WINDOW:
            return False
        a = self.env.agents[h]
        p = self.plans[h]
        speed = a.speed_counter.speed or 1.0
        tt = (len(p["path"]) - 1) / speed
        ms = self.env._max_episode_steps
        cancel = min(5.0 * tt, ms)
        n_stops = max(0, len(a.waypoints) - 2)
        sit = min(max(100.0, ms + tt - a.latest_arrival) + 50.0 * n_stops, ms)
        return sit + 100.0 < cancel           # require a clear margin

    # ---- economics: drop stops when they cost more than -50 --------------
    def _consider_drop_stops(self, h):
        """Salvage rule: if serving the remaining stops makes the target
        unreachable within the horizon (not-reached >= 100 each), or the
        train is already past hope on its full route, drop remaining stops
        (-50 each) and run direct. Returns True if the plan changed."""
        st, p = self.state[h], self.plans[h]
        a = self.env.agents[h]
        remaining_stops = [gi for gi in self._remaining_gis(h)[:-1]]
        if not remaining_stops or a.position is None:
            return False
        full_remaining = len(p["path"]) - 1 - st["ptr"]
        if self.t + full_remaining <= self.env._max_episode_steps - 10:
            return False                      # still feasible with stops
        pos = (tuple(a.position), int(a.direction))
        wps = a.waypoints[-1]
        goals = set()
        for w in wps:
            cell = tuple(w.position)
            if w.direction is None:
                goals.update((cell, d) for d in range(4))
            else:
                goals.add((cell, int(w.direction)))
        direct = self._sp(pos, goals)
        if direct is None or self.t + direct > self.env._max_episode_steps:
            return False                      # direct doesn't save it either
        # dropping k stops costs 50k; missing target costs >= 100 + lateness
        bid = self._bay_replan_ok(h)
        if bid is None:
            return False
        r, _ = self.router.route_from(a, pos, [len(a.waypoints) - 1],
                                      self._dyn_extra(h))
        if r is None:
            return False
        path, stops = r
        if self._install(h, path, stops):
            st["served"].update(remaining_stops)   # mark dropped (no re-aim)
            return True
        return False

    def _heading(self, frm, to):
        dr, dc = to[0] - frm[0], to[1] - frm[1]
        for d, (ddr, ddc) in {0: (-1, 0), 1: (0, 1),
                              2: (1, 0), 3: (0, -1)}.items():
            if (ddr, ddc) == (dr, dc):
                return d
        return None

    # -- replan from a bay: the train is parked legally, so rerouting the
    # remaining legs is always safe. Penalise blocks occupied by others and
    # bays with no free capacity so the new route prefers free siblings.
    def _replan_from_bay(self, h, avoid_frozen=False):
        bid = self._bay_replan_ok(h)
        if bid is None:
            return False
        a, st = self.env.agents[h], self.state[h]
        extra = self._dyn_extra(h, 1 + st["replans"], avoid_frozen=avoid_frozen)
        start = (tuple(a.position), int(a.direction))
        gis = self._remaining_gis(h)
        r, _ = self.router.route_from(a, start, gis, extra)
        if avoid_frozen and r is not None and not self._detour_ok(h, len(r[0]) - 1):
            return False
        # marginal stop economics: when late, compare serving the remaining
        # stops against running direct. Serving costs extra path length
        # (1 lateness-point per step once past latest_arrival); skipping
        # costs the flat 50 per stop. Pick the cheaper plan.
        n_stops = len(gis) - 1
        if n_stops > 0:
            rd, _ = self.router.route_from(a, start, gis[-1:], extra)
            if rd is not None:
                la = a.latest_arrival
                if r is None:
                    r, dropped = rd, True
                else:
                    len_full = len(r[0]) - 1
                    len_dir = len(rd[0]) - 1
                    late_full = max(0, self.t + len_full - la)
                    late_dir = max(0, self.t + len_dir - la)
                    cost_full = float(late_full)
                    # intermediate late-arrival penalty: 0.5/step past each
                    # stop's own latest arrival (ECML2026Rewards)
                    wp_la = a.waypoints_latest_arrival
                    for pidx, gi in r[1].items():
                        la_i = wp_la[gi] if gi < len(wp_la) else None
                        if la_i is not None:
                            cost_full += 0.5 * max(0, self.t + pidx - la_i)
                    cost_dir = late_dir + 50.0 * n_stops
                    dropped = cost_dir < cost_full
                    if dropped:
                        r = rd
                if dropped and self._install(h, *r):
                    st["served"].update(gis[:-1])
                    return True
        if r is None:
            return False
        path, stops = r
        return self._install(h, path, stops)

    # -- helpers --------------------------------------------------------------
    def _front_of_queue(self, h, ptr, span_end, occ):
        """A train may claim ahead only if no other train stands between it
        and the exit of its current bay span. Holders of forward claims can
        therefore always physically reach their section, which is what makes
        every claim-holder able to progress (liveness)."""
        path = self.plans[h]["path"]
        for idx in range(ptr + 1, span_end + 1):
            cell = path[idx][0]
            if cell in occ and occ[cell] != h:
                return False
        return True

    # -- local core-pair signal ---------------------------------------------
    def _ensure_core_signals(self):
        """Detect overloaded bidirectional adjacent-bay pairs once, lazily.

        LayerVariant mutates plans after base construction, so this runs on the
        first grant attempt rather than in __init__. The signal is deliberately
        generic: it keys off bidirectional transition demand and capacity, not
        scenario tags or block ids.
        """
        if hasattr(self, "_core_pairs"):
            return
        self._core_pairs = set()
        self._core_pair_signal = {}
        self._core_pair_flips = Counter()
        self._core_pair_denied = Counter()
        if not self.CORE_SIGNAL_ENABLED:
            return
        low_density = len(self.env.agents) < 50
        min_each = (self.CORE_SIGNAL_LOW_MIN_EACH_DIR if low_density
                    else self.CORE_SIGNAL_MIN_EACH_DIR)
        min_total = (self.CORE_SIGNAL_LOW_MIN_TRANSITIONS if low_density
                     else self.CORE_SIGNAL_MIN_TRANSITIONS)
        min_load = (self.CORE_SIGNAL_LOW_MIN_LOAD if low_density
                    else self.CORE_SIGNAL_MIN_LOAD)

        dir_counts = Counter()
        for h, p in self.plans.items():
            bays = [a[2] for a in p["anchors"] if a[2] >= 0]
            for a, b in zip(bays, bays[1:]):
                if a == b:
                    continue
                pk = tuple(sorted((a, b)))
                dir_counts[(pk, (a, b))] += 1

        by_pair = defaultdict(dict)
        for (pk, direction), count in dir_counts.items():
            by_pair[pk][direction] = count
        for pk, dirs in by_pair.items():
            a, b = pk
            ab = dirs.get((a, b), 0)
            ba = dirs.get((b, a), 0)
            total = ab + ba
            if min(ab, ba) < min_each:
                continue
            cap = max(1, self.bg.blocks[a].capacity + self.bg.blocks[b].capacity)
            if total < min_total:
                continue
            if total / cap < min_load:
                continue
            self._core_pairs.add(pk)

    def _core_pair_key(self, a, b):
        if a is None or b is None or a < 0 or b < 0 or a == b:
            return None
        pk = tuple(sorted((a, b)))
        return pk if pk in self._core_pairs else None

    def _core_claim_intent(self, h, k):
        """Return (pair_key, direction, mode) if claiming anchor k would either
        cross a controlled pair or enter the source bay for that pair."""
        self._ensure_core_signals()
        if not self._core_pairs:
            return None, None, None
        p, st = self.plans[h], self.state[h]
        anchors = p["anchors"]
        if k >= len(anchors):
            return None, None, None
        target = anchors[k][2]
        if target < 0:
            return None, None, None

        curk = st["anchor"]
        if curk < len(anchors):
            src = anchors[curk][2]
            pk = self._core_pair_key(src, target)
            if pk is not None:
                return pk, (src, target), "cross"

        if k + 1 < len(anchors):
            nxt = anchors[k + 1][2]
            pk = self._core_pair_key(target, nxt)
            if pk is not None:
                return pk, (target, nxt), "enter"
        return None, None, None

    def _core_queue_counts(self, pk):
        a, b = pk
        dirs = ((a, b), (b, a))
        counts, ages, local = Counter(), Counter(), Counter()
        for h in self.state:
            st = self.state.get(h)
            if st is None or self.env.agents[h].state == TrainState.DONE:
                continue
            p = self.plans[h]
            curk = st["anchor"]
            if curk + 1 < len(p["anchors"]):
                src = p["anchors"][curk][2]
                dst = p["anchors"][curk + 1][2]
                if self._core_pair_key(src, dst) == pk:
                    counts[(src, dst)] += 1
                    ages[(src, dst)] += st.get("stuck", 0)
                    local[(src, dst)] += 1
            k = curk if st.get("slot") is None else curk + 1
            if k < len(p["anchors"]) - 1:
                src = p["anchors"][k][2]
                dst = p["anchors"][k + 1][2]
                if self._core_pair_key(src, dst) == pk:
                    counts[(src, dst)] += 1
        return dirs, counts, ages, local

    def _core_signal_drain_before_flip(self):
        # MEASURED: the drain-before-flip logic, when actually active on clean
        # L2_s2, REGRESSES 100 -> 70. The committed code only "worked" because
        # the <=0.0 gate disabled it (clean scenarios carry rate ~1e-9 from
        # malfunction_interval=1e9). So keep it OFF. Left in place, gated off,
        # rather than deleted, pending a better drain design.
        return False

    def _core_signal_allow(self, h, k):
        if self.state[h]["slot"] is not None and k == self.state[h]["anchor"]:
            return True
        pk, direction, mode = self._core_claim_intent(h, k)
        if pk is None:
            return True

        if self.CORE_SIGNAL_OCC_GATE:
            a_, b_ = pk
            occ = len(self.ledger.phys.get(a_, ())) \
                + len(self.ledger.phys.get(b_, ()))
            cap = self.bg.blocks[a_].capacity + self.bg.blocks[b_].capacity
            if occ < self.CORE_SIGNAL_ENFORCE_FRAC * cap:
                return True            # ample room: per-track claims keep it safe

        dirs, counts, ages, local = self._core_queue_counts(pk)
        sig = self._core_pair_signal.get(pk, (None, self.t, False))
        if len(sig) == 2:
            active, since = sig
            closing = False
        else:
            active, since, closing = sig
        if active is None:
            active = max(dirs, key=lambda d: (counts[d], ages[d], -d[0], -d[1]))
            since = self.t
            closing = False
            self._core_pair_signal[pk] = (active, since, closing)

        other = dirs[1] if active == dirs[0] else dirs[0]
        age = self.t - since
        phase_min = (self.CORE_SIGNAL_LOW_PHASE_MIN
                     if len(self.env.agents) < 50
                     else self.CORE_SIGNAL_PHASE_MIN)
        phase_max = (self.CORE_SIGNAL_LOW_PHASE_MAX
                     if len(self.env.agents) < 50
                     else self.CORE_SIGNAL_PHASE_MAX)

        if self._core_signal_drain_before_flip():
            should_flip = False
            if closing and local[active] == 0:
                if counts[other] > 0:
                    should_flip = True
                else:
                    closing = False
            elif not closing:
                if age >= phase_min and local[active] == 0 \
                        and counts[other] > 0:
                    should_flip = True
                elif age >= phase_max and counts[other] > 0:
                    closing = True
                elif age >= phase_min \
                        and counts[other] >= counts[active] + 3 \
                        and ages[other] >= ages[active] \
                        and local[active] == 0:
                    should_flip = True
            if should_flip:
                active = other
                since = self.t
                closing = False
                self._core_pair_flips[pk] += 1
            self._core_pair_signal[pk] = (active, since, closing)
        else:
            should_flip = False
            if age >= phase_min and counts[active] == 0 \
                    and counts[other] > 0:
                should_flip = True
            elif age >= phase_max and counts[other] > 0:
                should_flip = True
            elif age >= phase_min \
                    and counts[other] >= counts[active] + 3 \
                    and ages[other] >= ages[active]:
                should_flip = True

            if should_flip:
                active = other
                since = self.t
                closing = False
                self._core_pair_flips[pk] += 1
            self._core_pair_signal[pk] = (active, since, closing)

        if direction != active:
            self._core_pair_denied[(direction, active)] += 1
            return False
        if closing and mode == "enter":
            self._core_pair_denied[(direction, "closing")] += 1
            return False
        return True

    def _claim_to_anchor(self, h, k):
        p, led = self.plans[h], self.ledger
        if self.state[h]["slot"] is not None and k == self.state[h]["anchor"]:
            return True
        if not self._core_signal_allow(h, k):
            return False
        blocks, node_movs, edges = p["sections"][k]
        bay, bay_end = p["anchors"][k][2], p["anchors"][k][3]
        if led.try_claim(h, blocks, node_movs, bay, bay_end, edges):
            st = self.state[h]
            for bid, _ in blocks:
                st["claims"].append(bid)
            st["edges_held"].extend(edges)
            st["slot"] = bay
            span = p["anchors"][k]
            if bay == SINK:
                st["frontier"] = len(p["path"]) - 1
            else:
                fr = span[1]
                if self.SPAWN_CLEAR:
                    # stand short of foreign spawn cells when room exists
                    while fr > span[0] and \
                            p["path"][fr][0] in self._spawn_cells_now:
                        fr -= 1
                st["frontier"] = fr
            return True
        return False

    def _on_advance(self, h, old_ptr, new_ptr):
        """Release resources left behind between old_ptr and new_ptr."""
        p, st, led = self.plans[h], self.state[h], self.ledger
        for idx in range(old_ptr, new_ptr):
            kind, rid = p["res"][idx]
            nkind, nrid = p["res"][idx + 1]
            if (kind, rid) == (nkind, nrid):
                continue
            if kind == "node":
                led.release_node(h, rid, node_movement(p["path"], idx))
                # release node-node edges that END at this node
                kept = []
                for key, fwd, b_cell in st["edges_held"]:
                    if b_cell == rid:
                        led.release_edge(h, key)
                    else:
                        kept.append((key, fwd, b_cell))
                st["edges_held"] = kept
            elif kind == "block":
                if rid in st["claims"]:
                    st["claims"].remove(rid)
                    led.release_transit(h, rid)
                if rid in st["pending"]:
                    st["pending"].remove(rid)
                    led.release_slot(h, rid)

    # -- partial-section waiting -------------------------------------------
    def _psw_advance(self, h, k):
        """Front-of-queue train at bay k advances into section k+1 and waits on
        its claimed through-blocks (bay k+1 is full), freeing bay k. Returns
        True if it advanced. Deadlock-safe gate: target bay has no parked
        train (occupants are in-transit and will drain)."""
        p, led, st = self.plans[h], self.ledger, self.state[h]
        if k + 1 >= len(p["anchors"]):
            return False
        bay = p["anchors"][k + 1][2]
        if bay == SINK:
            return False
        blocks, node_movs, edges = p["sections"][k + 1]
        if not blocks:
            return False                      # nowhere to wait in the section
        if led.standing_in(bay) - {h}:
            return False                      # GATE: parked train -> cycle risk
        if not led.try_claim_psw(h, blocks, node_movs, edges):
            return False                      # head-on / occupied: deny
        for bid, _ in blocks:
            st["claims"].append(bid)
        st["edges_held"].extend(edges)
        old = st["slot"]
        if old not in (None, SINK, PSW_SLOT):
            led.release_slot(h, old)          # free the bay behind us
        st["slot"] = PSW_SLOT
        st["psw_anchor"] = k + 1
        st["frontier"] = p["anchors"][k + 1][0] - 1   # last cell before bay
        return True

    def _psw_upgrade(self, h):
        """A partial-section-waiting train tries to claim its target bay slot
        (now that it is parked just outside it). Promotes to a normal slot."""
        led, st, p = self.ledger, self.state[h], self.plans[h]
        k1 = st.get("psw_anchor")
        if k1 is None:
            return False
        bay, bay_end = p["anchors"][k1][2], p["anchors"][k1][3]
        if bay == SINK:
            st["slot"], st["anchor"] = SINK, k1
            st["frontier"] = len(p["path"]) - 1
            st.pop("psw_anchor", None)
            return True
        if not led._slot_free(bay, bay_end, h):
            return False
        s = led.slots.setdefault(bay, [bay_end, set()])
        s[0] = bay_end
        s[1].add(h)
        led.slot_of[h] = bay
        st["slot"], st["anchor"] = bay, k1
        st["frontier"] = p["anchors"][k1][1]
        st.pop("psw_anchor", None)
        return True

    def _may_step(self, h, ptr, occ, nxt_cell):
        if nxt_cell in occ and occ[nxt_cell] != h:
            return False
        return ptr + 1 <= self.state[h]["frontier"]

    def _action_to(self, cur, want_dir):
        d = cur[1]
        rel = (want_dir - d) % 4
        if rel == 1:
            return A.MOVE_RIGHT
        if rel == 3:
            return A.MOVE_LEFT
        return A.MOVE_FORWARD

    # -- invariants (gate 1) ---------------------------------------------------
    def check_invariants(self):
        env, led, bg = self.env, self.ledger, self.bg
        for h, st in self.state.items():
            a = env.agents[h]
            if a.state in (TrainState.DONE,) or a.position is None:
                continue
            if st["frontier"] == -1 and not st["claims"]:
                continue          # endgame spawn-sitter / retreating: unclaimed by design
            assert st["slot"] is not None, f"J1 train {h} on map without a bay slot"
        for bid, (end, holders) in led.slots.items():
            b = bg.blocks[bid]
            phys = led.phys.get(bid, set())
            assert len(led.standing_in(bid)) <= b.capacity, \
                f"J2 bay {bid}: parked {led.standing_in(bid)} > cap {b.capacity}"
            t = led.transit.get(bid)
            foreign = set(t[1]) - set(holders) - phys if t else set()
            assert not (holders and foreign), \
                f"J3 bay {bid} parked by {holders} and claimed by {foreign}"
        # J5 (keep-one-clear) intentionally not asserted: it is a routing
        # preference handled by penalties, not a safety invariant.
        for bid, t in led.transit.items():
            assert all(c > 0 for c in t[1].values()), f"J4 negative claim on {bid}"
        return True
