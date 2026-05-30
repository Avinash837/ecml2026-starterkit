"""
Offline-style topology preprocessing for the fixed competition map.

The competition runs one fixed 120x150 / 28-station map for all 35 scenarios, so the
structure can be analysed once and reused. This module extracts, from any RailEnv:

  * the directed cell graph         succ[(cell,dir)] -> [(ncell,ndir), ...]
  * junction / corridor / dead-end classification (undirected degree)
  * single-track corridor SEGMENTS (the resource that causes head-on deadlocks)
  * station platform cells (from agent timetable waypoints)
  * a cached reverse-BFS distance FIELD to any target cell -- the primitive that
    multi-stop routing needs and that flatland's per-agent distance_map does NOT give
    (distance_map only stores distance to each agent's FINAL target).

It runs once on first env (cache by a cheap rail fingerprint) and is then reused.
Validated against flatland's own distance_map for the final-target case.
"""
import heapq
import numpy as np
from collections import deque
from flatland.envs.rail_env_action import RailEnvActions

_MOVES = [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]
_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # N E S W


def _rail_fingerprint(rail):
    """Cheap hash of the grid so we rebuild only when the map actually changes."""
    g = rail.grid
    return (g.shape, int(g.sum()), int(g[::7, ::7].sum()))


class Topology:
    _cache = {}

    def __new__(cls, env):
        fp = _rail_fingerprint(env.rail)
        if fp in cls._cache:
            return cls._cache[fp]
        self = super().__new__(cls)
        self._fp = fp
        self._build(env)
        cls._cache[fp] = self
        return self

    # ---- construction --------------------------------------------------------
    def _build(self, env):
        rail = env.rail
        H, W = env.height, env.width
        self.H, self.W = H, W

        # directed transition graph
        succ = {}
        cells = set()
        for r in range(H):
            for c in range(W):
                if rail.get_full_transitions(r, c) == 0:
                    continue
                cells.add((r, c))
                for d in range(4):
                    nxt = []
                    seen = set()
                    for act in _MOVES:
                        res = rail.apply_action_independent(RailEnvActions.from_value(act), ((r, c), d))
                        if res is None:
                            continue
                        (np_, nd_), _ = res
                        if (np_, nd_) == ((r, c), d) or (np_, nd_) in seen:
                            continue
                        seen.add((np_, nd_))
                        nxt.append((tuple(np_), int(nd_)))
                    if nxt:
                        succ[((r, c), d)] = nxt
        self.succ = succ
        self.cells = cells

        # reverse graph for distance fields
        pred = {}
        for (cell, d), outs in succ.items():
            for (ncell, nd) in outs:
                pred.setdefault((ncell, nd), []).append((cell, d))
        self.pred = pred

        # undirected neighbour degree -> junction / corridor / dead-end
        undirected = {c: set() for c in cells}
        for (cell, d), outs in succ.items():
            for (ncell, nd) in outs:
                undirected[cell].add(ncell)
                undirected.setdefault(ncell, set()).add(cell)
        self.degree = {c: len(undirected[c]) for c in cells}
        self.undirected = undirected
        self.junctions = {c for c in cells if self.degree[c] >= 3}
        self.deadends = {c for c in cells if self.degree[c] == 1}

        # corridor segments: maximal chains of degree-2 cells between boundaries
        boundary = self.junctions | self.deadends
        seg_of = {}
        seg_cells = []
        seen = set()
        for start in cells:
            if start in boundary or start in seen:
                continue
            comp = []
            stack = [start]
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in undirected[cur]:
                    if nb not in boundary and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            sid = len(seg_cells)
            seg_cells.append(comp)
            for cc in comp:
                seg_of[cc] = sid
        self.seg_of = seg_of
        self.seg_cells = seg_cells
        self.n_segments = len(seg_cells)

        # stations = timetable waypoint positions (origins, intermediates, targets)
        stations = set()
        for a in env.agents:
            for wp_alts in a.waypoints:
                for wp in wp_alts:
                    stations.add(tuple(wp.position))
        self.stations = stations

        self._distfield_cache = {}

    # ---- distance field to an arbitrary target cell --------------------------
    def dist_field(self, target_cell):
        """
        Reverse-BFS (in directed-edge steps) giving, for every (cell,dir), the minimum
        number of moves to REACH `target_cell` in any orientation. Cached per target.
        Returns dict[(cell,dir)] -> int.
        """
        target_cell = tuple(target_cell)
        if target_cell in self._distfield_cache:
            return self._distfield_cache[target_cell]
        dist = {}
        q = deque()
        for d in range(4):
            key = (target_cell, d)
            if key in self.succ or key in self.pred:
                dist[key] = 0
                q.append(key)
        while q:
            cur = q.popleft()
            for prev in self.pred.get(cur, ()):  # who can step INTO cur
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    q.append(prev)
        self._distfield_cache[target_cell] = dist
        return dist

    def distance(self, cell, direction, target_cell):
        return self.dist_field(target_cell).get((tuple(cell), int(direction)), np.inf)

    def route(self, start_cell, start_dir, target_cells):
        """
        Shortest directed path from (start_cell,start_dir) to ANY cell in
        `target_cells` (a stop may have several platform alternatives). Returns a list
        of (cell,dir) including both endpoints, or None if unreachable. Greedy descent
        on the cached distance field -> one shortest path, O(path length).
        """
        target_cells = {tuple(t) for t in target_cells}
        # Combined field: min distance over all alternative target cells.
        fields = [self.dist_field(t) for t in target_cells]

        def dval(key):
            return min((f.get(key, np.inf) for f in fields), default=np.inf)

        cur = (tuple(start_cell), int(start_dir))
        if dval(cur) == np.inf:
            return None
        path = [cur]
        guard = 0
        limit = self.H * self.W * 2 + 10
        while cur[0] not in target_cells:
            outs = self.succ.get(cur, [])
            nxt = min(outs, key=dval, default=None)
            if nxt is None or dval(nxt) == np.inf or dval(nxt) >= dval(cur) + 1e9:
                return None
            path.append(nxt)
            cur = nxt
            guard += 1
            if guard > limit:
                return None
        return path

    def k_routes(self, start_cell, start_dir, target_cells, K=3, pw=0.5, max_ratio=1.3):
        """Up to K DIVERSE shortest-ish routes via penalized Dijkstra on the directed cell
        graph: after each route, add a cost penalty to the cells it used so the next route
        prefers different corridors. Cheap (~ms; graph is small) -- the scalable replacement
        for flatland's get_k_shortest_paths (~3.5s/agent). Returns list of (cell,dir) paths."""
        target_cells = {tuple(t) for t in target_cells}
        start = (tuple(start_cell), int(start_dir))
        if start not in self.succ:
            return []
        penalties = {}
        routes = []
        seen = set()
        for _ in range(K):
            dist = {start: 0.0}
            prev = {}
            pq = [(0.0, start)]
            goal = None
            while pq:
                d, node = heapq.heappop(pq)
                if d > dist.get(node, np.inf):
                    continue
                if node[0] in target_cells:
                    goal = node
                    break
                for nb in self.succ.get(node, ()):
                    nd = d + 1.0 + pw * penalties.get(nb[0], 0)
                    if nd < dist.get(nb, np.inf):
                        dist[nb] = nd
                        prev[nb] = node
                        heapq.heappush(pq, (nd, nb))
            if goal is None:
                break
            path = []
            cur = goal
            while cur != start:
                path.append(cur)
                cur = prev[cur]
            path.append(start)
            path.reverse()
            for c, _ in path:
                penalties[c] = penalties.get(c, 0) + 1
            sig = tuple(c for c, _ in path)
            if sig in seen:
                continue                     # penalized but duplicate; try once more
            # keep only near-shortest alternatives; long detours arrive late and hurt
            if routes and len(path) > max_ratio * len(routes[0]):
                continue
            seen.add(sig)
            routes.append(path)
        if not routes:
            r = self.route(start_cell, start_dir, target_cells)
            if r:
                routes = [r]
        return routes

    def multistop_route(self, agent):
        """
        Full path threading every timetable waypoint in order:
        origin -> stop_1 -> ... -> target. Returns (path, stop_indices) where
        path is a contiguous list of (cell,dir) and stop_indices marks the path index
        at which each intermediate/target waypoint is reached. None if any leg fails.
        """
        wp_alt_cells = [{tuple(w.position) for w in alts} for alts in agent.waypoints]
        cur_cell = tuple(agent.initial_position)
        cur_dir = int(agent.initial_direction)
        full = [(cur_cell, cur_dir)]
        stop_idx = []
        for alts in wp_alt_cells[1:]:                       # skip origin
            leg = self.route(cur_cell, cur_dir, alts)
            if leg is None:
                return None, None
            full.extend(leg[1:])                            # drop duplicated junction cell
            stop_idx.append(len(full) - 1)
            cur_cell, cur_dir = full[-1]
        return full, stop_idx

    def stats(self):
        seg_lens = [len(s) for s in self.seg_cells]
        return dict(
            cells=len(self.cells),
            junctions=len(self.junctions),
            deadends=len(self.deadends),
            segments=self.n_segments,
            stations=len(self.stations),
            longest_corridor=max(seg_lens) if seg_lens else 0,
            mean_corridor=round(float(np.mean(seg_lens)), 2) if seg_lens else 0,
        )


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")
    from eval_harness import make_env

    print("=== Topology extraction sanity check ===\n")

    # 1) competition-scale generated map
    env = make_env(0, 1, seed=1, map_cfg=dict(x_dim=100, y_dim=80, n_cities=10))
    env.reset(random_seed=1)
    t = time.time()
    topo = Topology(env)
    build_s = time.time() - t
    print(f"generated 80x100/10-city map  (build {build_s*1000:.0f} ms)")
    for k, v in topo.stats().items():
        print(f"    {k:16s} {v}")

    # cache hit on rebuild
    t = time.time()
    Topology(env)
    print(f"    rebuild (cached)  {(time.time()-t)*1e6:.0f} us")

    # 2) validate dist_field against flatland's own distance_map for an agent target
    dm = env.distance_map.get()
    bad = 0
    checked = 0
    for h, a in enumerate(env.agents[:5]):
        df = topo.dist_field(a.target)
        p = tuple(a.initial_position)
        d = int(a.initial_direction)
        mine = df.get((p, d), np.inf)
        theirs = dm[h, p[0], p[1], d]
        ok = (np.isinf(mine) and not np.isfinite(theirs)) or abs(mine - theirs) < 1e-6
        checked += 1
        bad += (not ok)
        print(f"    agent {h}: my dist={mine}  flatland dist={theirs}  {'OK' if ok else 'MISMATCH'}")
    print(f"  distance-field validation: {checked-bad}/{checked} match flatland distance_map")

    # 3) debug pkl
    from flatland.envs.persistence import RailEnvPersister
    denv, _ = RailEnvPersister.load_new("debug-environments/debug-environments/Test_0/Level_1.pkl")
    denv.reset(random_seed=1)
    dtopo = Topology(denv)
    print("\ndebug Test_0/Level_1 (25x25):")
    for k, v in dtopo.stats().items():
        print(f"    {k:16s} {v}")
