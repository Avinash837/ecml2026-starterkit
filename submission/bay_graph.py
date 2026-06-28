"""
Bay graph: contraction of a Flatland rail grid into operational resources.

Resources
---------
NODE resources (capacity 1, transit-only, never a standing spot):
  * junction  -- a cell where some incoming direction has >= 2 outgoing
                 transitions (all switch types: simple, symmetric, slips).
  * crossing  -- a degree-4 cell where NO incoming direction has a choice
                 (diamond crossing): two independent corridors sharing one
                 cell. Functionally a corridor cell of two corridors at once,
                 so it must be its own capacity-1 resource.

BLOCK resources (segments): maximal chains of plain cells (degree <= 2, no
choice) between node resources. A block is single-track: it may be occupied
in one direction of travel at a time.
  * through   -- both ends attach to node resources, no parallel sibling.
  * loop      -- belongs to a parallel group (>= 2 routes between the same
                 node pair, counting zero-length direct links). Longer loops
                 are usable standing bays; tiny loops are treated as passing
                 lanes only, because parking there creates junction queues.
  * stub      -- one end is a dead end. Standing here blocks nobody.
  * isolated  -- a closed ring with no node resource on it (rare).

BAYS = places a train may stand indefinitely without cutting through-traffic:
every long-enough 'loop' or 'stub' block, plus virtual depot bays (off-map
trains) and sink bays (targets; remove_agents_at_target frees the cell). A
platform on a 'through' block is dwellable (1-step stop) but NOT a standing bay.

Parallel groups carry the keep-one-clear rule: standing trains may occupy at
most (n_routes - 1) sibling blocks of a group, so one route always stays
clear. n_routes counts zero-length junction-junction direct links, which can
never hold a standing train but always provide the clear route.

Everything is plain-Python picklable. Build once per map (cached by rail
fingerprint), ship as a pickle.
"""
from collections import deque
import hashlib

OPP = {0: 2, 1: 3, 2: 0, 3: 1}
DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # N E S W
MIN_LOOP_BAY_LEN = 4


def _neighbour(cell, d):
    dr, dc = DELTA[d]
    return (cell[0] + dr, cell[1] + dc)


def _fingerprint(rail):
    g = rail.grid
    return (tuple(g.shape), str(g.dtype),
            hashlib.blake2b(g.tobytes(), digest_size=16).hexdigest())


class Block:
    __slots__ = ("bid", "chain", "ends", "kind", "is_bay", "capacity",
                 "group", "dir_ok", "platforms")

    def __init__(self, bid, chain, ends):
        self.bid = bid
        self.chain = chain          # ordered list of cells, end A first
        self.ends = ends            # (node cell or None, node cell or None)
        self.kind = "through"       # through | loop | stub | isolated
        self.is_bay = False
        self.capacity = len(chain)  # standing trains it can hold (1 per cell)
        self.group = None           # parallel-group id, if any
        self.dir_ok = set()         # subset of {"AB", "BA"}: traversable ways
        self.platforms = set()      # station platform cells inside this block

    def end_toward(self, idx_from, idx_to):
        """Which end ('A'/'B') a train walking chain[idx_from]->chain[idx_to]
        is heading for."""
        return "B" if idx_to > idx_from else "A"


class ParallelGroup:
    __slots__ = ("gid", "ends", "blocks", "direct_links", "n_routes",
                 "max_standing_blocks")

    def __init__(self, gid, ends, blocks, direct_links):
        self.gid = gid
        self.ends = ends                       # frozenset of node cells
        self.blocks = blocks                   # bids of member blocks
        self.direct_links = direct_links       # count of zero-length links
        self.n_routes = len(blocks) + direct_links
        # keep-one-clear: standing trains in at most n_routes-1 members
        self.max_standing_blocks = self.n_routes - 1


class BayGraph:
    _cache = {}
    # BOUND the cross-map cache: over a multi-level eval run the orchestrator
    # feeds many different maps to ONE long-lived process. An unbounded cache
    # would hold a full bay-graph per map and grow without limit -> the
    # container bloats and gets OOM-killed at a level boundary (uncatchable, no
    # score). The real competition map is FIXED -> 1 entry, always reused; a
    # changing map set stays capped here. Keep just the most-recent few.
    _CACHE_MAX = 2

    def __new__(cls, env):
        fp = _fingerprint(env.rail)
        hit = cls._cache.get(fp)
        if hit is not None:
            hit._refresh_stations(env)
            return hit
        self = super().__new__(cls)
        self._fp = fp
        self._build(env)
        while len(cls._cache) >= cls._CACHE_MAX:   # FIFO-evict stale maps
            cls._cache.pop(next(iter(cls._cache)))
        cls._cache[fp] = self
        return self

    def __reduce__(self):
        return (_restore_baygraph, (self.__dict__.copy(),))

    # ------------------------------------------------------------------ build
    def _build(self, env):
        rail = env.rail
        H, W = env.height, env.width
        self.H, self.W = H, W

        # 1. directed graph over (cell, dir) straight from the transition maps
        succ, cells = {}, set()
        for r in range(H):
            for c in range(W):
                if rail.get_full_transitions(r, c) == 0:
                    continue
                cell = (r, c)
                cells.add(cell)
                for d in range(4):
                    tr = rail.get_transitions((cell, d))
                    outs = [( _neighbour(cell, nd), nd) for nd in range(4) if tr[nd]]
                    if outs:
                        succ[(cell, d)] = outs
        pred = {}
        for k, outs in succ.items():
            for nb in outs:
                pred.setdefault(nb, []).append(k)
        self.succ, self.pred, self.cells = succ, pred, cells

        # 2. undirected adjacency, degree, decision flags
        und = {c: set() for c in cells}
        for (cell, _), outs in succ.items():
            for ncell, _ in outs:
                und[cell].add(ncell)
                und.setdefault(ncell, set()).add(cell)
        self.und = und
        self.degree = {c: len(und[c]) for c in cells}
        has_choice = {c: False for c in cells}
        for (cell, _), outs in succ.items():
            if len({nb for nb in outs}) >= 2:
                has_choice[cell] = True

        # 3. node resources: junctions (any choice) and pure crossings
        self.junctions = {c for c in cells if has_choice[c]}
        self.crossings = {c for c in cells
                          if not has_choice[c] and self.degree[c] >= 3}
        self.nodes = self.junctions | self.crossings
        self.deadend_cells = {c for c in cells if self.degree[c] == 1}

        # 4. blocks: ordered chains of plain cells between node resources
        plain = cells - self.nodes
        self.blocks, self.block_of = [], {}
        visited = set()
        for start in sorted(plain):
            if start in visited:
                continue
            comp = self._component(start, plain, visited)
            chain, ends = self._order_chain(comp)
            b = Block(len(self.blocks), chain, ends)
            for cc in chain:
                self.block_of[cc] = b.bid
            self.blocks.append(b)

        # 5. zero-length node-node direct links (adjacent node resources)
        direct = {}
        for n in self.nodes:
            for nb in und[n]:
                if nb in self.nodes:
                    key = frozenset((n, nb))
                    if len(key) == 2:
                        direct[key] = direct.get(key, 0) + 1
        self.direct_links = {k: v // 2 for k, v in direct.items()}  # both saw it

        # 6. traversable directions per block (one-way corridors exist)
        for b in self.blocks:
            b.dir_ok = self._traversable(b)

        # 7. classify: stubs, isolated rings, parallel groups -> loop bays
        by_ends = {}
        for b in self.blocks:
            a_end, b_end = b.ends
            if a_end is None and b_end is None:
                if all(self.degree[c] == 2 for c in b.chain):
                    b.kind = "isolated"            # true closed ring
                else:
                    b.kind, b.is_bay = "stub", True  # shuttle: both ends dead
                continue
            if a_end is None or b_end is None:
                b.kind, b.is_bay = "stub", True
                continue
            by_ends.setdefault(frozenset((a_end, b_end)), []).append(b.bid)
        self.groups = []
        for ends, bids in sorted(by_ends.items(), key=lambda kv: sorted(kv[1])):
            dl = self.direct_links.get(ends, 0)
            if len(bids) + dl >= 2:
                g = ParallelGroup(len(self.groups), ends, sorted(bids), dl)
                self.groups.append(g)
                for bid in bids:
                    blk = self.blocks[bid]
                    blk.kind, blk.group = "loop", g.gid
                    blk.is_bay = len(blk.chain) >= MIN_LOOP_BAY_LEN

        # 8. station platforms and sinks from the timetable
        self._registered_platform_cells = set()
        self.platform_cells = set()
        self.sink_cells = set()
        self._refresh_stations(env)

    def _component(self, start, allowed, visited):
        comp, q = [start], deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            for nb in self.und[cur]:
                if nb in allowed and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
                    comp.append(nb)
        return comp

    def _order_chain(self, comp):
        """Order a plain-cell component into a path; return (chain, ends)."""
        compset = set(comp)
        und = self.und
        # endpoints: plain cells with <=1 plain neighbour inside the component
        endpoints = [c for c in comp
                     if len(und[c] & compset) <= 1]
        if not endpoints:                      # closed ring or single loop cell
            endpoints = [min(comp)]
        start = min(endpoints)
        chain, seen, cur = [start], {start}, start
        while True:
            nxt = [n for n in und[cur] & compset if n not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            chain.append(cur)
        end_a = self._attached_node(chain[0], chain)
        end_b = self._attached_node(chain[-1], chain)
        if len(chain) > 1 and end_a == end_b and end_a is not None:
            # balloon loop: both chain ends touch the same junction -- fine
            pass
        return chain, (end_a, end_b)

    def _attached_node(self, cell, chain):
        outside = self.und[cell] - set(chain)
        for nb in outside:
            if nb in self.nodes:
                return nb
        return None                            # dead end (or isolated ring)

    def _traversable(self, b):
        ok = set()
        ch = b.chain
        if len(ch) == 1:
            cell = ch[0]
            for d in range(4):
                for ncell, _ in self.succ.get((cell, d), ()):
                    if ncell == b.ends[1] or (b.ends[1] is None and ncell not in ch):
                        ok.add("AB")
                    if ncell == b.ends[0] or (b.ends[0] is None and ncell not in ch):
                        ok.add("BA")
            return ok or {"AB", "BA"}
        for tag, seq in (("AB", ch), ("BA", ch[::-1])):
            good = True
            for i in range(len(seq) - 1):
                if not any(nb[0] == seq[i + 1]
                           for d in range(4)
                           for nb in self.succ.get((seq[i], d), ())):
                    good = False
                    break
            if good:
                ok.add(tag)
        return ok

    def _refresh_stations(self, env):
        plat, sinks = set(), set()
        for a in env.agents:
            wps = a.waypoints
            for alts in wps[1:-1]:
                for wp in alts:
                    plat.add(tuple(wp.position))
            sinks.add(tuple(a.target))
        registered = getattr(self, "_registered_platform_cells", set())
        self.platform_cells = set(registered) | plat
        self.sink_cells = sinks
        for b in self.blocks:
            b.platforms = {c for c in b.chain if c in self.platform_cells}

    def register_stations(self, train_stations):
        """Ingest stations.pkl-style data: a list per city of
        ((row, col), platform_idx) entries. Records every platform cell,
        groups platforms by city, and re-marks blocks. Returns
        {city_index: set(platform cells)}."""
        by_city = {}
        for ci, plats in enumerate(train_stations):
            cells = {tuple(p[0]) for p in plats}
            by_city[ci] = cells
            self._registered_platform_cells |= cells
        self.platform_cells = set(self._registered_platform_cells) | set(self.platform_cells)
        for b in self.blocks:
            b.platforms = {c for c in b.chain if c in self.platform_cells}
        self.station_platforms = by_city
        return by_city

    # -------------------------------------------------------------- queries
    def resource_of(self, cell):
        """('node', cell) for junctions/crossings, ('block', bid) otherwise."""
        if cell in self.nodes:
            return ("node", cell)
        return ("block", self.block_of[cell])

    def exit_end(self, cell, d):
        """For a cell inside a block: which end node the train at (cell, d)
        is heading to (None = dead end / off-ring). Junction cells: None."""
        bid = self.block_of.get(cell)
        if bid is None:
            return None
        b = self.blocks[bid]
        idx = b.chain.index(cell)
        for ncell, _ in self.succ.get((cell, d), ()):
            if ncell in b.chain[idx + 1: idx + 2]:
                return b.ends[1]
            if idx > 0 and ncell == b.chain[idx - 1]:
                return b.ends[0]
            if ncell == b.ends[1]:
                return b.ends[1]
            if ncell == b.ends[0]:
                return b.ends[0]
        return None

    def bays(self):
        return [b for b in self.blocks if b.is_bay]

    def stats(self):
        kinds = {}
        for b in self.blocks:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
        bay_caps = sum(b.capacity for b in self.blocks if b.is_bay)
        return dict(cells=len(self.cells), junctions=len(self.junctions),
                    crossings=len(self.crossings), deadends=len(self.deadend_cells),
                    blocks=len(self.blocks), **kinds,
                    parallel_groups=len(self.groups),
                    direct_links=sum(self.direct_links.values()),
                    bays=len(self.bays()), bay_capacity=bay_caps,
                    platform_cells=len(self.platform_cells),
                    platforms_on_through=sum(len(b.platforms) for b in self.blocks
                                             if b.kind == "through" and b.platforms))

    # ----------------------------------------------------------- validation
    def validate(self):
        """Programmatic invariants. Raises AssertionError on the first lie."""
        # I1: partition -- every rail cell is node xor exactly one block cell
        seen = set()
        for b in self.blocks:
            for c in b.chain:
                assert c not in self.nodes, f"I1 block cell {c} is a node"
                assert c not in seen, f"I1 cell {c} in two blocks"
                seen.add(c)
        assert seen | self.nodes == self.cells, "I1 coverage gap"

        for b in self.blocks:
            # I2: chain contiguity + interior plainness
            for i in range(len(b.chain) - 1):
                assert b.chain[i + 1] in self.und[b.chain[i]], \
                    f"I2 chain break in block {b.bid} at {i}"
            for c in b.chain:
                assert self.degree[c] <= 2, f"I2 plain cell {c} degree>2"
            # I3: recorded ends really attach to the chain endpoints
            for cell, end in ((b.chain[0], b.ends[0]), (b.chain[-1], b.ends[1])):
                if end is not None:
                    assert end in self.und[cell], \
                        f"I3 block {b.bid}: end {end} not adjacent to {cell}"
                    assert end in self.nodes, f"I3 end {end} not a node"
                else:
                    nb_nodes = self.und[cell] & self.nodes
                    outside = self.und[cell] - set(b.chain)
                    assert not (outside & self.nodes), \
                        f"I3 block {b.bid}: missed node at {cell}"
            # I4: stubs touch a dead end, loops have a group, capacities sane
            if b.kind == "stub":
                assert None in b.ends, f"I4 stub {b.bid} has two node ends"
            if b.kind == "loop":
                assert b.group is not None, f"I4 loop {b.bid}"
                assert b.is_bay == (len(b.chain) >= MIN_LOOP_BAY_LEN), \
                    f"I4 loop {b.bid} bay eligibility"
            assert b.capacity == len(b.chain), f"I4 capacity block {b.bid}"
            assert b.dir_ok, f"I4 block {b.bid} traversable in no direction"

        # I5: directed walk from inside a block reaches the recorded end
        for b in self.blocks:
            if len(b.chain) < 2 or None in b.ends:
                continue
            cell = b.chain[len(b.chain) // 2]
            for d in range(4):
                if (cell, d) not in self.succ:
                    continue
                got = self.exit_end(cell, d)
                walk = self._brute_exit(cell, d)
                assert got == walk, \
                    f"I5 block {b.bid} ({cell},{d}): exit {got} != walk {walk}"

        # I6: crossings really have no choice, junctions really do
        for c in self.crossings:
            assert all(len(self.succ.get((c, d), [()])) <= 1 for d in range(4)), \
                f"I6 crossing {c} has a choice"
            assert self.degree[c] >= 3, f"I6 crossing {c} degree<3"
        for c in self.junctions:
            assert any(len(self.succ.get((c, d), ())) >= 2 for d in range(4)), \
                f"I6 junction {c} has no choice"

        # I7: parallel groups consistent
        for g in self.groups:
            assert g.n_routes >= 2, f"I7 group {g.gid} not parallel"
            for bid in g.blocks:
                b = self.blocks[bid]
                assert frozenset(e for e in b.ends) == g.ends or \
                    frozenset(b.ends) == g.ends, f"I7 group {g.gid} member ends"
                assert b.kind == "loop", f"I7 member {bid} not loop"
            assert g.max_standing_blocks == g.n_routes - 1, f"I7 group {g.gid}"
        return True

    def _brute_exit(self, cell, d):
        cur, seen = (cell, d), {cell}
        for _ in range(self.H * self.W):
            outs = self.succ.get(cur, ())
            if not outs:
                return None
            nc, nd = outs[0]
            if nc in self.nodes:
                return nc
            if nc in seen:
                return None
            seen.add(nc)
            cur = (nc, nd)
        return None


def _restore_baygraph(state):
    obj = object.__new__(BayGraph)
    obj.__dict__.update(state)
    BayGraph._cache.setdefault(state["_fp"], obj)
    return obj
