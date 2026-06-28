"""Evaluation-aware routing layer on top of the interlocking controller.

This module is intentionally small: it reuses the bay graph, section claims,
bay slots, recovery, and action logic from ``interlocking.py``.  The only new
piece is a route-selection layer that compares a few route candidates for each
train before handing the selected path back to the same safety machinery.

The important architectural rule is that this layer never grants movement.
It only chooses the planned path; the base interlocking remains the authority
for whether a train can actually move.
"""
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy

from submission.bay_graph import BayGraph
from submission.interlocking import InterlockingController, Router


@dataclass
class RouteCandidate:
    name: str
    route: Tuple[list, dict]
    penalty: float
    missed: int
    served: int
    travel_time: float
    direct_extra: int = 0


class LayerRouteSelector:
    """Choose routes by marginal ECML reward economics, not level gates."""

    MISS_STOP_COST = 50.0
    LOW_AGENT_MISS_STOP_COST = 80.0
    TARGET_MISS_COST = 100.0
    DETOUR_TRIALS = 4
    DETOUR_MAX_RATIO = 1.16
    DETOUR_MAX_EXTRA = 42
    LOW_AGENT_DETOUR_TRIALS = 8
    LOW_AGENT_DETOUR_MAX_RATIO = 1.35
    LOW_AGENT_DETOUR_MAX_EXTRA = 120
    CLOSE_MARGIN = 18.0
    SOFTMAX_TEMP = 9.0

    def __init__(self, controller: "LayerVariantController"):
        self.controller = controller
        self.env = controller.env

    @staticmethod
    def stable_unit(handle: int, salt: int = 0) -> float:
        x = (int(handle) * 1103515245 + 12345 + 2654435761 * salt)
        x &= 0x7fffffff
        return x / float(0x80000000)

    def choose(self, router: Router, agent, start, gis, extra=None):
        if self._runtime_replan(agent):
            return self._cheap_runtime_route(router, agent, start, gis, extra)
        candidates = self._candidates(router, agent, start, gis, extra)
        if not candidates:
            return None, "no layer candidate"
        if len(candidates) == 1:
            self._record(candidates[0])
            return candidates[0].route, None

        candidates.sort(key=lambda c: (c.penalty, c.travel_time, c.name))
        best = candidates[0]
        close = [c for c in candidates
                 if c.penalty <= best.penalty + self.CLOSE_MARGIN]
        if len(close) == 1:
            self._record(best)
            return best.route, None

        chosen = self._soft_choice(agent.handle, close)
        self._record(chosen)
        return chosen.route, None

    def _runtime_replan(self, agent) -> bool:
        if not getattr(self.controller, "_runtime_routing_enabled", False):
            return False
        state = getattr(self.controller, "state", None)
        return state is not None and agent.handle in state

    def _cheap_runtime_route(self, router, agent, start, gis, extra):
        route, err = router.route_direct_opportunistic(
            agent, start, gis, extra, collect_stops=True)
        if route is None:
            return None, err
        cand = self._score("runtime_direct", agent, gis, route, extra)
        self._record(cand)
        return route, None

    def _record(self, cand: RouteCandidate):
        counts = getattr(self.controller, "route_decisions", None)
        if counts is not None:
            counts[cand.name] = counts.get(cand.name, 0) + 1
        metrics = getattr(self.controller, "route_metrics", None)
        if metrics is not None:
            metrics["choices"] += 1
            metrics["planned_stops"] += cand.served
            metrics["missed_stops"] += cand.missed
            metrics["direct_extra"] += cand.direct_extra
            if cand.name == "opportunistic" and cand.served:
                metrics["free_stop_choices"] += 1
                metrics["free_stops"] += cand.served
            if cand.name.startswith("detour_stop_"):
                metrics["detour_choices"] += 1
                metrics["detour_stops"] += cand.served

    def _soft_choice(self, handle: int, candidates: Sequence[RouteCandidate]):
        best = candidates[0].penalty
        weights = [
            math.exp(-(c.penalty - best) / self.SOFTMAX_TEMP)
            for c in candidates
        ]
        total = sum(weights)
        draw = self.stable_unit(handle, len(candidates)) * total
        acc = 0.0
        for cand, weight in zip(candidates, weights):
            acc += weight
            if draw <= acc:
                return cand
        return candidates[0]

    def _candidates(self, router: Router, agent, start, gis, extra):
        if not gis:
            route = ([start], {})
            return [self._score("empty", agent, gis, route, extra)]

        out = []
        seen = set()
        final_gi = len(agent.waypoints) - 1
        mids = [gi for gi in gis if 1 <= gi < final_gi]

        def add(name, route):
            if route is None:
                return
            path, stops = route
            key = (tuple(path), tuple(sorted(stops.items())))
            if key in seen:
                return
            seen.add(key)
            out.append(self._score(name, agent, gis, route, extra))

        full, _ = router._route_from_full(agent, start, gis, extra)
        add("full", full)

        if mids:
            direct, _ = router.route_direct_opportunistic(
                agent, start, gis, extra, collect_stops=False)
            add("direct", direct)
            opportunistic, _ = router.route_direct_opportunistic(
                agent, start, gis, extra, collect_stops=True)
            add("opportunistic", opportunistic)
            for gi, route in self._bounded_detour_routes(
                    router, agent, start, gis, direct, extra):
                add(f"detour_stop_{gi}", route)

        return out

    def _bounded_detour_routes(self, router, agent, start, gis, direct, extra):
        """Small prize-collecting detours around the direct target route.

        A stop is worth roughly 50 reward units, but a large station detour can
        block many other trains.  So candidate generation first bounds detour
        size against the direct route; scoring then decides whether the stop is
        still worth its route-time, dwell, congestion and malfunction risk.
        """
        if direct is None:
            return []
        final_gi = len(agent.waypoints) - 1
        mids = [gi for gi in gis if 1 <= gi < final_gi]
        if not mids:
            return []

        direct_len = max(0, len(direct[0]) - 1)
        if len(self.env.agents) < 50:
            trials = self.LOW_AGENT_DETOUR_TRIALS
            max_ratio = self.LOW_AGENT_DETOUR_MAX_RATIO
            max_extra_cap = self.LOW_AGENT_DETOUR_MAX_EXTRA
        else:
            trials = self.DETOUR_TRIALS
            max_ratio = self.DETOUR_MAX_RATIO
            max_extra_cap = self.DETOUR_MAX_EXTRA
        max_extra = max(8, min(max_extra_cap,
                               int(direct_len * (max_ratio - 1.0))))
        final_goals = router._goals_for(agent, final_gi)
        final_field = router.dist_field(final_goals)
        ranked = []
        for gi in mids:
            goals = router._goals_for(agent, gi)
            to_stop = router.dist_field(goals).get(start)
            if to_stop is None:
                continue
            from_stop = min(
                (final_field[g] for g in goals if g in final_field),
                default=None,
            )
            if from_stop is None:
                continue
            est_extra = to_stop + from_stop - direct_len
            if est_extra <= max_extra:
                ranked.append((est_extra, gi))

        routes = []
        for _estimate, gi in sorted(ranked)[:trials]:
            route, _ = router._route_from_full(
                agent, start, [gi, final_gi], extra)
            if route is not None and len(route[0]) - 1 <= direct_len + max_extra:
                routes.append((gi, route))
        return routes

    def _score(self, name, agent, gis, route, extra):
        path, stops = route
        final_gi = len(agent.waypoints) - 1
        mids = {gi for gi in gis if 1 <= gi < final_gi}
        served = {gi for gi in stops.values() if gi in mids}
        missed = mids - served
        speed = agent.speed_counter.speed or 1.0
        dwell_steps = len(served)
        travel_time = (max(0, len(path) - 1) / speed) + dwell_steps
        t0 = max(getattr(self.controller, "t", 0),
                 getattr(agent, "earliest_departure", 0) or 0)
        arrival = t0 + travel_time

        miss_cost = (self.LOW_AGENT_MISS_STOP_COST
                     if len(self.env.agents) < 50
                     else self.MISS_STOP_COST)
        penalty = miss_cost * len(missed)
        penalty += self._target_penalty(agent, arrival)
        penalty += self._intermediate_timing_penalty(
            agent, path, stops, served, speed, t0)
        penalty += self._throughput_shadow_cost(
            name, agent, path, extra, travel_time, dwell_steps, len(mids))

        return RouteCandidate(
            name=name,
            route=route,
            penalty=penalty,
            missed=len(missed),
            served=len(served),
            travel_time=travel_time,
        )

    def _target_penalty(self, agent, arrival: float) -> float:
        late = max(0.0, arrival - float(agent.latest_arrival))
        horizon = max(0.0, arrival - float(self.env._max_episode_steps))
        if horizon <= 0:
            return late
        return late + self.TARGET_MISS_COST + 2.0 * horizon

    def _intermediate_timing_penalty(
            self, agent, path, stops, served, speed, t0: float) -> float:
        latest = getattr(agent, "waypoints_latest_arrival", None)
        earliest = getattr(agent, "waypoints_earliest_departure", None)
        if not latest and not earliest:
            return 0.0

        penalty = 0.0
        dwell_seen = 0
        for pidx, gi in sorted(stops.items()):
            if gi not in served:
                continue
            stop_time = t0 + (pidx / speed) + dwell_seen
            dwell_seen += 1
            if latest and gi < len(latest) and latest[gi] is not None:
                penalty += 0.5 * max(0.0, stop_time - float(latest[gi]))
            if earliest and gi < len(earliest) and earliest[gi] is not None:
                penalty += 0.5 * max(0.0, float(earliest[gi]) - stop_time)
        return penalty

    def _throughput_shadow_cost(
            self, name, agent, path, extra, travel_time, dwell_steps, mid_count):
        n_agents = max(1, len(self.env.agents))
        mrate = float(getattr(self.controller, "_mrate", 0.0) or 0.0)
        complexity = min(1.0, mid_count / 4.0)

        # This is the social price of occupying scarce track. It is continuous
        # in route complexity, fleet size, speed and malfunction risk; no level
        # or scenario id is used.
        time_weight = 0.08 + 0.035 * complexity + min(0.06, n_agents / 6000.0)
        if agent.speed_counter.speed <= 0.5:
            time_weight += 0.025
        time_cost = time_weight * travel_time

        dwell_base = 2.0 + min(16.0, n_agents / 45.0) + 14000.0 * mrate
        if name == "full" or name.startswith("detour_stop_"):
            station_pressure = n_agents + max(0.0, n_agents - 30.0)
            dwell_base += min(90.0, station_pressure)
            if getattr(self.controller, "_runtime_routing_enabled", False) \
                    and agent.handle in getattr(self.controller, "state", {}):
                dwell_base += 85.0
        dwell_risk = dwell_steps * dwell_base

        congestion = 0.0
        if extra:
            for node in path:
                cell, direction = node
                congestion += extra.get(cell, 0.0) + extra.get(node, 0.0)
            congestion *= 0.025

        return time_cost + dwell_risk + congestion


class LayerVariantRouter(Router):
    def __init__(self, bay_graph, controller):
        super().__init__(bay_graph)
        self.selector = LayerRouteSelector(controller)

    def route_from(self, agent, start, gis, extra=None):
        return self.selector.choose(self, agent, start, gis, extra)


class LayerVariantController(InterlockingController):
    """Interlocking controller with a separate route-selection layer."""

    USE_GLOBAL_PRIORITY = False
    BALANCE_ITERS = 6
    BALANCE_FRAC = 0.45
    BALANCE_STRETCH = 1.35

    def __init__(self, env, bay_graph):
        self.global_priority_rank = {}
        self.global_priority_score = {}
        self.priority_metrics = {}
        self.balance_metrics = {}
        self._runtime_routing_enabled = False
        super().__init__(env, bay_graph)
        self._load_balance_parallel_routes()
        self._rebuild_state_for_current_plans()
        if self.USE_GLOBAL_PRIORITY:
            self._build_global_priority_order()
        self._runtime_routing_enabled = True

    def _make_router(self, bay_graph):
        self.route_decisions = {}
        self.route_metrics = dict(
            choices=0,
            planned_stops=0,
            missed_stops=0,
            direct_extra=0,
            free_stop_choices=0,
            free_stops=0,
            detour_choices=0,
            detour_stops=0,
            actual_served=0,
        )
        return LayerVariantRouter(bay_graph, self)

    def _grant_order(self, pressure):
        if not self.USE_GLOBAL_PRIORITY or not self.global_priority_rank:
            return super()._grant_order(pressure)
        missing = len(self.global_priority_rank) + 1
        return sorted(
            self.state,
            key=lambda h: (
                self.global_priority_rank.get(h, missing),
                -pressure(h),
                h,
            ),
        )

    def _rebuild_state_for_current_plans(self):
        self.state = {
            h: dict(ptr=0, anchor=0, slot=None, claims=[],
                    edges_held=[], frontier=-1, served=set(),
                    departed=False, stuck=0, pending=[], replans=0,
                    last_bay=None, last_bay_ptr=0, midstuck=0)
            for h in self.plans
        }

    def _parallel_overload_ratio(self):
        """System-wide standing demand vs capacity on parallel-group bays.
        >1 means the alternative-track structure is genuinely over capacity
        (a real dispatcher only lengthens runs to decongest when it is)."""
        cap = 0
        for g in self.bg.groups:
            for m in g.blocks:
                if self.bg.blocks[m].is_bay:
                    cap += max(1, self.bg.blocks[m].capacity)
        dem = 0
        for p in self.plans.values():
            for anchor in p["anchors"]:
                bid = anchor[2]
                if bid >= 0 and self.bg.blocks[bid].group is not None:
                    dem += 1
        return dem / max(1, cap)

    def _load_balance_parallel_routes(self):
        """Min-cost-flow style relaxation over parallel sibling blocks.

        The bay graph already exposes each group of alternative blocks between
        the same nodes.  We treat overloaded siblings as having a shadow price,
        reroute the trains paying the highest prices, then recompute prices.
        This is the cheap submit-safe version of a min-cost flow assignment.
        """
        moved_total = 0
        attempts_total = 0
        # strict (equal-or-shorter) by default; only allow lengthening
        # reroutes when the parallel structure is truly over capacity.
        ratio = self._parallel_overload_ratio()
        self._balance_stretch = self.BALANCE_STRETCH if ratio > 1.2 else 1.0
        for _ in range(self.BALANCE_ITERS):
            load, dir_load, per_train = self._parallel_loads(self.plans)
            prices = self._parallel_prices(load, dir_load)
            if not prices:
                break
            current_obj = self._parallel_objective(prices)

            def train_cost(h):
                return sum(prices.get(bid, 0.0)
                           for bid, _end in per_train.get(h, ()))

            ranked = [h for h in sorted(self.plans, key=train_cost, reverse=True)
                      if train_cost(h) > 0.0]
            if not ranked:
                break
            limit = max(1, int(len(ranked) * self.BALANCE_FRAC))
            moved = 0
            for h in ranked[:limit]:
                attempts_total += 1
                old_plan = self.plans[h]
                old_len = max(1, len(old_plan["path"]) - 1)
                old_cost = train_cost(h)
                remove = per_train.get(h, ())
                extra, trial_prices = self._parallel_balance_extra(
                    load, dir_load, remove)
                a = self.env.agents[h]
                start = (tuple(a.initial_position), int(a.initial_direction))
                gis = list(range(1, len(a.waypoints)))
                route, _ = self.router.route_from(a, start, gis, extra)
                if route is None:
                    continue
                path, stops = route
                new_len = max(1, len(path) - 1)
                # only relocate to a sibling that is no longer than the
                # current route (a real dispatcher does not lengthen a run to
                # decongest a section that is not over capacity); the stretch
                # allowance only applies once the system is genuinely
                # oversubscribed (set per-scenario in _load_balance_parallel_routes)
                if new_len > int(old_len * self._balance_stretch):
                    continue
                if len(stops) < len(old_plan["stops"]):
                    continue
                new_entries = self._group_entries(path)
                trial_load, trial_dir = self._copy_parallel_loads(load, dir_load)
                self._apply_group_entries(trial_load, trial_dir, remove, -1)
                self._apply_group_entries(trial_load, trial_dir, new_entries, 1)
                trial_prices = self._parallel_prices(trial_load, trial_dir)
                trial_obj = self._parallel_objective(trial_prices)
                if trial_obj >= current_obj * 0.995:
                    continue
                self.plans[h] = self._make_plan(a, path, stops)
                load, dir_load, prices = trial_load, trial_dir, trial_prices
                per_train[h] = new_entries
                current_obj = trial_obj
                moved += 1
            moved_total += moved
            if moved == 0:
                break

        load, dir_load, _per_train = self._parallel_loads(self.plans)
        prices = self._parallel_prices(load, dir_load)
        self.balance_metrics = dict(
            moved=moved_total,
            attempts=attempts_total,
            priced_blocks=len(prices),
            max_price=round(max(prices.values()), 3) if prices else 0.0,
            objective=round(self._parallel_objective(prices), 3),
        )

    def _parallel_loads(self, plans):
        load = defaultdict(int)
        dir_load = defaultdict(lambda: defaultdict(int))
        per_train = {}
        for h, plan in plans.items():
            seen = []
            for bid, end in self._path_blocks(plan["path"]):
                block = self.bg.blocks[bid]
                if block.group is None:
                    continue
                load[bid] += 1
                dir_load[bid][end] += 1
                seen.append((bid, end))
            per_train[h] = seen
        return load, dir_load, per_train

    def _group_entries(self, path):
        return [(bid, end) for bid, end in self._path_blocks(path)
                if self.bg.blocks[bid].group is not None]

    @staticmethod
    def _parallel_objective(prices):
        if not prices:
            return 0.0
        mx = max(prices.values())
        return sum(v * v for v in prices.values()) + 25.0 * mx * mx

    @staticmethod
    def _copy_parallel_loads(load, dir_load):
        out_load = defaultdict(int)
        out_load.update(load)
        out_dir = defaultdict(lambda: defaultdict(int))
        for bid, ends in dir_load.items():
            out_dir[bid].update(ends)
        return out_load, out_dir

    @staticmethod
    def _apply_group_entries(load, dir_load, entries, delta):
        for bid, end in entries:
            load[bid] = max(0, load[bid] + delta)
            dir_load[bid][end] = max(0, dir_load[bid][end] + delta)
            if dir_load[bid][end] == 0:
                del dir_load[bid][end]

    def _parallel_prices(self, load, dir_load):
        prices = {}
        for group in self.bg.groups:
            members = list(group.blocks)
            if len(members) < 2:
                continue
            member_loads = {bid: load.get(bid, 0) for bid in members}
            total = sum(member_loads.values())
            if total <= len(members):
                continue
            low = min(member_loads.values())
            ideal = max(1.0, total / float(len(members) + group.direct_links))
            for bid in members:
                ends = dir_load.get(bid, {})
                mixed = min(ends.values()) if len(ends) >= 2 else 0
                excess = max(0.0, member_loads[bid] - ideal)
                imbalance = max(0.0, member_loads[bid] - low)
                price = 2.0 * member_loads[bid] + 14.0 * excess \
                    + 18.0 * imbalance + 16.0 * mixed
                if price > 0.0:
                    prices[bid] = price
        return prices

    def _parallel_balance_extra(self, load, dir_load, remove):
        trial_load = defaultdict(int, load)
        trial_dir = defaultdict(lambda: defaultdict(int))
        for bid, ends in dir_load.items():
            trial_dir[bid].update(ends)
        for bid, end in remove:
            trial_load[bid] = max(0, trial_load[bid] - 1)
            if end in trial_dir[bid]:
                trial_dir[bid][end] = max(0, trial_dir[bid][end] - 1)
                if trial_dir[bid][end] == 0:
                    del trial_dir[bid][end]
        prices = self._parallel_prices(trial_load, trial_dir)
        extra = dict(self.oneway)
        for bid, price in prices.items():
            for cell in self.bg.blocks[bid].chain:
                extra[cell] = extra.get(cell, 0.0) + price
        return extra, prices

    def _build_global_priority_order(self):
        """Forecast bottlenecks and produce a stable conflict-aware order.

        This is the first, cheap slice of a PBS-style layer.  It does not
        branch/search yet; it takes the current route set, finds resources
        demanded above capacity, orders same-direction batches through each
        bottleneck, and aggregates those local preferences into one global
        grant priority for the event-driven executor.
        """
        if not self.plans:
            self.global_priority_rank = {}
            self.global_priority_score = {}
            self.priority_metrics = {}
            return

        stats = {}
        demand = defaultdict(dict)
        for h, p in self.plans.items():
            a = self.env.agents[h]
            speed = float(a.speed_counter.speed or 1.0)
            length = max(0, len(p["path"]) - 1)
            slack = float(a.latest_arrival - a.earliest_departure
                          - length / speed)
            stats[h] = dict(
                slack=slack,
                length=length,
                earliest=a.earliest_departure,
            )
            for idx, (bid, end) in enumerate(self._path_blocks(p["path"])):
                old = demand[bid].get(h)
                if old is None or idx < old[0]:
                    demand[bid][h] = (idx, end)

        scores = {h: 0.0 for h in self.plans}
        bottlenecks = 0
        constraints = 0
        for bid, entries_by_handle in demand.items():
            block = self.bg.blocks[bid]
            capacity = max(1, block.capacity if block.is_bay else 1)
            load = len(entries_by_handle)
            if load <= capacity:
                continue
            bottlenecks += 1
            excess = load - capacity
            weight = min(50.0, 1.0 + excess / capacity)

            by_end = defaultdict(list)
            for h, (idx, end) in entries_by_handle.items():
                by_end[end].append((h, idx))

            def group_key(item):
                end, members = item
                best_slack = min(stats[h]["slack"] for h, _ in members)
                best_idx = min(idx for _h, idx in members)
                return (-len(members), best_slack, best_idx, str(end))

            ordered = []
            for _end, members in sorted(by_end.items(), key=group_key):
                members.sort(key=lambda hi: (
                    hi[1],
                    stats[hi[0]]["slack"],
                    stats[hi[0]]["earliest"],
                    hi[0],
                ))
                ordered.extend(h for h, _idx in members)

            n = len(ordered)
            for pos, h in enumerate(ordered):
                scores[h] += weight * (n - pos)
            constraints += max(0, n - 1)

        ordered_handles = sorted(
            self.plans,
            key=lambda h: (
                -scores[h],
                stats[h]["slack"],
                stats[h]["length"],
                stats[h]["earliest"],
                h,
            ),
        )
        self.global_priority_score = scores
        self.global_priority_rank = {
            h: i for i, h in enumerate(ordered_handles)
        }
        top = ordered_handles[0] if ordered_handles else None
        self.priority_metrics = dict(
            bottlenecks=bottlenecks,
            constraints=constraints,
            ordered=len(ordered_handles),
            top_handle=top,
            top_score=round(scores.get(top, 0.0), 3) if top is not None else 0.0,
        )

    def act(self):
        actions = super().act()
        self.route_metrics["actual_served"] = sum(
            len(st.get("served", ())) for st in self.state.values())
        return actions


class LayerVariantPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self._ctl = None
        self._env_id = None
        self._step_seen = None
        self._actions: Dict[int, RailEnvActions] = {}
        self._dead = False        # fail-safe latch (see act_many)

    def _ensure_controller(self, env):
        fresh = (self._ctl is None or self._env_id != id(env)
                 or env._elapsed_steps == 0 and self._step_seen not in (None, 0))
        if fresh:
            bg = BayGraph(env)
            self._ctl = LayerVariantController(env, bg)
            self._env_id = id(env)
            self._step_seen = None

    def act_many(self, handles: List[int], observations: List[Any],
                 **kwargs) -> Dict[int, RailEnvActions]:
        # FAIL-SAFE: never let an exception escape into the eval runner -- an
        # uncaught error there turns the ENTIRE submission into a generic
        # "General failure" (zero score, no signal). On any error we dump the
        # traceback to stderr (which surfaces in the competition eval log, so we
        # learn the exact cause) and idle the trains instead of crashing.
        if self._dead:
            return {h: RailEnvActions.DO_NOTHING for h in handles}
        try:
            env = observations[0]
            self._ensure_controller(env)
            step = env._elapsed_steps
            if step != self._step_seen:
                self._actions = self._ctl.act()
                self._step_seen = step
            return {h: self._actions.get(h, RailEnvActions.DO_NOTHING)
                    for h in handles}
        except Exception:
            import sys, traceback
            sys.stderr.write(
                "=== SUBMISSION act_many FAILED (fail-safe -> DO_NOTHING) ===\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self._dead = True     # stop retrying: no log spam, no repeated plan
            return {h: RailEnvActions.DO_NOTHING for h in handles}

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions.DO_NOTHING


class MyPolicy(LayerVariantPolicy):
    """Convenience entrypoint when running this module as the submitted policy."""
