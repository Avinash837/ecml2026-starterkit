"""
Competition policy = the V6 deadlock-free reservation dispatcher
(submission.dispatcher).

The evaluation runner calls `act_many(handles, observations=list(observations.values()))`
every step. Our MyObservationBuilder returns the live RailEnv as each agent's observation,
so observations[0] is the env the planner needs. On the first call it plans all trains
once (space-time reservations); thereafter it just executes the timed plan. Plans are
conflict-free => collision- and deadlock-free by construction.
"""
from typing import Any, Dict, List

from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_action import RailEnvActions

from submission.dispatcher import V5Planner
# NOTE: do NOT import submission.priority here -- it imports torch, which is NOT in the
# container (requirements.txt ships only flatland-rl). Importing it crashed the policy load
# in the competition runner (the cause of the first two failed submissions). fast-first needs
# no trained net, so there's no torch dependency.


def fast_first_priority(env, topo, h):
    """Planning/release order: faster trains first (they clear the network and free capacity).
    MUST be a module-level function (NOT a lambda/closure): the competition runner PICKLES the
    policy, and a lambda is unpicklable -> the job fails to start. Validated +4-7pp on clean
    levels (real-map, 6 seeds), neutral on malfunction levels."""
    return -float(env.agents[h].speed_counter.speed)


class MyPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self._planner = V5Planner()
        # V6 keeps V5's direct completion-first routing, but collects
        # low-density intermediate stops that already lie on the chosen route.
        self._planner.opportunistic_stops = True
        self._planner.opportunistic_stop_agent_cap = 60
        self._planner.prefer_route_stops = True
        self._planner.prefer_route_stops_agent_cap = 30
        self._planner.prefer_route_stops_bonus = 20
        # Malfunction drift can make an otherwise conflict-free timetable stale.
        # A late/frozen train inside a single-track segment temporarily owns that
        # segment direction, preventing the head-to-head corridor swaps found in
        # L3/L4 diagnostics. Clean guards stayed unchanged in local proxy tests.
        self._planner.late_segment_guard = True
        self._planner.late_guard_threshold = 30
        self._planner.late_guard_reroute_after = 0
        self._planner.late_guard_reroute_cooldown = 200
        self._planner.max_late_guard_reroutes = 1
        # V6 throughput gate: directional corridor load helps low-density
        # maps but hurts dense maps, so act_many gates that by agent count.
        # Fast dynamic release stayed broadly positive on the local proxies
        # and is enabled for every density.
        self._low_density_agent_cap = 60
        # FAST-FIRST planning order: plan/release FAST trains first. They clear the network
        # quickly and free track capacity, so more trains finish before the (tight) horizon.
        # Validated on the RECONSTRUCTED REAL competition map, 6 seeds x 4 conditions:
        #   clean 250ag +4.3pp, clean 150ag +6.9pp (the malfunction-free levels 0-2 -- the score
        #   backbone), malfunction 250ag -0.1pp (neutral), malfunction 150ag +1.8pp. Net-positive
        #   everywhere, lower variance than the slack default, and FREE (pure ordering). The
        #   opposite (slow/long-distance first) is catastrophic (-16pp), confirming the mechanism.
        #   (signal_guard / load_weight looked good on proxies but collapsed under real-map 6-seed
        #   testing; fast-first is the one lever that held up.)
        self._planner.priority_fn = fast_first_priority   # module-level fn (picklable, no torch)
        # V7: directional flow separation at DENSE. Real-map route analysis shows
        # shortest-path routing funnels both directions onto the same corridors
        # (rows 75/80, col 65) -> ~923 segment head-on; a directional opposing
        # penalty pulls the flows onto the map's parallel corridors (junction-dense,
        # 443 segs) -> ~277 head-on (-70%), so a frozen train no longer blocks the
        # whole opposing stream. Off (0.0) reproduces v6 exactly; tune via --set.
        self._planner.dense_dir_weight = 0.0

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        env = observations[0]            # MyObservationBuilder hands us the live RailEnv
        n = env.get_num_agents()
        low_density = n <= self._low_density_agent_cap
        # V7: DEPARTURE METERING at dense. The dense map is ~3x over capacity, so
        # injecting all due trains saturates it -> the head-on gridlock a malfunction
        # makes permanent. Cap concurrent on-map trains (held trains wait off-map,
        # cost-free) so the network stays fluid. Adaptive cap (scales with n) so it
        # binds at mid-density (L4 150ag: 52->59) without starving the 532 levels;
        # off at low density (not saturated). Validated +7 malf / neutral clean.
        self._planner.meter_cap = (10 ** 9 if low_density else
                                   max(self._planner.meter_cap_floor,
                                       n // self._planner.meter_div))
        self._planner.dir_weight = 0.5 if low_density else self._planner.dense_dir_weight
        self._planner.late_guard_reroute_after = 0 if low_density else 200
        self._planner.exec_fast_first = True
        self._planner.signal_guard = False
        # NOTE: signal_guard / block_lock / crit_weight / greedy_advance / release_interval are
        # all available on the planner but kept OFF -- verification showed signal_guard is only
        # net-positive on SMALL malfunction scenes and slightly negative on dense ones, so it is
        # not the clean zero-downside hedge it first appeared. We ship the PROVEN 8.24 behavior.
        return self._planner.act_many(handles, [env])

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        # not used (act_many drives the episode); return a safe no-op
        return RailEnvActions.DO_NOTHING
