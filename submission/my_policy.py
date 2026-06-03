"""
Competition policy = the deadlock-free reservation dispatcher (submission.dispatcher).

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

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        env = observations[0]            # MyObservationBuilder hands us the live RailEnv
        # NOTE: signal_guard / block_lock / crit_weight / greedy_advance / release_interval are
        # all available on the planner but kept OFF -- verification showed signal_guard is only
        # net-positive on SMALL malfunction scenes and slightly negative on dense ones, so it is
        # not the clean zero-downside hedge it first appeared. We ship the PROVEN 8.24 behavior.
        return self._planner.act_many(handles, [env])

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        # not used (act_many drives the episode); return a safe no-op
        return RailEnvActions.DO_NOTHING
