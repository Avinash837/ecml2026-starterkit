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


class MyPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self._planner = V5Planner()

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        env = observations[0]            # MyObservationBuilder hands us the live RailEnv
        return self._planner.act_many(handles, [env])

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        # not used (act_many drives the episode); return a safe no-op
        return RailEnvActions.DO_NOTHING
