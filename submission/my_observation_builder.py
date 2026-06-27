from flatland.core.env_observation_builder import ObservationBuilder


class MyObservationBuilder(ObservationBuilder):
    """The interlocking dispatcher is a global controller: it needs the whole
    env state (rail, agents, timetable), not a per-agent view. The
    "observation" handed to the policy is therefore the live RailEnv itself.
    (ObservationBuilder.set_env sets self.env before observations are
    requested.)"""

    def reset(self):
        pass

    def get(self, handle: int = 0):
        return self.env

    def get_many(self, handles=None):
        if handles is None:
            handles = range(self.env.get_num_agents())
        return {h: self.env for h in handles}
