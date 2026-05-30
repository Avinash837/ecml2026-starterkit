from flatland.core.env_observation_builder import ObservationBuilder


class MyObservationBuilder(ObservationBuilder):
    """
    The dispatcher (submission.dispatcher.V5Planner) is a global planner: it needs the
    whole env state (rail, agent positions, timetable), not a per-agent local view. So
    the "observation" handed to the policy is simply the live RailEnv. The policy reads
    env.rail / env.agents / env._elapsed_steps directly. (ObservationBuilder.set_env sets
    self.env before observations are requested.)
    """

    def reset(self):
        pass

    def get(self, handle: int = 0):
        return self.env

    def get_many(self, handles=None):
        return {h: self.env for h in (handles or [])}
