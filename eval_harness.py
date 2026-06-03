"""
Competition-accurate evaluation harness for the ECML 2026 Flatland challenge.

Why this exists
---------------
The provided debug environments are 25x25 / 5 agents / 2-stop lines and are scored
here (historically) with a naive "arrived/N" count. The real competition:
  * uses ECML2026Rewards (cancellation x5, target-not-reached -100,
    INTERMEDIATE-STOP-NOT-SERVED -50, collision x250) -- NOT the mild defaults;
  * runs MULTI-STOP timetabled lines (3-6 stops per train), so a planner that only
    routes origin->target silently eats -50 per missed intermediate stop;
  * aggregates score as the SUM of per-scenario normalized rewards in [0,1], and
    ABORTS a level early if <25% of trains complete.

This harness scores any policy with the exact official metric
(`flatland.evaluators.evaluator_callback.FlatlandEvaluatorCallbacks` logic) so local
numbers track the leaderboard instead of a toy proxy.

NOTE on the map: the real competition runs a single FIXED 120x150 / 28-station map
that is not distributed locally. We approximate scenario *scale* (agent count, stops,
malfunctions, grid size) with env_generator's random topology. Absolute scores will
differ from the leaderboard, but relative improvements and failure modes transfer.
"""
import time
import numpy as np
from flatland.env_generation.env_generator import env_generator
from flatland.envs.rewards import ECML2026Rewards
from flatland.envs.step_utils.states import TrainState


# ---- Level / scenario table (mirrors the official level config page) ----------
# stops == line_length (number of timetabled waypoints incl. origin & target).
# malf: (interval, dur_min, dur_max); interval=0 means no malfunctions.
# Map is the fixed competition map in reality; here we scale a random one.
FULL_MAP = dict(x_dim=150, y_dim=120, n_cities=28)

LEVELS = {
    0: dict(agents=[8, 11, 14, 26, 28],    stops=[3, 3, 4, 6, 6], malf=(0, 0, 0)),
    1: dict(agents=[36, 50, 62, 118, 210], stops=[3, 3, 4, 6, 6], malf=(0, 0, 0)),
    2: dict(agents=[90, 125, 150, 300, 532], stops=[3, 3, 4, 6, 6], malf=(0, 0, 0)),
    3: dict(agents=[36, 50, 62, 118, 210], stops=[3, 3, 4, 6, 6], malf=(540, 20, 50)),
    4: dict(agents=[90, 125, 150, 300, 532], stops=[3, 3, 4, 6, 6], malf=(360, 20, 50)),
    5: dict(agents=[90, 125, 150, 300, 532], stops=[3, 3, 4, 6, 6], malf=(180, 20, 50)),
    6: dict(agents=[532] * 5,               stops=[6] * 5,         malf=(120, 20, 50)),
}


def make_env(level, scenario, seed, map_cfg=None):
    """Build one scenario env at competition scale with the competition reward."""
    cfg = LEVELS[level]
    n_agents = cfg["agents"][scenario]
    stops = cfg["stops"][scenario]
    interval, dmin, dmax = cfg["malf"]
    m = map_cfg or FULL_MAP
    kwargs = dict(
        n_agents=n_agents,
        x_dim=m["x_dim"], y_dim=m["y_dim"], n_cities=m["n_cities"],
        line_length=stops,
        seed=seed,
        rewards=ECML2026Rewards(),
    )
    if interval > 0:
        kwargs.update(malfunction_interval=interval,
                      malfunction_duration_min=dmin, malfunction_duration_max=dmax)
    else:
        # BUG FIX: env_generator DEFAULTS to malfunction_interval=540, so NOT passing it added
        # malfunctions to the malfunction-free levels (0-2) too -- contaminating every local
        # baseline (it ~halves completion: big200 42%->22%). Disable explicitly so levels 0-2
        # match the real competition (which has no malfunctions there).
        kwargs.update(malfunction_interval=10 ** 9,
                      malfunction_duration_min=0, malfunction_duration_max=0)
    env, _, _ = env_generator(**kwargs)
    return env


def score_episode(env, policy, max_steps=None, time_limit=1800.0, seed=None, verbose=False):
    """
    Run `policy` on `env` and score with the EXACT official metric.

    `policy` must expose act_many(handles, [env]*n) -> {handle: action}, like the
    controllers in this repo. Returns a dict mirroring the official evaluation_state.
    """
    env.reset(random_seed=seed)
    n = env.get_num_agents()
    handles = list(range(n))
    cap = max_steps or env._max_episode_steps
    t_wall = time.time()
    term_cause = None
    step = 0
    for step in range(cap):
        if time.time() - t_wall > time_limit:
            term_cause = "TIMEOUT"
            break
        actions = policy.act_many(handles, [env] * n)
        _, _, dones, _ = env.step(actions)
        if dones["__all__"]:
            break
    wall = time.time() - t_wall

    # Official scoring: normalize over per-agent cumulative rewards.
    rewards = list(env.rewards_dict.values())
    normalized = env.rewards.normalize(*rewards, num_agents=n, max_episode_steps=env._max_episode_steps)
    cumulative = env.rewards.cumulate(*rewards)
    complete = sum(1 for a in env.agents if a.state == TrainState.DONE)
    pct = complete / n

    # Extra diagnostics the official metric hides but that we need to improve.
    departed = sum(1 for a in env.agents if not a.state.is_off_map_state() or a.state == TrainState.DONE)
    never_departed = sum(1 for a in env.agents if a.state.is_off_map_state() and a.state != TrainState.DONE)

    res = dict(
        normalized_reward=float(normalized) if normalized is not None else None,
        percentage_complete=pct,
        complete=complete, n_agents=n,
        never_departed=never_departed, departed=departed,
        cumulative=cumulative if np.isscalar(cumulative) else float(np.sum(list(cumulative.values())) if hasattr(cumulative, "values") else 0),
        steps=step + 1, max_steps=env._max_episode_steps,
        wall_s=wall, term_cause=term_cause,
        aborted_25pct=pct < 0.25,
    )
    if verbose:
        print(f"    norm={res['normalized_reward']:.4f}  complete={complete}/{n} ({pct*100:.1f}%)  "
              f"never_departed={never_departed}  steps={res['steps']}/{res['max_steps']}  {wall:.1f}s"
              + (f"  [{term_cause}]" if term_cause else "")
              + ("  <<25% ABORT" if res['aborted_25pct'] else ""))
    return res


def evaluate(policy_factory, levels, scenarios, seed=1, map_cfg=None, time_limit=1800.0):
    """Run a policy across (level, scenario) pairs and sum normalized rewards."""
    total = 0.0
    print(f"{'level':>5} {'scn':>3} {'agents':>6} {'stops':>5} | {'norm':>7} {'compl%':>7} {'nodep':>5} {'steps':>10} {'wall':>7}")
    for lvl in levels:
        for scn in scenarios:
            if scn >= len(LEVELS[lvl]["agents"]):
                continue
            env = make_env(lvl, scn, seed, map_cfg=map_cfg)
            pol = policy_factory()
            r = score_episode(env, pol, seed=seed, time_limit=time_limit)
            total += r["normalized_reward"] or 0.0
            flag = " <<ABORT" if r["aborted_25pct"] else ""
            print(f"{lvl:>5} {scn:>3} {r['n_agents']:>6} {LEVELS[lvl]['stops'][scn]:>5} | "
                  f"{r['normalized_reward']:>7.4f} {r['percentage_complete']*100:>6.1f}% {r['never_departed']:>5} "
                  f"{str(r['steps'])+'/'+str(r['max_steps']):>10} {r['wall_s']:>6.1f}s{flag}")
    print(f"\n  SUM normalized reward = {total:.4f}  (max = {len(levels)*len(scenarios):.0f})")
    return total


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from v2_scheduler import FastMAPFController

    # Small/fast map so the baseline runs quickly; raises scenarios gradually.
    QUICK_MAP = dict(x_dim=100, y_dim=80, n_cities=10)

    print("=== BASELINE: v2_scheduler.FastMAPFController, competition reward (ECML2026Rewards) ===")
    print("(QUICK_MAP 100x80/10 cities -- scale proxy, not the real fixed map)\n")
    evaluate(lambda: FastMAPFController(K=3),
             levels=[0], scenarios=[0, 1, 2], seed=1,
             map_cfg=QUICK_MAP, time_limit=120.0)
