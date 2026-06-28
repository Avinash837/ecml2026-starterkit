import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards

from eval_harness import score_episode


def _parse_value(raw):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _load_class(path):
    mod_name, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), cls_name)


def _state_name(state):
    return getattr(state, "name", str(state).split(".")[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--policy", default="submission.my_policy.MyPolicy")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--time-limit", type=float, default=240.0)
    ap.add_argument("--set", action="append", default=[],
                    help="Set planner attribute, e.g. --set frozen_reroute=true")
    args = ap.parse_args()

    env, _ = RailEnvPersister.load_new(args.scenario, rewards=ECML2026Rewards())
    policy = _load_class(args.policy)()
    planner = getattr(policy, "_planner", policy)
    for item in args.set:
        name, raw = item.split("=", 1)
        setattr(planner, name, _parse_value(raw))

    result = score_episode(env, policy, seed=args.seed, time_limit=args.time_limit)
    states = Counter(_state_name(a.state) for a in env.agents)
    planned_to_target = 0
    plans = getattr(planner, "plans", {})
    for h, plan in plans.items():
        if plan and plan[-1][0] == tuple(env.agents[h].target):
            planned_to_target += 1
    no_plan = env.get_num_agents() - sum(1 for p in plans.values() if p)
    debug = getattr(planner, "debug_counts", {})

    print(
        f"complete={result['complete']}/{result['n_agents']},"
        f"norm={result['normalized_reward']:.9f},"
        f"steps={result['steps']}/{result['max_steps']},"
        f"planned_to_target={planned_to_target},no_plan={no_plan},"
        f"wall={result['wall_s']:.2f}s"
    )
    print(f"states={dict(states)}")
    if debug:
        print(f"debug={dict(sorted(debug.items()))}")


if __name__ == "__main__":
    main()
