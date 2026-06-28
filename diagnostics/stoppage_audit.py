import argparse
import csv
import importlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rewards import ECML2026Rewards
from flatland.envs.step_utils.states import TrainState


DO = RailEnvActions.DO_NOTHING
FWD = RailEnvActions.MOVE_FORWARD
STOP = RailEnvActions.STOP_MOVING


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


def _fmt_cell(cell):
    if cell is None:
        return ""
    return f"{cell[0]}:{cell[1]}"


def _plan_index(planner, h, agent):
    plan = planner.plans.get(h)
    if not plan or agent.position is None:
        return None
    cp = tuple(agent.position)
    i = planner.ptr.get(h, 0)
    while i + 1 < len(plan) and plan[i][0] != cp:
        i += 1
    if i < len(plan) and plan[i][0] == cp:
        return i
    return None


def _classify(h, agent, action, planner, diag, t):
    plan = planner.plans.get(h)
    malf = getattr(agent.malfunction_handler, "malfunction_down_counter", 0)
    state = _state_name(agent.state)
    detail = {"blocker": "", "desired": "", "plan_index": "", "lateness": ""}

    if agent.state == TrainState.DONE:
        return "done", detail
    if malf > 0:
        return ("malfunction_off_map" if agent.position is None else "malfunction_on_map"), detail
    if not plan:
        return ("offmap_no_plan" if agent.position is None else "onmap_no_plan"), detail
    if agent.position is None:
        if t < plan[0][2]:
            return "offmap_wait_departure", detail
        if action == FWD:
            return "offmap_depart_command", detail
        return "offmap_ready_no_command", detail

    if action not in (STOP, DO):
        return "moving_command", detail

    service_stops = diag.get("service_stops", set())
    if h in service_stops:
        return "service_stop", detail

    for name, removed in diag.get("removed", {}).items():
        if h in removed:
            detail["desired"] = _fmt_cell(removed[h])
            return f"held_by_{name}", detail

    blocked = diag.get("blocked", {})
    if h in blocked:
        reason, blocker, desired = blocked[h]
        detail["blocker"] = "" if blocker is None else str(blocker)
        detail["desired"] = _fmt_cell(desired)
        return reason, detail

    i = _plan_index(planner, h, agent)
    if i is None:
        return "off_plan_desync_wait", detail
    detail["plan_index"] = str(i)
    if i + 1 >= len(plan):
        if plan[-1][0] == tuple(agent.target):
            return "at_target_not_done_yet", detail
        return "parked_truncated_plan_end", detail
    k = max(1, int(round(1 / max(agent.speed_counter.speed, 1e-9))))
    scheduled_move = plan[i][3] - k
    detail["lateness"] = str(max(0, t - scheduled_move))
    if t < scheduled_move:
        return "timetable_wait", detail
    if h not in diag.get("desired_initial", {}):
        return "no_desired_unclassified", detail
    return "stopped_unclassified", detail


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--policy", default="submission.my_policy.MyPolicy")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--out", default="diagnostics/runs/stoppage_audit")
    ap.add_argument("--timeline-every", type=int, default=100)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    env, _ = RailEnvPersister.load_new(args.scenario, rewards=ECML2026Rewards())
    env.reset(random_seed=args.seed)
    policy = _load_class(args.policy)()
    planner = getattr(policy, "_planner", policy)
    planner.collect_diagnostics = True
    for item in args.set:
        name, raw = item.split("=", 1)
        setattr(planner, name, _parse_value(raw))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = env.get_num_agents()
    handles = list(range(n))
    cap = env._max_episode_steps
    t0 = time.time()
    reason_steps = Counter()
    reason_agents = defaultdict(set)
    final_reason = {}
    final_detail = {}
    blocker_counts = Counter()
    timeline_rows = []

    for step in range(cap):
        if time.time() - t0 > args.time_limit:
            break
        actions = policy.act_many(handles, [env] * n)
        diag = getattr(planner, "last_diag", {})
        step_counts = Counter()
        for h, agent in enumerate(env.agents):
            action = actions.get(h, DO)
            reason, detail = _classify(h, agent, action, planner, diag, env._elapsed_steps)
            if reason in ("done", "moving_command", "offmap_depart_command"):
                continue
            reason_steps[reason] += 1
            reason_agents[reason].add(h)
            step_counts[reason] += 1
            final_reason[h] = reason
            final_detail[h] = detail
            if detail.get("blocker"):
                blocker_counts[int(detail["blocker"])] += 1
        if step % args.timeline_every == 0:
            row = {"step": step}
            for reason, count in step_counts.most_common(12):
                row[reason] = count
            timeline_rows.append(row)
        _, _, dones, _ = env.step(actions)
        if dones["__all__"]:
            break

    rewards = list(env.rewards_dict.values())
    normalized = env.rewards.normalize(*rewards, num_agents=n, max_episode_steps=env._max_episode_steps)
    complete = sum(1 for a in env.agents if a.state == TrainState.DONE)
    states = Counter(_state_name(a.state) for a in env.agents)

    summary_rows = []
    for reason, count in reason_steps.most_common():
        final_count = sum(1 for h, a in enumerate(env.agents)
                          if a.state != TrainState.DONE and final_reason.get(h) == reason)
        summary_rows.append({
            "reason": reason,
            "agent_steps": count,
            "unique_agents": len(reason_agents[reason]),
            "final_unfinished_agents": final_count,
        })
    _write_csv(out / "stoppage_summary.csv", summary_rows,
               ["reason", "agent_steps", "unique_agents", "final_unfinished_agents"])

    final_rows = []
    for h, agent in enumerate(env.agents):
        if agent.state == TrainState.DONE:
            continue
        detail = final_detail.get(h, {})
        final_rows.append({
            "handle": h,
            "state": _state_name(agent.state),
            "reason": final_reason.get(h, ""),
            "position": _fmt_cell(tuple(agent.position) if agent.position is not None else None),
            "target": _fmt_cell(tuple(agent.target)),
            "blocker": detail.get("blocker", ""),
            "desired": detail.get("desired", ""),
            "plan_index": detail.get("plan_index", ""),
            "lateness": detail.get("lateness", ""),
        })
    _write_csv(out / "final_unfinished_agents.csv", final_rows,
               ["handle", "state", "reason", "position", "target", "blocker",
                "desired", "plan_index", "lateness"])

    blocker_rows = []
    for h, count in blocker_counts.most_common(50):
        agent = env.agents[h]
        blocker_rows.append({
            "blocker": h,
            "count": count,
            "state": _state_name(agent.state),
            "position": _fmt_cell(tuple(agent.position) if agent.position is not None else None),
        })
    _write_csv(out / "top_blockers.csv", blocker_rows,
               ["blocker", "count", "state", "position"])

    timeline_fields = ["step"] + sorted({k for row in timeline_rows for k in row if k != "step"})
    _write_csv(out / "stoppage_timeline.csv", timeline_rows, timeline_fields)

    print(f"complete={complete}/{n},norm={normalized:.9f},steps={step + 1}/{cap},wall={time.time() - t0:.2f}s")
    print(f"states={dict(states)}")
    print(f"out={out}")
    print("top_reasons=" + ";".join(
        f"{r['reason']}:{r['agent_steps']}:{r['final_unfinished_agents']}" for r in summary_rows[:10]))


if __name__ == "__main__":
    main()
