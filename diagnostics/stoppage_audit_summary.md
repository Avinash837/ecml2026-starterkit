# Stoppage Audit Summary

Scenario audited:

`L4_s2_a150_ll4_malf360_seed11.pkl`

Current V6 default:

- Before targeted late-guard reroute: `complete=48/150`, `norm=0.686561126`
- After targeted late-guard reroute: `complete=52/150`, `norm=0.703267077`
- final states after reroute: `DONE=52; STOPPED=58; MALFUNCTION=13; READY_TO_DEPART=24; MALFUNCTION_OFF_MAP=3`

## Final Stoppage Reasons

The direct final reasons are:

- `blocked_by_stopped_train`: 46 final trains
- `offmap_no_plan`: 22 final trains
- `malfunction_on_map`: 13 final trains
- `blocked_by_malfunction_train`: 8 final trains
- `held_by_late_guard`: 6 final trains
- `malfunction_off_map`: 3 final trains
- `blocked_by_adjacent_swap`: 2 final trains
- `offmap_ready_no_command`: 2 final trains

But `blocked_by_stopped_train` is mostly a symptom. Following blocker chains gives the root causes:

- `held_by_late_guard`: 40 unfinished trains
- `malfunction_on_map`: 32 unfinished trains
- `offmap_no_plan`: 22 unfinished trains
- `malfunction_off_map`: 3 unfinished trains
- adjacent-swap cycles: 3 unfinished trains
- `offmap_ready_no_command`: 2 unfinished trains

## Decisions

- `late_guard_release_after`: ruled out. It removes guard-root holds, but recreates adjacent-swap cycles and does not increase completion.
- `frozen_reroute`: helps L4 (`48 -> 50`) but is too slow (`132s` for one 150-agent scenario) and does not help L3 completion, so it is not safe as a default.
- `release_interval`: ruled out again; no completion change.
- `signal_guard`: ruled out; neutral with the current guard.
- `late_guard_reroute_after=200`, `max_late_guard_reroutes=1`, cooldown `200`: kept for maps with more than 60 agents. It improves L4 (`48 -> 52`) without changing L2 clean or dense clean; it is disabled on low-density maps because forcing it on L3 kept completion but lowered normalized reward slightly.

## Current Remaining Roots

After the targeted reroute, blocker-chain roots on L4 are:

- `offmap_no_plan`: 22 unfinished trains
- `malfunction_on_map`: 21 unfinished trains
- `held_by_late_guard`: 21 unfinished trains
- `timetable_wait`: 19 unfinished trains
- residual adjacent-swap cycles: 10 unfinished trains
- `malfunction_off_map`: 3 unfinished trains
- `offmap_ready_no_command`: 2 unfinished trains

## Next Possible Direction

The largest remaining fixable-looking bucket is no longer one clear blocker. The next audit should inspect the 19 final `timetable_wait` roots and the remaining 21 `held_by_late_guard` roots to see whether they are near targets/dead routes or just waiting behind unresolved malfunction queues. Do not use blunt release; it recreates swaps.
