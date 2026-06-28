# Stoppage Audit Summary

Scenario audited:

`L4_s2_a150_ll4_malf360_seed11.pkl`

Current V6 default:

- `complete=48/150`
- `norm=0.686561126`
- final states: `DONE=48; STOPPED=62; MALFUNCTION=13; READY_TO_DEPART=24; MALFUNCTION_OFF_MAP=3`

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

## Next Useful Direction

The late guard is preventing real head-on failures. A blunt release is wrong. The next viable idea is not to release into the same blocked segment, but to create a targeted alternative for the six guard-root trains:

1. Detect a train held by `late_guard` for a long time.
2. Only if it is currently at a switch or has a clear alternate first step, try a reroute away from the guarded segment.
3. Otherwise keep holding, because releasing into the guarded segment recreates the adjacent swap.

That is a narrower version of `frozen_reroute`: reroute only long-held guard roots, not every train near every frozen segment.
