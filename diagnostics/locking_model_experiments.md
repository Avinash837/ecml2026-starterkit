# Locking Model Experiments

Date: 2026-06-29
Branch: `codex/clean-complete-v8`

## Baseline Audit

Scenario: `L2_s4_a532_ll6_clean_seed11.pkl`

Current clean-completion model:

- `complete=263/532`
- `norm=0.763103964`
- `planned_to_target=356`
- `no_plan=176`
- final states: `DONE=263; STOPPED=88; READY_TO_DEPART=181`

Stoppage audit output:

- `diagnostics/runs/L2_s4_clean_model/stoppage_summary.csv`
- `diagnostics/runs/L2_s4_clean_model/final_unfinished_agents.csv`
- `diagnostics/runs/L2_s4_clean_model/top_blockers.csv`
- `diagnostics/runs/L2_s4_clean_model/stoppage_timeline.csv`

Main reasons:

- `offmap_no_plan`: 176 final unfinished agents
- `blocked_by_stopped_train`: 86 final unfinished agents
- `blocked_by_adjacent_swap`: 2 final unfinished agents

The highest blocker concentration is around cells `102..111 / 42..51`, especially
`106:46`, `106:47`, `106:45`, and `104:46`. This is a nearby junction/corridor grid,
not a single station-cell problem.

## Ruled Out

Increasing route alternatives did not help 532 initial fit:

- `K=4`: `planned=345`, `no_plan=187`
- `K=6`: `planned=344`, `no_plan=188`

Increasing reservation wait did not help 532 initial fit:

- `max_wait=800`: `planned=345`, `no_plan=187`
- `max_wait=1200`: `planned=345`, `no_plan=187`
- `max_wait=1800`: `planned=345`, `no_plan=187`

Soft route pressure over nearby station/junction grids was tested and removed. It
slightly changed initial route fit, but damaged full execution:

- `grid_weight=0.4`, `L2_s2_a150_ll4_clean_seed11.pkl`: `108/150`, worse than `150/150`
- `grid_weight=0.4`, `L1_s4_a210_ll6_clean_seed11.pkl`: `164/210`, worse than `210/210`

Runtime grid-admission gating was also tested and removed. It preserved the smaller
clean cases, but hurt the 532 stress case:

- `grid_admission=true`, `L2_s2_a150_ll4_clean_seed11.pkl`: `150/150`
- `grid_admission=true`, `L1_s4_a210_ll6_clean_seed11.pkl`: `210/210`
- `grid_admission=true`, `L2_s4_a532_ll6_clean_seed11.pkl`: `256/532`, worse than `263/532`

## Kept Direction

The bottleneck is two-layered:

1. Offline admission: 176 trains still never receive a complete plan after release retries.
2. Live execution: the entered trains still form stopped chains around one nearby grid.

The useful direction was not a hard station hold. It was clean-only surgical stuck
replanning: after 40 blocked steps, replan only that train from its current position
while leaving the rest of the timetable intact.

Result:

- `L2_s4_a532_ll6_clean_seed11.pkl`: `263/532 -> 294/532`
- `L2_s2_a150_ll4_clean_seed11.pkl`: preserved at `150/150`
- `L1_s4_a210_ll6_clean_seed11.pkl`: preserved at `210/210`
- `L4_s2_a150_ll4_malf360_seed11.pkl`: preserved at `52/150` because the lever is clean-only
