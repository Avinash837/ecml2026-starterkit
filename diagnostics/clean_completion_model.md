# Clean Completion Model

Goal: improve clean level completion without using level IDs or scenario names.

The model uses a workload ratio:

`workload = num_agents * average_waypoints / max_episode_steps`

This estimates train-service demand per available timestep. The policy uses it
only when `env.malfunction_process_data` is clean: near-zero malfunction rate and
zero malfunction duration.

Two continuous levers come from this workload:

- Route shadow price: `load_weight` increases with workload and caps at `0.2`.
  This makes route search pay a cost for already-loaded cells, spreading traffic
  away from overloaded corridors before stopped chains form.
- Late admission cadence: `release_interval` is derived from workload and
  quantized to 100-step timetable buckets. Light clean cases release at `500`;
  heavier clean cases release at `400`.

This is not keyed to `level_1`, `level_2`, or exact agent counts. It is a small
model of congestion pressure and available horizon.

## Validation Snapshot

See `diagnostics/clean_completion_model_results.csv`.

Key proxy results:

- `L2_s2_a150_ll4_clean_seed11`: `125/150 -> 150/150`
- `L1_s4_a210_ll6_clean_seed11`: `118/210 -> 210/210`
- `L2_s4_a532_ll6_clean_seed11`: `171/532 -> 263/532`
- `L4_s2_a150_ll4_malf360_seed11`: unchanged at `52/150`

## Ruled Out

Naive SIPP/time-space repair was tested as an initial and release-time repair.
It added more planned trains but worsened execution by creating stopped chains.
The useful mechanism is route-load shadow pricing plus workload-timed late
admission, not simply forcing more trains into the plan.
