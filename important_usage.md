# Important Usage Notes

Use this file before starting any new V6 debugging run. The goal is to avoid
repeating the same setup, baseline checks, and conclusions.

## Current Working Branch

- Worktree: `C:\Users\Avinash\Downloads\submission_interlocking\submission_pkg\.branch_inspect_v5_dispatcher`
- Branch: `v6-dispatcher-multistop`
- Code baseline before late-guard experiments: `fb18be6`
- Documentation/baseline checkpoint commit: `8b44c4d`
- Stable parent: `v5-dispatcher-submission`
- Do not mix this with `v7-multistop-scaleup` unless the user explicitly asks.

## Submission Defaults

The submitted policy is `submission.my_policy.MyPolicy`.

Current V6 default behavior:

- Based on stable V5 dispatcher.
- Completion-first direct routing stays active.
- Opportunistic low-density intermediate stops are enabled.
- Route-native stop preference is only enabled for very small maps.
- `dir_weight=0.5` only when `env.get_num_agents() <= 60`, else `0.0`.
- `exec_fast_first=True`.
- `late_segment_guard=True`, threshold `30`, to prevent stale-timetable/malfunction
  corridor swaps before opposing trains enter the same single-track segment.
- `signal_guard=False`, `block_lock=False`, `stuck_replan=False`, `frozen_reroute=False`.

Do not enable old broad knobs by default without comparing against
`diagnostics/baseline_stats.csv`.

## Known Saved Scenarios

Saved proxy scenarios live outside this branch:

`C:\Users\Avinash\Downloads\submission_interlocking\submission_pkg\scenarios\proxy\levels`

Important files:

- `L0_s4_a28_ll6_clean_seed11.pkl`
- `L1_s1_a50_ll3_clean_seed11.pkl`
- `L2_s2_a150_ll4_clean_seed11.pkl`
- `L3_s1_a50_ll3_malf540_seed11.pkl`
- `L4_s2_a150_ll4_malf360_seed11.pkl`
- `L2_s4_a532_ll6_clean_seed11.pkl`

## Baseline CSV

Baseline numbers are stored in:

`diagnostics/baseline_stats.csv`

Stoppage-specific experiments are stored in:

- `diagnostics/stoppage_experiments.csv`
- `diagnostics/stoppage_audit_summary.md`
- audit CSVs under `diagnostics/runs/`

The `source` column matters:

- `prior_user_trace`: pasted or earlier trace output. Keep it for history.
- `prior_verified_notes`: verified before this helper existed.
- `eval_pickle_load_new`: reproducible with `diagnostics/eval_pickle.py`, using
  `RailEnvPersister.load_new(..., rewards=ECML2026Rewards())`.

Before judging any new patch:

1. Run the same scenario.
2. Compare `complete`, `normalized_reward`, `planned_to_target`, `no_plan`, and final state counts.
3. For malfunction scenarios, do not trust a small normalized reward gain unless completions or blocking behavior also improve.

## Docker Validation

Known image from the V6 smoke run:

`submission/v6-multistop-smoke`

Run a saved scenario from this worktree:

```powershell
docker run --rm --entrypoint bash `
  -v "C:\Users\Avinash\Downloads\submission_interlocking\submission_pkg\.branch_inspect_v5_dispatcher:/workspace" `
  -v "C:\Users\Avinash\Downloads\submission_interlocking\submission_pkg\scenarios:/scenarios:ro" `
  submission/v6-multistop-smoke `
  -lc "source /home/conda/.bashrc && conda activate flatland-baselines && cd /workspace && python diagnostics/eval_pickle.py /scenarios/proxy/levels/L4_s2_a150_ll4_malf360_seed11.pkl"
```

Optional planner toggles can be passed as:

```powershell
python diagnostics/eval_pickle.py /scenarios/proxy/levels/L4_s2_a150_ll4_malf360_seed11.pkl --set frozen_reroute=true
```

Current helper-loader comparison on `L4_s2_a150_ll4_malf360_seed11.pkl`:

- V6 default: `45/150`, norm `0.682238640`
- `late_segment_guard=true`: `48/150`, norm `0.686561126`
- V6 default after enabling `late_segment_guard`: `48/150`, norm `0.686561126`
- `L3_s1_a50_ll3_malf540_seed11.pkl` improved from `32/50` to `41/50`
- Clean guard `L2_s2_a150_ll4_clean_seed11.pkl` stayed `125/150`, norm `0.885435356`

Run the stoppage audit:

```powershell
python diagnostics/stoppage_audit.py /scenarios/proxy/levels/L4_s2_a150_ll4_malf360_seed11.pkl --out diagnostics/runs/L4_s2_v6_guard
```

Current stoppage decisions:

- `late_guard_release_after` is ruled out: it removes guard holds but recreates adjacent swaps.
- `frozen_reroute` is not a safe default: L4 improves `48 -> 50`, but one scenario takes ~`132s`.
- `release_interval` and `signal_guard` are neutral with the current guard.
- Next possible direction: reroute only long-held late-guard roots from a switch/clear alternate, instead of releasing them into the guarded segment.

## Main Diagnosis So Far

Clean medium density is not execution blocked:

- `L2_s2_a150_ll4_clean_seed11.pkl` completed exactly the trains it planned.
- The remaining trains were intentionally left off-map because no full conflict-free route fit.

Medium malfunction density is blocked by formed head-to-head corridor swaps:

- `L4_s2_a150_ll4_malf360_seed11.pkl` baseline is `37/150`.
- It planned `125` trains to target and left `25` with no plan.
- Final state included many `STOPPED` trains and only `37` completed.
- Concrete failures included adjacent swaps like train `0` vs train `25` around row `75`.

The important lesson: after a plain-corridor adjacent swap forms, local repair is usually too late.
The next useful direction is prevention before trains enter an opposing corridor late relative to
the original timetable.

## Rules For Future Work

- Start by reading this file and `diagnostics/baseline_stats.csv`.
- Change one mechanism at a time.
- Test on `L4_s2_a150_ll4_malf360_seed11.pkl` and at least one clean guard scenario.
- Keep V6 defaults conservative unless a patch improves malfunction behavior without hurting clean throughput.
- Commit V6 work on `v6-dispatcher-multistop`; do not silently move it to V7.
