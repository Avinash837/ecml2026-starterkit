# ECML 2026 Flatland — Dispatcher Investigation Summary

_Generated during the autonomous overnight session of 2026-06-02. Nothing committed to git._

## TL;DR

**The submitted dispatcher (v5, leaderboard score 8.24) is a correct, near-ceiling solution for its
architecture. An exhaustive audit found NO bug and NO missed detail. Every one of ~24 tested
improvements is either noise or strictly worse, because the binding constraint is a fundamental
*realizability gap*, not a fixable flaw. Recommendation: bank 8.24.**

---

## 1. What v5 is

- **Algorithm:** prioritized (sequential, greedy) planning + a-priori **space-time reservation**
  (each cell/edge booked for a time window), executed with chain-resolution.
- **Routing:** fast penalized-Dijkstra k-shortest paths (`topology.k_routes`, K=4) — direct to
  the final target. Intermediate timetable stops are intentionally **skipped**.
- **Deadlock-free by construction** in the plan (verified: 0 cell conflicts, 0 head-on conflicts).
- **Key design wins:** never park a stuck train mid-map (delay its departure off-map instead);
  K-route diversity to spread traffic; remove-at-target frees the platform cell.

## 2. The mechanics are PROVEN correct (full audit)

| Layer | Check | Result |
|---|---|---|
| Topology graph | every Flatland transition present? | 6114 edges, **0 missing / 0 extra** |
| Actions/direction | planned moves execute? | **0** fallbacks, **0** failed moves, **0** desyncs (full episode) |
| Speed | fractional speeds modeled? | `_kof` exact (0.25→4 steps/cell); ~75% of trains are slow |
| Plan | conflict-free + speed-correct? | **0** cell conflicts, **0** head-on conflicts |
| Route diversity | k_routes distinct? | 3.55 distinct / 4, 25% overlap |
| Feasibility | can every train finish in time? | **0%** infeasible |
| Executor | does it leak vs the plan? | open-loop == chain-resolution (no leak) |

## 3. The core finding — the realizability gap

The plan schedules **158/200 trains to complete, conflict-free**. Execution delivers **~45**.
That 113-train gap is the whole story:

- The conflict-free plan assumes a synchronized clock + deterministic motion.
- Flatland executes all trains **simultaneously**; under any drift a train slips off its slot.
- On a single-track section, a slipped train meets an opposing train → **irreversible head-on**
  (Flatland trains cannot reverse) → the corridor dies → the queue behind it cascades.
- ~7-8 such events per episode convert ~110 would-complete trains into stuck ones.

This is **the** central hard problem of Flatland. It is not a bug; it is the gap between one-shot
MAPF planning and reactive multi-agent execution at high density.

### 3a. Backtrack to the root (the decisive diagnostic)

A blocking-graph + trajectory backtrack (`deadlock_backtrack.py`) showed the structure precisely:
- **~7 head-on "cores" per episode are the root of 100% of the stuck cascade** (big200: 7 cores →
  110/110 stuck trains feed them; big250: 7 → 107/107). Break the 7 cores and ~all stuck trains free.
- The opposing pair typically enters the shared single-track segment within **~5 steps** of each
  other (drift), and most core segments **do** have a tight parallel loop.
- So the theoretically-correct fix is **reactive meet-pass**: divert the entering train onto the
  parallel loop instead of holding it. Built and tested (`meet_pass`): **marginal (~+0.7pp), and the
  head-on count stays ~7 — the cores RELOCATE, they don't break.**
- **Why (the irreducible floor):** a real passing loop is a *siding* where a train **stops** while
  the opposing one passes. This map has **no sidings** — 0 dead-ends, single-platform stations, a
  dense mesh of *through-routes*. The "tight loops" are just *other* contested single-tracks, so
  diverting moves the train into a different single-track and a new core forms. Combined with **no
  reversal anywhere** (0 dead-ends), head-on cores are **structural**: nowhere to reverse, nowhere
  to wait safely, and holding/diverting only relocates the conflict.

## 4. Everything tested (and why each failed)

| Approach | Result | Why |
|---|---|---|
| Real flatland-baselines DLA (min_free 1/2, competition cfg) | **8.8% vs v5 19.6%**, 15× slower, cfg crashes | reactive evasion loses to reservation at density |
| Online re-dispatch (replan_interval) | 22%→5% | discards good plans, thrashes |
| Surgical stuck-replan | neutral/worse | stuck trains are in irreversible positions; can't reroute |
| Release valve (re-release never-departed) | 0 effect | network reservation-saturated; late releases can't finish in horizon |
| block_lock (single-track directional interlock) | neutral/negative | preventing a head-on just relocates the jam |
| signal_guard (home-signal junction protection) | +1pp small malf only, −1pp dense | density-dependent, not clean |
| Criticality routing (route by blast-radius danger) | high-variance noise; big250 −2.2pp | danger concentrated on FORCED chokepoints (can't avoid) |
| Directional separation (double-track up/down line) | noise; head-ons NOT reduced | "passing loops" are long detours, not adjacent parallel track |
| Longest-distance / speed priority | high-variance noise; craters dense scenes | pure sequencing can't add capacity |
| Following buffer / slack | hurts (big250 19.6→14.4) | cuts density; gap isn't a micro-stall cascade |
| Greedy advance (drop timetable gate) | == baseline | trains already advance ASAP; ~0 wasted idle |
| LNS / ES priority / RL hybrids | flat/marginal | engine ceiling, not sequencing |
| **multistop=True (serve all stations)** | **0.73→0.53 score, 32.5%→5%** | threading waypoints over-lengthens routes; 80/120 never depart |

## 5. Multi-station / platform ideas (specifically investigated)

The real competition is multi-stop (3-6 stations served). Tested on a generated `line_length=4`
env with competition-accurate scoring:

- **Serving intermediates is far worse** (score 0.73→0.53, completion 32.5%→5%). The −50/intermediate
  penalties are *dwarfed* by the value of completing trains. v5's skip-intermediates default is correct.
- **No platform alternatives exist** — every waypoint is a single fixed cell ({1:480}). Meet-pass /
  platform-routing has no lever.
- **Intermediate stations are single-platform** — so "hold at the station as a safe buffer" doesn't
  hold; a waiting train blocks the station.

## 6. Why we're capped — two gaps

1. **Planning gap:** greedy prioritization (not global optimum). Early commitments block later trains
   (158 planned, 42 abandoned). A global solver could fit more.
2. **Realizability gap:** even the 158-train plan only realizes ~45 under execution drift.

Closing either needs a **different algorithm class** — online re-planning that continuously
re-derives the schedule from reality, or near-optimal coupled MAPF (CBS / MAPF-LNS2). Both are
multi-day builds with real risk of landing *below* 8.24.

## 7. Recommendation

**Bank 8.24.** It beats the reference baselines, the submission files are synced and verified
(`submission/` reproduces the proven baseline exactly), and ~24 experiments show no safe improvement.
The remaining gap to the leader (10.73) is most likely (a) the real competition map being less dense
than our random proxies, and/or (b) the leader using a near-optimal coupled MAPF — neither closable
by a tweak before the 2026-06-08 deadline.

**Deploy steps (require the user):** trigger the `docker` GitHub action on the fork → grant the
Flatland Competition account access to the image → enter the image URL at competition.flatland.cloud.
