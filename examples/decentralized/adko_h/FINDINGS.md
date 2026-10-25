# ADKO-H findings

Newest first. One entry per finding: date, what was run (oracle, arms, seeds, budget, model),
the number, and what it means. Numbers come from `report.py` or `run.json`, never from memory.

## 2026-10-24 — batch 5: relaying inputs two hops makes the wide channel the best arm

`zaleplon_mpo`, Sonnet-medium planning, hit at 0.4, random 4-regular graph. New switch
`--input-hops 2`: each token also carries the inputs its sender heard the round before (8 bits a
coordinate), read only by the peer-input filter. Baselines are saved seeds from batches 2–4, so
every comparison is paired. 66 planner runs, 0 errors.

| Arm (10 agents × 1,000 rounds, seeds 42–49) | diverse hits | duplicates | distinct molecules | bits/token | $/run |
|---|---|---|---|---|---|
| `adko` | 186.1 | 1,223 | 8,777 | 533 | – |
| `adkoh3-skip` | 399.6 | 1,217 | 8,783 | 1,274 | 3.61 |
| `adkoh3-rich-skip` | 393.0 | 2,917 | 7,083 | 1,950 | 5.38 |
| `adkoh3-skip` + `--input-hops 2` | 457.1 | 62 | 9,939 | 2,298 | 3.44 |
| **`adkoh3-rich-skip` + `--input-hops 2`** | **501.6** | 600 | 9,400 | 2,960 | 5.21 |

- **The relay works on both channels.** Compact +57.5 (7/8, p = 0.039), wide +108.6 (8/8,
  p = 0.008); distinct molecules +1,156 and +2,317, scaffolds +911 and +1,893. Wide + relay is
  +102.0 over `adkoh3-skip` at 8/8. AUC top-10 and the best score do not move. Without a planner
  (`adko`, token noise 0.1732, filter on) it is +17.9 (6/8, p = 0.062), duplicates 606 → 30.
- **It pays late.** Nothing before 2,500 evaluations; the gain grows to the end of the run. Wide
  minus compact, both relayed: +14.5 at 1,000 (8/8), +52.9 at 2,500 (7/8, p = 0.016), +44.5 at
  10,000 (6/8, p = 0.078). Batch 4's crossing near 5,000 is gone.
- **Why it works:** before it, 96–97% of the filtered arms' repeats were of runs made exactly two
  hops away, which a one-hop filter cannot see. The 0.30 ball itself removes almost nothing else:
  at most 0.41% of the molecules any arm ran lie inside another run's simulated report ball.
- **What the wide channel still wastes:** about 270 repeats a run come from the filter's fallback
  (when every candidate drawn in a region was reported, the unfiltered draw stands; 49% of its
  picks see a short pool), and about 330 from beyond two hops — verdicts and hit boxes travel
  three hops, inputs two.
- **500 evaluations** (10 agents × 50 rounds, campaigns 5–15, seeds 42–60, filter on): wide 16.8
  against compact 12.3 diverse hits, 11 better / 4 worse / 4 tied, p = 0.057; 5–11 duplicates a
  run, so the relay has nothing to do there.
- **100 agents × 100 rounds** (the same 10,000 evaluations, seeds 42–47, compact channel): plain
  `adko` 134.2, `adkoh3-skip` 174.5 (+40.3, 6/6, p = 0.031), with the relay 184.3 (+9.8 over no
  relay, 5/6, p = 0.41). Two hops reach 15 of 99 agents and see 15% of the repeats; a relay
  would need four hops to see 79%, at about 12,600 bits a token in this encoding.

## 2026-10-24 — batches 3–4: the algorithm settles; redundancy is removed on mechanism, not on the endpoint

`zaleplon_mpo`, the same settings throughout (10 agents × 1,000 rounds, random 4-regular graph,
seeds 42–49, campaigns of 20–60, 10,000 evaluations, hit at 0.4, Sonnet-medium planning). Code
`7b584baa`. Batch 3: the composed arm and one ablation per part, 5 arms × 8 seeds, 0 errors.
Batch 4: the same two channels with `--skip-peer-inputs --skip-radius 0.30`, 2 arms × 8 seeds,
0 errors. Batch 4's baselines are batch 3's saved seeds, so every comparison is paired.

| Arm | diverse hits | hit evals | evals / hit scaffold | re-found | distinct molecules | scaffolds | AUC top-10 | $/run |
|---|---|---|---|---|---|---|---|---|
| `adko` | 186.1 | 260.0 | 1.40 | 25.6% | 8,777 | 6,473 | 0.472 | – |
| `adkoh2` | 336.0 | 608.2 | 1.81 | 41.2% | 7,770 | 6,436 | 0.489 | 3.63 |
| `adkoh3` | 358.4 | 605.4 | 1.68 | 37.6% | 7,972 | 6,661 | 0.489 | 3.65 |
| `adkoh3-blind` | 331.2 | 530.9 | 1.60 | 33.9% | 8,147 | 6,548 | 0.482 | 3.62 |
| `adkoh3-nofindings` | 333.4 | 531.9 | 1.59 | 34.3% | 8,087 | 6,870 | 0.480 | 3.00 |
| `adkoh3-penalty` | 354.6 | 575.5 | 1.62 | 35.5% | 8,179 | 6,819 | 0.484 | 3.58 |
| `adkoh3-rich` | 367.2 | 820.9 | 2.26 | 52.9% | 6,646 | 5,509 | 0.487 | 5.43 |
| `adkoh3-rich-skip` | 393.0 | 728.8 | 1.86 | 43.1% | 7,083 | 5,836 | 0.484 | 5.38 |
| **`adkoh3-skip`** | **399.6** | 547.4 | **1.37** | **23.8%** | 8,783 | **7,330** | 0.482 | 3.61 |

- **Batch 3: no ablation can be told apart from the composed arm.** All five composed arms beat
  `adko` at every seed (+145.1 to +181.1, p = 0.008 each), and none beats another: `adkoh3` over
  `-penalty` +3.8 (6/8, p = 0.55), over `-blind` +27.1 (6/8, p = 0.25), over `-nofindings` +25.0
  (7/8, p = 0.15); `-rich` over `adkoh3` +8.9 (5/8, p = 0.77). The paired spread is 53–59 hits,
  about the two arms' spreads added in quadrature, so the seed does not pair these arms and more
  seeds would cost 35 to 1,700 of them. **The planner's region choice stopped being the
  constraint**; what limits the endpoint is that a third to a half of every hit found is one
  another agent already had.
- **The token noise is free.** `adkoh3` against clean-channel `adkoh2` (the same method, noise
  the only difference) is +22.4 (4/8, p = 0.46). The batch-3 pilot's −65 on seed 42 was a draw.
- **Naming the objective is no longer a cost.** `adkoh3` over `adkoh3-blind` is +27.1 (6/8).
  Batch 2's +58 for blind was measured with claims *off* (`adkoh2-noclaims-blind` 308.0 against
  `adkoh2-noclaims` 250.0); with claims on and findings fed back the sign reverses.
- **Batch 4: the peer-input filter does everything it was built to do, at 8 of 8 seeds.** On the
  compact channel: duplicate evaluations 2,028 → 1,217, distinct molecules +811, distinct
  scaffolds +669, evaluations per new hit scaffold 1.68 → 1.37, each at 8/8 (p = 0.008 each);
  cross-agent re-finds 37.6% → 23.8%, below plain `adko`'s 25.6%. It runs *fewer* hit
  experiments (547 against 605) and converts them better. #Circles is flat (−0.25, 4/8), AUC
  0.482 against 0.489 (2/8), region hit density unchanged (2.75% against 2.61%): the planner
  behaves the same, and what changed is what reaches the oracle.
- **Its endpoint gain is positive but not significant.** +41.2 diverse hits (5/8, p = 0.20), and
  216 of the 330-hit total is seed 42 alone — the one seed the pilot ran, and `adkoh3`'s worst
  of the eight. Over seeds 43–49 it is +16.3 (4/7). 80% power would need 29 seeds (~$150).
  Adopt the filter because it is free and positive on every measure it targets, not because the
  endpoint proves it. At a 0.35 threshold it is +104.9 (7/8, p = 0.055) and at 0.45 +2.4 (4/8):
  it widens the band of moderately good scaffolds rather than reaching higher peaks.
- **Coverage stops being the price of hits.** `adkoh3-skip` holds the study's records for both
  diverse hits (399.6) and distinct scaffolds (7,330 — above `random-campaign`'s 7,077 and 857
  above `adko`), with distinct molecules at parity with `adko` (8,783 against 8,777). Every
  earlier planner arm gave up 500–2,100 distinct molecules. It is also the only arm whose
  discovery rate does not decay: 36.5 new hit scaffolds per 1,000 evaluations in the first
  quarter of the run and 36.8 in the last, where the other composed arms fall to 72–95%.
- **The collateral cost of deleting a ball rather than a point did not appear here.** On the
  noisy no-planner arm the radius skip lost 2.9 diverse hits (`adko-noise-skip` against
  `adko-noise`), because G is largest exactly where a reported point sits. On the compact
  channel the ball covers 89,167 of 2,560,000 candidate slots a run (3.5%) and distinct
  molecules return to `adko`'s level, so a nearest-molecule-only filter has nothing to recover.
- **The wide prose channel buys nothing at the full budget — but wins at a short one.** Filtered,
  the two channels are indistinguishable at 10,000 evaluations (`adkoh3-skip` −
  `adkoh3-rich-skip` = +6.6, 4/8, p = 0.98, 1,369 seeds to resolve), while the wide one costs 49%
  more a run, 1,700 distinct molecules and 1,495 scaffolds (8/8 each), and skips 24% of every
  pool it draws (44% at its worst seed) because its small boxes mine out and fail to refill.
  Reading the same runs at earlier cut-offs reverses it: `adkoh3-rich-skip` leads
  `adkoh3-skip` by +19.4 at 1,000 evaluations (7/8, p = 0.039) and +45.8 at 2,500 (7/8,
  p = 0.047), by +18.2 and +47.8 against unfiltered `adkoh3` (p = 0.016 each), then loses the
  lead by 10,000. **This is the only contrast in the study that reaches p < 0.05 other than
  against the no-planner arms, and it is budget-dependent.** These are truncations of long runs,
  not short runs: a 500-evaluation run would plan about five times as often per evaluation, so
  the result is suggestive about the regime and not a measurement of it.
- **Settled configuration:** prompt v2; claims and verdicts on, carrying findings; no claim
  penalty; the objective named; token input noise at 0.1732; the compact channel; and
  `--skip-peer-inputs --skip-radius 0.30`. Three of the eight decisions rest on cost, simplicity
  or mechanism rather than on the endpoint, because the endpoint cannot separate them at any
  number of seeds worth buying.
- **Cost and speed.** `adkoh3-skip` $3.61 a run (91 batched gateway calls carrying 249 campaign
  designs, one design per 40 evaluations, 1,274 bits a knowledge token); the wide channel $5.38
  and 1,950 bits. Batch 4 cost $71.93 for 16 runs; its main stage, 14 runs 8 at a time, took
  50 minutes. Designs scale with evaluations ÷ campaign length, not with agent count.
- **What this does not establish.** One oracle, one pool, one model, ten agents. Nothing further
  on `zaleplon_mpo` would change a design decision; every open question is about transfer.

## 2026-10-23 — batches 1–2: the first planner's regions were a prior; feedback about hits makes it plan

`zaleplon_mpo` with the first screen's settings (10 agents × 1,000 rounds, random 4-regular
graph, seeds 42–49, campaigns of 20–60; the two region controls use 30, as that planner chose).
Code `4c03c879`. A rerun of `adko` with it reproduced all 8 seeds of the first screen record for
record. Planner arms through MAG: 40 runs, 0 errors, 0 fallbacks.

| Arm | Regions from | diverse hits | vs `adko` (p, every seed) | AUC top-10 | $/run |
|---|---|---|---|---|---|
| `adko` | — | 186.1 | – | 0.472 | – |
| `adko-skip` | — | 203.3 | +17.1 (0.195, no) | 0.469 | – |
| `adko-noise` | — | 201.6 | +15.5 (0.078, no) | 0.472 | – |
| `adko-noise-skip` | — | 198.8 | +12.6 (0.070, no) | 0.469 | – |
| `random-region` | random boxes on pool molecules, the planner's size | 171.3 | −14.9 (0.086, no) | 0.468 | – |
| `replay-regions` | the first `adkoh-noclaims` planner's regions, replayed | 229.1 | +43.0 (0.008, yes) | 0.476 | – |
| `adkoh`, Haiku | `claude-haiku-5-5@amazon`, prompt v1 | 126.4 | −59.8 (0.008, worse) | 0.464 | 0.24 |
| `adkoh-noclaims`, Haiku | the same, claims off | 215.0 | +28.9 (0.016, no) | 0.472 | 0.20 |
| `adkoh2-noclaims` | `claude-sonnet-5-5-medium@amazon`, prompt v2 | 250.0 | +63.9 (0.016, no) | 0.474 | 2.35 |
| `adkoh2-noclaims-blind` | the same, objective not named | 308.0 | +121.9 (0.008, yes) | 0.482 (yes) | 2.32 |
| `adkoh2` | prompt v2, claims carry findings, no claim penalty | **336.0** | +149.9 (0.008, yes) | **0.489** (yes) | 3.63 |

For comparison, the first screen: `adkoh-noclaims` 227.0, `adkoh` 147.3, `adko-div` 198.8.

- **The first planner's gain was where its regions sit, not planning.** Replayed in random order
  with no model, its regions give 229.1 against the planner's own 227.0 (+2.1, p = 0.63). Both
  beat random boxes of the same size at every seed (+57.9, +55.8). The rule's yield inside a
  region is the same for random and planner boxes (about 2.5×); what differs is the hit share
  inside them (0.91% against 1.36%; the pool's is 0.80%).
- **Prompt v2 makes the planner learn during a run.** v2 defines a hit, feeds back each
  campaign's hits, gives the agent's base rates, adds the tests `hits` and `new_hit_families`, lists
  the rules without a default and drops the instruction to work elsewhere. Region quality under v1
  is flat from the first campaign to the last (1.3–1.4%). Under v2 it rises to 1.6–1.7% (named,
  no claims), 2.0% (blind) and 3.1% (claims carrying findings). v2 agents keep a region after a
  campaign with 3 or more hits (overlap with the previous region 0.94) and leave after none
  (0.63); v1 agents moved every time (0.07).
- **Claims help once they carry findings.** In `adkoh2`, 24% of campaigns move into a region where
  a neighbour just found 3 or more hits, and they hit 7.0% of the time (other moves 3.7%, staying
  7.4%). Without verdicts, v2 agents almost never make that move (2% and 0.4% of campaigns).
  `adkoh2` beats `adkoh2-noclaims` by +86.0 at every seed.
- **Naming the objective anchors the planner.** The named v2 planner centres regions at logP 2.5
  (IQR 2.5–2.75), near zaleplon's own properties, which it cites. The pool's hits sit at logP 3.6
  (3.1–4.1). Blind regions centre at 3.25, and blind beats named by +58.0 at every seed. Named and
  without claims, v2 gains only +23.0 over v1 (6 of 8 seeds, p = 0.15).
- **v2's tests carry information.** 53–58% pass, against 96–99.5% under v1. Brier skill against
  the base rate is about 0 (−0.02 to +0.01): calibrated, not yet discriminating.
- **Haiku with prompt v1** picks richer regions than Sonnet when claims are off (hit share 1.93%
  against 1.37%), but runs 38% of those campaigns under `explore` or `novelty`, which hit about 1%
  against 4.1% under `adko`. Net −12.0 against Sonnet without claims (not significant) and −20.9
  with claims (every seed), at $0.20–0.24 a run.
- **Skipping reported inputs** sends 412 (exact match) or 616 (with noise, radius 0.30) more of the
  10,000 evaluations to new molecules, at every seed; diverse hits +12.6 to +17.1, not
  significant. Token noise at v2's level (0.173 of each coordinate's spread) costs nothing
  measurable. It defeats an exact-match skip but not a radius one: its 99% displacement (0.27) is
  far below the distance to the nearest other molecule (median 0.82). The v2 arms repeat more
  (7,770–8,232 distinct molecules against 8,777); skipping there is untested.
- **Cost.** $69.91 for the 40 MAG runs, pilot included. Prompt v2 is cheaper per run than v1
  ($2.32–3.63 against $4.21–5.12) because it drops the two lists of 20 scaffold SMILES.
- **Not yet shown:** which part of v2 does the work; claims with findings and a blind objective
  together; any other oracle.

## 2026-10-23 — first full screen with a MAG planner: claims cost hits

`zaleplon_mpo`, 10 agents × 1,000 rounds (10,000 evaluations), random 4-regular graph, seeds
42–49, campaigns of 20–60 in every arm. The four `adkoh*` arms planned with
`claude-sonnet-5-5-medium@amazon`; the other three use no model. Every model run planned with
the model: 10,475 designs, 0 errors, 0 fallbacks, no truncated replies.

| Arm | diverse hits | #Circles | hull | AUC top-10 | $/run |
|---|---|---|---|---|---|
| `adko` | 186.1 | 212.0 | 159.3 | 0.472 | – |
| `adko-div` | 198.8 | 216.8 | 166.3 | 0.467 | – |
| `random-campaign` | 76.8 | **274.0** | **213.1** | 0.449 | – |
| `adkoh` | 147.3 | 217.9 | 164.3 | 0.465 | 5.12 |
| `adkoh-noclaims` | **227.0** | 208.1 | 151.8 | **0.477** | 4.21 |
| `adkoh-noterritory` | 149.5 | 221.9 | 167.5 | 0.470 | 4.45 |
| `adkoh-nofootprint` | 162.9 | 215.6 | 159.5 | 0.466 | 4.53 |

- **The method as specified does not pass.** `adkoh` loses to `adko` on diverse hits at every
  seed (−38.9, p = 0.008) and gains nothing significant on #Circles, hull or AUC.
- **Without claims and verdicts it does.** `adkoh-noclaims` beats `adko`, `adko-div`,
  `random-campaign` and `adkoh` on diverse hits at every seed (+40.9, +28.3, +150.3, +79.8;
  p = 0.008 each, the floor for 8 paired seeds), with AUC no worse. It adds no coverage
  (#Circles, hull). So the claim channel is what costs hits here. How much of the gain is the
  model's regions is not yet separated: the random control keeps its claims and draws its rule
  at random (about 80% of its campaigns use a rule other than `adko`). A control with random
  regions, rule `adko` and claims off would separate it.
- **The planner barely uses its levers.** 98.9% of the designs chose rule `adko` (101
  `novelty`, 11 `exploit`, 1 `explore`), and every budget was 30 or 40. The model contributes
  the region and the test, not the rule, so the rule headroom measured on 2026-10-19 goes unused.
- **Its tests are too easy.** 96–99.5% pass, and Brier skill against its own base rate is
  negative in every arm (−1.5 to −7.1).
- **Cost and speed.** $146.45 for 32 runs (a one-seed pilot included). Claims and verdicts
  raise bits per token from ~530 (`adko`) to 1,150–1,420; `noclaims` sends 630. 31 runs, 8 at a
  time on one node, took 1 h 42 min at ~100 calls a minute: the per-user rate limit, not CPU,
  set the pace (one run alone takes ~13 min).

## 2026-10-23 — MAG planner, first calls

- `claude-haiku-5-5@amazon`: ~$0.0003 and ~3 s per campaign call; designs parse.
- `claude-sonnet-5-5-medium@amazon`: ~$0.014 and ~4 s per call on the PMO prompt (~4.5k input
  tokens, ~370 output); 12/12 usable. Its pilot campaigns passed 8/8 of their own tests —
  watch whether the tests are too easy.
- `claude-opus-5-5-high@amazon`: `finish_reason=content_filter` with empty text on 11/11 PMO
  campaign calls, billed ~$0.02 each. The 20% failure guard stopped the run. Avoid this route.
- The first real-server run exposed a bug the mock planner could not: a new event loop per
  round broke the HTTP client after round one ("Event loop is closed"). Fixed; regression test
  added. This would have affected vLLM runs equally.

## 2026-10-19 — rule headroom on PMO

`zaleplon_mpo`, 10 agents, 200 rounds (2,000 evaluations), random 4-regular graph, seeds 0–1,
no planner, rule fixed per arm (`--rule`):

| Rule | diverse hits | #Circles | hull | AUC top-10 |
|---|---|---|---|---|
| `adko` | 36.0 | 141.0 | 108.5 | 0.422 |
| `indep` | 30.0 | 158.5 | 128.4 | 0.409 |
| `exploit` | 31.5 | 154.0 | 119.5 | 0.418 |
| `explore` | 5.5 | 202.0 | 173.8 | 0.355 |
| `novelty` | 7.5 | 198.0 | 163.5 | 0.382 |
| `adko` + novelty bonus (w=1) | 35.0 | 137.0 | 105.8 | 0.412 |

Exploration buys coverage at about five times fewer hits; no fixed rule does both. The order
within each group flipped between the two seeds. At weight 1 the bonus does not add coverage.

Also: one full-budget run (10,000 evaluations, mock planner, seed 0) took 146 s: 6,735
scaffolds, 87 diverse hits at 0.4, 283 #Circles, hull 211.4, AUC-top-10 0.459, 490 campaigns
(269 passed), Brier 0.271. Making peer footprints outlive the token FIFO left the PMO `novelty`
runs bit-identical (with 250k candidates the max-min pick is dominated by the pool's geometry).
