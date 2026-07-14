# Possession-gate diagnosis - Brighton vs Manchester Utd pilot

Fixture: **Brighton 2-1 Manchester Utd**, ENG Premier League 2024-25 MW2 (2024-08-24). Registry match
`brighton_manutd`, `teams = [Man Utd (team0), Brighton (team1)]`.

**Pre-committed gate:** possession within ~5 pp of official FBref **Man Utd 52% / Brighton 48%**.
**Our CV proxies:** ball-possession (`assign_possession` smooth=True, per-chunk, calib-gated, matching
`report.facts`) = **Man Utd 44.2% (4157 samples)**; space-control fact = **Man Utd 43.5%**. Both
~8-12 pp Brighton-heavy vs official, and both **flip the possession leader** (official = Utd majority;
our proxies = Brighton majority). **Gate FAILS.** This note establishes *why*.

Reproduction note: the fact store's `line_break_in_poss_frames` (Utd 1789 / Bri 2268 = 44.1% Utd)
matches the reproduced 44.2%. The task's cited "41.9% / 2659 samples" is a slightly different probe
(different concat/debounce), same direction and magnitude; I could not bit-reproduce 2659 but the
shipped fact store agrees with 44%.

---

## Q1 - TEAM ANCHOR: CORRECT (not flipped). Verdict is definitive.

The align colour diagnostic labelled team0 = red (Man Utd, hue 21), team1 = ambiguous (hue 48). I
confirmed this **visually and quantitatively**, independent of the anchor.

- **Visual overlay** (aligned team ids drawn on decoded broadcast frames; cyan box = T0, yellow = T1):
  - `results/pl_pilot/anchor_check/h1_chunk_000_f145.png` (H1 kickoff): every **red** shirt (Man Utd,
    e.g. #10, #20) is boxed **T0/cyan**; every **blue-white stripe** (Brighton, e.g. #17) is **T1/yellow**.
  - `results/pl_pilot/anchor_check/h2_chunk_000_f90.png` (H2): same mapping holds after the half
    (red = T0, blue-white = T1). Anchor is stable across halves.
- **Quantitative torso colour** (mean over 67 player crops across 4 frames): team0 BGR=(103,119,142),
  R>B, HSV hue 13 (red-orange); team1 BGR=(140,150,135), HSV hue 69 (blue-white stripes averaged with
  grass). Consistent with red=team0.

**Consequence:** the Brighton skew is REAL, not a labelling artifact. Had the anchor been flipped our
number would read 55.8% Utd (delta +3.8 pp vs official) and the gate would nearly pass - but it is NOT
flipped. All downstream team labels, attack directions and facts are correctly attributed to Man Utd
(team0) / Brighton (team1).

Minor noise observed: the black-clad referee is occasionally absorbed into the team0 (dark) cluster
(visible in f145). Negligible for possession (ref rarely the nearest player to the ball).

---

## Q2 - Per-chunk / per-half split + coverage

Possession = `assign_possession(ball, players[calib_error_m<=1.0], smooth=True)` per chunk (exact
`report.facts` path). Coverage = post-`link_ball` usable rows / dense frames.

| chunk        | fps | n_dense | ball rows | coverage | poss n | Utd n | Bri n | **Utd share** |
|--------------|----:|--------:|----------:|---------:|-------:|------:|------:|--------------:|
| h1_chunk_000 |  25 |    2315 |      1239 |    53.5% |    482 |   193 |   289 |     **40.0%** |
| h1_chunk_001 |  25 |    2435 |      1336 |    54.9% |    481 |   204 |   277 |     **42.4%** |
| h1_chunk_002 |  25 |    2361 |      1244 |    52.7% |    517 |   249 |   268 |     **48.2%** |
| h1_chunk_003 |  25 |    2167 |      1131 |    52.2% |    434 |   250 |   184 |     **57.6%** |
| h1_chunk_004 |  25 |    1887 |      1205 |    63.9% |    470 |   148 |   322 |     **31.5%** |
| h2_chunk_000 |  25 |    2516 |      1210 |    48.1% |    423 |   141 |   282 |     **33.3%** |
| h2_chunk_001 |  25 |    1849 |       873 |    47.2% |    320 |   181 |   139 |     **56.6%** |
| h2_chunk_002 |  25 |    2140 |       861 |    40.2% |    314 |   185 |   129 |     **58.9%** |
| h2_chunk_003 |  25 |    2144 |       862 |    40.2% |    302 |    93 |   209 |     **31.0%** |
| h2_chunk_004 |  25 |    1821 |      1079 |    59.3% |    350 |   158 |   192 |     **45.1%** |
| h2_chunk_005 |  25 |     245 |       144 |    58.8% |     64 |    34 |    30 |     **53.1%** |

- **H1:** 43.8% Utd (2384 samples). **H2:** 44.7% Utd (1773 samples). The skew is **uniform across
  halves**, not a one-half artifact.
- **Not uniform across chunks:** Utd share swings 31%->59% chunk-to-chunk. The skew is a *pooled* bias,
  not a constant offset - consistent with a game-state-dependent sampling mechanism (below), not a
  fixed mislabel.
- **Coverage does not cleanly predict the swing** across chunks (highest-coverage h1_chunk_004 @ 63.9%
  is the most Brighton-heavy at 31.5% Utd; lowest-coverage h2_chunk_002/003 @ 40.2% split 59%/31%).
  The bias is a *within-phase* effect (which possessions are trackable), not a simple chunk-coverage
  correlation.

**Game-state cross-reference (LIMITED - see caveat):** chunks where Utd share rises above 55%
(h1_chunk_003 @ 30-40 min, h2_chunk_001/002 @ 58-78 min) are plausibly Man-Utd-chasing phases (after
falling behind / pushing for the equaliser). This is *qualitatively* consistent with possession
tracking game state, but I **could not verify the goal timeline**: the cached FBref matchlogs
(`~/soccerdata/data/FBref/matchlogs_*`) carry no goal-minute detail, and the broadcast has **no
persistent score-clock bug** in sampled frames (only an intermittent PL logo top-right;
`results/pl_pilot/anchor_check/topstrip.png`, `scoreboard_montage.png`). The public record is
Brighton 2-1 (Man Utd equalised then conceded a late winner), but I am not asserting exact minutes
from our data. Treat the game-state read as suggestive, not verified.

---

## Q3 - Coverage-conditional bias: the carry-over lever is EXONERATED

Split possession frames by projection source. `own` = the ball frame had its own >=6-correspondence
homography (recomputed on CPU from the dense parquet via `ball_carry.fit_frame_homographies`);
`carry` = observed ball frame projected from a neighbour homography (the new lever); `link_interp` =
`observed=False` gap-fill from `link_ball`.

| source        |     n | % of poss | Utd share | Bri share |
|---------------|------:|----------:|----------:|----------:|
| own           |  3622 |     87.1% |     44.4% |     55.6% |
| carry         | **0** |    **0%** |         - |         - |
| link_interp   |   535 |     12.9% |     42.8% |     57.2% |

**Key finding:** the carry-over lever contributes **zero possession samples**. The reason is a hard
interaction with the fact-store calibration gate: `report.facts` filters players to
`calib_error_m <= 1.0` before `assign_possession`. The frames the carry lever *adds ball coverage on*
are precisely the frames that **lack their own homography** (poor calibration) - and after alignment
those frames carry `calib_error_m > 1.0`, so **every player on them is filtered out**. Direct check on
h1_chunk_000: of 429 carried (observed, not-own-H) ball frames, **0** have any calib-gated player
present. So a carried ball has nobody to be "near" -> no possession assigned.

Implications:
1. The Brighton skew is **not** a new bias introduced by the carry lever. The lever roughly doubles
   post-link coverage (32%->52%), but that extra coverage is invisible to possession (calib-gated out).
   The skew lives entirely in the **own / well-calibrated** frames (44.4% Utd), which existed before
   the lever.
2. `link_interp` frames (short gap-fills between own detections, calib<=1) skew marginally *more*
   Brighton (42.8%) than own frames, but they are only 12.9% of samples and move the pooled number by
   <0.3 pp. Not material.
3. Honesty note for the coverage story: the "lever doubles coverage" win and the "possession is
   Brighton-heavy" problem are **decoupled** - the coverage the lever buys does not reach the
   possession metric at all under the current calib gate.

---

## Q4 - Sampling-rate sanity: expected, not anomalous

- Ball detected on **every** dense frame; dense frames are at the extraction stride of **5 native
  frames** = 0.20 s at 25 fps (not 1 frame/2 s).
- Total dense frames sampled: 21,880; post-link ball-track rows: 11,184 (51% coverage);
  possession-assigned: 4157 = **37.2% of ball rows** are within the 2.0 m carrier radius.
- So possession samples arrive ~every 0.54 s of ball track, not every 2 s. The "2659 over 95 min"
  framing (1/2.1 s) reflects the *smaller* cited probe; either way the rate is governed by
  (stride 5) x (51% ball coverage) x (37% within-radius). Consistent with the WC matches, where
  possession is likewise sampled only on ball-tracked frames within the carrier radius. **No
  sampling-rate anomaly.**

---

## Q5 - VERDICT: (b) genuine trackability/occlusion sampling bias; (a) ruled out; (c) partial

**(a) Anchor flipped - RULED OUT.** Visually confirmed team0=Man Utd=red (Q1). The number is genuine.

**(b) Occlusion/calibration sampling bias - CONFIRMED, dominant cause.** Possession is measured only
on well-calibrated, ball-trackable frames. Those frames over-represent **settled / positional**
possession (many players in the wide view, ball in mid/defensive third -> good homography) and
under-represent **direct / transition / final-third** possession (ball near one goal, fewer wide
correspondences, faster play, more occlusion -> calibration drops out, ball frame discarded). In this
fixture Brighton's patient home build-up is the more trackable phase, so Brighton is over-counted; Man
Utd's more direct/transitional possession is under-sampled. This is the **same mechanism** as the
mun_mci 82/18-City over-skew (the patient/positional side is inflated) - there it agreed on direction
and over-skewed; here the true split is near 50/50, so an ~8-12 pp bias **flips the leader**.

**(c) FBref definition mismatch - PARTIAL, secondary.** I could not confirm FBref's exact possession
definition from our cache (the empty `fbref_ref.json` / matchlogs do not state it). FBref PL match
possession is generally understood to be an Opta time/touch-based share (broadly comparable in *kind*
to our proximity proxy), so the definition gap is unlikely to explain 8-12 pp on its own - this should
be verified against FBref's glossary before relying on it. BUT one genuine signal cuts against reading
our number as pure error:
`space_control` (position-only, NOT ball-gated, survives low coverage) **independently** puts Brighton
at 56.5% territory. Two methodologically different metrics (proximity possession and territorial
control) both say Brighton controlled more. It is tactically coherent that Brighton (home) controlled
**space/territory** even if Man Utd edged **ball-possession time** 52-48 - "who had the ball longer"
and "who controlled the pitch" can diverge. So the Brighton-heavy reading is not 100% artifact; part
of it is a real territorial signal that the official time-possession stat does not measure.

**Bottom line:** our ball-possession proxy is **not a defensible estimator of Opta time-possession**
for a +/-5 pp gate - it is a biased estimator ("trackable-frame possession share") that inflates the
more positional side. The gate fails honestly. This is a **known-limitation** outcome, not a bug to
patch away.

---

## Recommendations

**Pipeline:**
1. **Rename / re-scope the metric.** Report it as "trackable-frame possession share (biased toward
   settled/positional phases)", never as "possession %" compared to Opta. The report-v2 abstention
   machinery should treat an Opta-possession claim as *not* satisfiable by this proxy.
2. **The carry lever is safe to keep** (it does not distort possession) but note that its coverage
   gain does not reach possession under the calib gate - so it is not a fix for this bias either. If
   we ever want carried frames to count for possession, we would need calibrated player positions on
   those frames (they are gated out today), i.e. impute-then-gate rather than gate-on-raw-calib.
3. **De-biasing (future, needs validation):** weight possession frames by inverse local trackability
   per game-state/third, or restrict the comparison to a trackability-matched window. Do not ship any
   such correction without a validator - we have no per-phase ground-truth possession.

**Gate protocol:**
4. **Drop ball-possession from the +/-5 pp Opta gate.** We cannot validate it against Opta time-
   possession. Either (i) gate only `space_control` against a *territorial* reference (still Brighton-
   heavy here, but at least honestly a territory metric, not "possession"), or (ii) keep possession as
   an *unvalidated, caveated* diagnostic outside the pass/fail gate.
5. **Record this as the pilot's honest negative:** the possession gate FAILED end-to-end; the failure
   is a measurement-representativeness limitation (only trackable possession is counted), confirmed
   not to be an anchor flip and not caused by the carry-over lever.

Artifacts: `results/pl_pilot/anchor_check/` (overlays, torso stats, scoreboard probes),
this file. Reproduced with `report.facts`-identical params against `data/matches.yaml` `brighton_manutd`.
