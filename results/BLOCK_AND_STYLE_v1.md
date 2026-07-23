# BLOCK_AND_STYLE_v1 -- defensive block geometry + season style-factor profile

**Phase-0/1 build C** of the style-analysis v2 architecture (`results/STYLE_RESEARCH_METHODS.md`, Axis 2
rows #2 and #4). Two deliverables:

1. **Defensive block-height + compactness** on trusted-geometry frames (`fingerprint/block_height.py`).
2. **Event-count playing-style factor profile** from public FBref season data (`fingerprint/style_factors.py`).

**Gate status.** Block-height Gate 1 (hand-annotated line-height validation) is **PENDING** -- every block
number below is descriptive geometry on our de-biased line scale, not human-certified. Nothing here ships
into a report as a validated claim until Gate 1 passes.

Modules: `fingerprint/block_height.py`, `fingerprint/style_factors.py`. Test: `tests/test_block_and_style.py`
(3 passed). Both ruff-clean at 100 cols. CPU only, no GPU, no writes to `data/matches.yaml`.

> **2026-07-23 CORRECTION:** `southampton_manutd` + `tottenham_manutd` team mappings corrected (see
> `results/PAIR_ANALYSIS_v1.md`). The `southampton_manutd` rows below had Man Utd and Southampton
> **swapped**: the 37.4 m high block (34.5% high spells) is **United's** (a 0-3 away win, United's
> highest line in the corpus), not Southampton's (who sat at 24.0 m). The prior
> **Southampton-pressed-high claim is retracted** and the ten-Hag regime aggregate is recomputed. Only
> `southampton_manutd` and the ten-Hag row change; the other 10 legs were already correctly labelled.

---

## Part 1 -- Defensive block geometry

### Method

Per team, per match, on the **linked-ball chunks** only (possession needs the ball):

- **Out-of-possession filter** = the **Viterbi possession smoother** (`generator.ball.assign_possession`,
  `smooth=True`). Each frame, the team NOT in possession is the defending team; its own block is measured.
- **Block line height** = the **de-biased rearmost-outfielder line** (`generator.impute.line_estimates`,
  `line_debiased`) -- the exact convention as the shipped/validated defensive line (the 16.4->7.2 m
  held-out line work), so the block line is on the same metre scale (own goal = 0 m, attacked goal = 105 m).
- **Block vertical spread (compactness)** = std of the defending outfield along its attacking axis
  (goal-to-goal block depth); smaller = more compact.
- **Ball-to-block distance** = ball advancement in the defender's own-goal=0 frame minus the block line
  (positive = ball is ahead of / in front of the block).
- **Geometry gate** = `calib_error_m <= 1.0` (identical to `report/facts.CALIB_MAX_M`) and >=3 visible
  back-line defenders.

**Declared thresholds** on the de-biased line (the de-bias already removes the ~11 m broadcast inflation
that `fingerprint.phase_metrics` carries on the raw visible line, so these are un-inflated true-scale cuts):

| class | de-biased median line over the spell |
|-------|--------------------------------------|
| low block  | `< 33 m` from own goal |
| mid block  | `33 - 50 m` |
| high block / press | `>= 50 m` |

Each **out-of-possession spell** (contiguous frames a team defends within a chunk, >=5 usable frames) is
classified by its median line. Class shares below are weighted by usable frames.

### Per-match block table (11 processed legs shown; `tottenham_manutd` was processed after this table and is not folded in here -- kept isolated to the mapping correction)

`line` = mean de-biased block line (m from own goal). `vspr` = mean vertical spread (m). `b2b` = mean
ball-to-block (m). `cov` = out-of-possession coverage. `bias` = broadcast-bias delta (m, see below).
`low/mid/high` = spell-class shares. **Man Utd rows are the focus team.**

| match | team | line | vspr | b2b | cov | bias | low/mid/high | modal |
|-------|------|-----:|-----:|----:|----:|-----:|--------------|-------|
| brighton_manutd (TH) | **Man Utd** | 33.2 | 6.2 | 14.7 | 0.96 | +0.3 | .518/.262/.221 | low |
| brighton_manutd | Brighton | 23.8 | 5.8 | 25.1 | 0.95 | +9.4 | .728/.131/.141 | low |
| manutd_liverpool (TH) | **Man Utd** | 29.5 | 5.9 | 13.0 | 0.97 | -3.6 | .585/.266/.149 | low |
| manutd_liverpool | Liverpool | 25.8 | 6.3 | 20.0 | 0.95 | +9.3 | .659/.258/.083 | low |
| manutd_fulham (TH) | **Man Utd** | 25.1 | 6.1 | 21.3 | 0.97 | +9.3 | .726/.164/.109 | low |
| manutd_fulham | Fulham | 21.2 | 5.7 | 31.2 | 0.96 | +6.4 | .798/.172/.030 | low |
| palace_manutd (TH) | **Man Utd** | 26.7 | 5.8 | 15.2 | 0.90 | +5.4 | .624/.307/.069 | low |
| palace_manutd | Crystal Palace | 28.4 | 5.7 | 18.9 | 0.96 | -6.0 | .664/.280/.055 | low |
| manutd_tottenham (TH) | **Man Utd** | 26.0 | 6.2 | 25.7 | 0.97 | +10.3 | .677/.216/.108 | low |
| manutd_tottenham | Tottenham | 27.1 | 6.1 | 17.1 | 0.93 | +5.0 | .625/.216/.158 | low |
| southampton_manutd (TH) | **Man Utd** | 37.4 | 5.8 | 6.4 | 0.82 | -1.7 | .499/.155/.345 | low |
| southampton_manutd | Southampton | 24.0 | 5.5 | 26.0 | 0.78 | -4.7 | .719/.226/.055 | low |
| liverpool_manutd (AM) | **Man Utd** | 20.8 | 5.7 | 30.1 | 0.96 | +0.5 | .768/.147/.086 | low |
| liverpool_manutd | Liverpool | 27.6 | 5.8 | 21.5 | 0.87 | +13.2 | .642/.121/.237 | low |
| manutd_brighton (AM) | **Man Utd** | 29.6 | 5.9 | 15.4 | 0.95 | +2.4 | .572/.314/.114 | low |
| manutd_brighton | Brighton | 22.4 | 6.1 | 30.1 | 0.96 | +2.9 | .734/.207/.059 | low |
| fulham_manutd (AM) | **Man Utd** | 27.0 | 6.6 | 21.6 | 0.93 | +12.4 | .647/.240/.113 | low |
| fulham_manutd | Fulham | 21.7 | 6.1 | 34.6 | 0.95 | +8.4 | .712/.173/.115 | low |
| manutd_palace (AM) | **Man Utd** | 30.1 | 6.6 | 17.7 | 0.94 | +4.1 | .574/.198/.228 | low |
| manutd_palace | Crystal Palace | 24.7 | 5.6 | 24.6 | 0.96 | +12.8 | .658/.267/.075 | low |
| manutd_southampton (AM) | **Man Utd** | 25.2 | 6.9 | 29.2 | 0.96 | +7.7 | .709/.152/.138 | low |
| manutd_southampton | Southampton | 29.7 | 6.0 | 13.1 | 0.91 | +0.1 | .498/.279/.223 | low |

(TH = Ten Hag regime, AM = Amorim regime, from `data/matches.yaml` `manager` field.)

### Coverage and the broadcast-bias disclosure (MANDATORY honesty numbers)

- **Coverage** (median **0.95**, min 0.78) is the fraction of out-of-possession frames **with a ball /
  possession fix** that also have usable block geometry. This denominator is itself bounded by ball-track
  coverage (32-52% of match time), so a high number means geometry is **dense within observed defensive
  moments**, NOT that all defensive time is seen. The scarcity is upstream (ball + homography), and the
  earlier ~26.4%-usable-frame figure from the research is a pipeline-wide denominator, not this metric's.
  The two low-coverage matches (`southampton_manutd` 0.78/0.82) are the honest floor.
- **Broadcast bias** = mean ball advancement on usable-geometry frames minus on missing-geometry frames.
  Across all 22 team-matches: **median +5.2 m, mean +4.7 m, range [-6.0, +13.2] m**. The sign is
  predominantly **positive** -- confirming the disclosure that *the broadcast frames the block preferentially
  when the ball is further forward*. The sampled block is therefore measured when the ball is ~5 m more
  advanced than a random defensive moment, biasing the observed block to read **deeper/lower** than its true
  average. Caveat: with coverage ~0.95 the "missing" sample is small (37-367 frames), so per-match bias is
  noisy; the corpus-level sign is the robust finding.

### The "everyone reads low" caveat

**Every team's modal block is `low`** and mean lines cluster 20-37 m. Two things are true at once: (a) EPL
sides genuinely spend most *observed* out-of-possession time in low-to-mid blocks, and (b) the de-biased
line has a **~7 m held-out absolute-scale uncertainty** and the broadcast bias pushes the sampled block
deeper -- so the *absolute* class is not trustworthy until Gate 1. The **usable signal is relative**: the
spell-class shares and the cross-team/cross-match ordering differentiate teams even where the absolute label
does not. Read the high/mid **shares**, not the modal label.

### Man Utd block by manager regime (directions only; n=6 Ten Hag, n=5 Amorim)

| regime | n | mean line (m) | vspread (m) | ball-to-block (m) | low share | high share | coverage |
|--------|--:|--------------:|------------:|------------------:|----------:|-----------:|---------:|
| Ten Hag | 6 | 29.7 | 6.00 | 16.1 | 0.605 | 0.167 | 0.930 |
| Amorim  | 5 | 26.5 | 6.34 | 22.8 | 0.654 | 0.136 | 0.949 |

Directional read (NOT significant at this n): after the mapping correction, under **Ten Hag** United's own
block reads a shade **higher on the mean line** (29.7 vs 26.5 m) with a **higher high-block spell share**
(0.167 vs 0.136), while Amorim shows a **wider vertical spread** (6.34 vs 6.00 m). This **reverses** the
pre-correction read (which had Amorim marginally deeper / more bimodal) and is driven largely by the one
corrected leg -- southampton_manutd's 37.4 m high line moving into the ten-Hag column -- so treat it as a
fragile direction, not a manager law. Either way the block metric does **not** cleanly separate the two
regimes, so it is not (yet) a strong manager fingerprint on its own. (Amorim held at n=5 as originally
reported; `tottenham_manutd`, now processed, is not folded in so the correction stays isolated.)

### Man Utd vs opponents (block read)

- **Opponents sit deep against United -- no high-press exception.** In all 11 fixtures the non-United side's
  block line is 20-30 m (low/mid): every opponent **drops into a low block vs United** rather than press. The
  apparent "Southampton pressed high (37.4 m, 34.5% high spells) and lost 0-3" exception was a **team-mapping
  flip** -- that 37.4 m high line is **United's own** (their highest line in the corpus, in the 0-3 away
  win); Southampton at home actually sat at **24.0 m** (5.5% high spells), squarely low.
- **United's own block** is low-block-dominant home and away (modal low in every leg) but its mean line
  spans 24-37 m -- the top end is now the **corrected southampton_manutd away leg (37.4 m)**, United's
  highest in the corpus. The ball-to-block gap is *larger* in most away/reverse fixtures (e.g. 30.1 m at
  Anfield) -- United defending with the ball further in front, conceding territory -- but small (6.4 m) in
  that Southampton away win where United defended higher.

---

## Part 2 -- Season-scale playing-style factor profile (Fernandez-Navarro 2016)

### Public data fetched

Via soccerdata's rate-limited session into `data/fbref_style_cache/` (gitignored, **3.2 MB** total, well
under the 100 MB budget). Fully-populated FBref 2024-25 PL squad tables from the **/stats/ page**:
`standard`, `shooting`, `misc`, in both `_for` (own play) and `_against` (conceded) form, all 20 PL teams.

**Data limitation (honest, not hidden).** The richer FBref squad **passing / possession / defense** pages --
which carry long-ball share (directness) and **tackle-by-third (the clean season-scale pressing-height
signal)** -- were **blocked by FBref anti-scraping**: the connection was refused (`WinError 10061`, the same
403/block `tools/oracle.py` documents) after the /stats/ page fetched, and the pages that did return had
**empty stat cells**. We did **not** hammer them. Consequence: this season profile spans **possession,
attacking directness/volume, width, and defensive engagement**, but **not pressing height at season scale**.
Pressing/block height comes instead from Part 1 (our own tracking). Interception/tackle/foul rates here are
labelled **defensive engagement**, not height.

### Feature set (11 features, per-90 or share, z-scored across 20 teams)

`poss_pct`, `shots_p90`, `sot_pct`, `goals_per_shot`, `offsides_p90` (high line / running in behind),
`crosses_p90` (width), `interceptions_p90`, `tackles_won_p90`, `fouls_p90` (defensive engagement),
`shots_against_p90` + `crosses_against_p90` (what is conceded).

### PCA (all-20 fit; explained variance PC1 39.3%, PC2 19.2%, PC3 15.2%; cumulative 73.7%)

Loadings (feature -> component), naming each axis from its own loadings -- **not** forcing the FN labels:

| feature | PC1 | PC2 | PC3 |
|---------|----:|----:|----:|
| poss_pct | **+0.42** | +0.10 | -0.14 |
| shots_p90 | **+0.40** | +0.25 | +0.13 |
| sot_pct | +0.10 | -0.30 | **+0.56** |
| goals_per_shot | +0.23 | -0.36 | **+0.45** |
| offsides_p90 | -0.14 | +0.07 | **+0.43** |
| crosses_p90 | +0.25 | +0.29 | +0.25 |
| interceptions_p90 | -0.33 | **+0.38** | +0.19 |
| tackles_won_p90 | -0.23 | **+0.35** | +0.37 |
| fouls_p90 | -0.19 | **+0.46** | +0.00 |
| shots_against_p90 | **-0.38** | -0.27 | -0.11 |
| crosses_against_p90 | **-0.42** | -0.24 | +0.12 |

- **PC1 = territorial control** (+possession, +shots taken, -shots/crosses conceded, -interceptions).
  High = dominant/possession side; low = dominated, defends a lot. Maps to the FN **direct<->possession**
  axis at the *team-control* end.
- **PC2 = defensive engagement / physicality** (+fouls, +interceptions, +tackles won; -clinical finishing).
  A disruption/aggression axis. **This is the closest available proxy to pressing but is NOT height** --
  season data cannot say *where* the tackles happen.
- **PC3 = shot quality + high-line push** (+SoT%, +goals/shot, +offsides). Clinical, vertical, plays a
  higher line.

### Corpus style coordinates (Man Utd + the 6 opponents)

| team | PC1 (control) | PC2 (engagement) | PC3 (shot-quality/high-line) |
|------|-------------:|------------------:|------------------------------:|
| **Man Utd** | -0.54 | **+2.46** | -0.05 |
| Brighton | +1.19 | +0.89 | +0.26 |
| Liverpool | **+2.79** | +1.08 | +0.38 |
| Fulham | +1.34 | +1.39 | -0.34 |
| Crystal Palace | -1.52 | +0.61 | +0.86 |
| Tottenham | +0.33 | +0.13 | +1.31 |
| Southampton | **-2.65** | -0.44 | **-2.75** |

### Man Utd read (season style)

United 2024-25 are **middling on control** (PC1 -0.54: 53.5% possession, 13.8 shots/90 but concede 10.8
shots and 15.7 crosses/90) yet **the corpus extreme on defensive engagement** (PC2 +2.46, far clear of the
field): league-high tackles-won (12.97/90) and interceptions (9.42/90) among these seven. In plain terms the
public counts describe a team that **does a lot of defensive work without dominating territory** -- it makes
tackles and interceptions at a high rate but concedes shots and crosses like a mid-table side, and finishes
below par (0.07 goals/shot, tied lowest with Southampton). Liverpool is the control outlier (+2.79);
Southampton is dominated on both control and shot-quality (the relegated profile).

**Consistency with Part 1:** United's high defensive-engagement season profile sits alongside a low-block
tracking profile (Part 1) and opponents dropping deep against them -- a coherent picture of a side defending
often and deep, not one pressing high or dominating. The one caveat is that PC2 is engagement, not height,
so this is suggestive, not a pressing-height claim.

### Cross-check vs our per-match validated counts (where they exist)

From the fact stores (`outputs/facts/*.json`), which carry Sofascore-validated pass counts on the
**linked-ball chunks only** (not full-match totals -- absolute counts are low and NOT comparable to FBref
season totals; `mean_pass_m` is the usable directness proxy):

| team | our mean_pass_m (matches) | FBref PC1 (control) |
|------|---------------------------|--------------------:|
| Man Utd | 8.7 / 8.5 / 7.5 / 8.5 / 9.1 | -0.54 |
| Liverpool | 8.4 | +2.79 |
| Tottenham | 9.3 | +0.33 |
| Southampton | 8.4 | -2.65 |
| Crystal Palace | 7.2 | -1.52 |

Directional only (n=1 match for most opponents): the lower-control sides in the FBref PCA (Palace 7.2 m) also
show shorter validated passes in our tracking, but the sample is far too thin to call agreement -- reported
as a sanity link, not a validation.

---

## Caveats (inline summary)

1. **Gate 1 pending** -- no block-height number is human-validated; absolute low/mid/high labels are
   provisional (the whole corpus reads "low", partly a ~7 m de-bias scale uncertainty + the forward-pass
   broadcast bias). Use relative shares, not absolute classes.
2. **Broadcast bias is real and mostly deepening** (+5 m median) -- the block is sampled when the ball is
   forward, so observed blocks read lower than truth.
3. **Coverage is high only within ball-observed defensive time**; true defensive-time coverage is bounded by
   ball-track coverage (32-52%).
4. **Pressing HEIGHT is absent from the season profile** (FBref passing/possession/defense pages blocked);
   PC2 is engagement, not height. Season-scale pressing height would need the FBref defense (tackle-by-third)
   table -- a future fetch when FBref is not blocking, or SkillCorner.
5. **Manager split and opponent reads are directions, not significance** (n=5-6 per regime, n=1 per
   opponent fixture).
6. **`tottenham_manutd` is now processed** but is deliberately not folded into this table -- the
   2026-07-23 correction is kept isolated to the two mislabelled legs (11 of 12 fixtures shown here).
