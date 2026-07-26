# StatsBomb open data — audit for the geometric-tactics project (2026-07-26)

Everything below was measured by fetching the real repository and the real JSON, on this machine,
today. No blog summaries, no recall. Repo tree read via the GitHub trees API (`recursive=1`,
`truncated: false`); competition list read via `statsbombpy` (already installed at
`AppData/Roaming/Python/Python314/site-packages/statsbombpy`, nothing new was pip-installed); six
matches downloaded to scratch and deleted after measurement.

Question this audit serves: `docs/GEOMETRIC_TACTICS_PROPOSAL.md` — team shape and formation under
ball-biased partial observation, and specifically whether StatsBomb 360 can act as an **independent
validation set** for that.

**Short answer, stated first: no ground truth exists behind the censoring in StatsBomb 360. It is a
second censored sample, not a validation set. It is, however, the best available measurement of the
censoring operator itself, and that is worth roughly one day of work.**

---

## 1. Inventory (measured)

Repository `statsbomb/open-data`, branch `master`, 8 987 files.

| directory | files | total size |
|---|---|---|
| `data/events` | **4 235** | 12.82 GB |
| `data/lineups` | 4 235 | 0.08 GB |
| `data/three-sixty` | **426** | 3.21 GB (mean 7.54 MB per match) |
| `data/matches` | 80 season files | 0.01 GB |
| `doc` | 6 PDFs | — |

Competition/season files list **3 961 matches** across **80 competition-seasons**. There are 274
event files for matches that appear in no competition list (orphans, reachable only by match id).
Every listed match has an events file; every 360 file belongs to a listed match.

### 1.1 The twelve competition-seasons with 360 (all of it)

| competition | season | gender | matches | with 360 |
|---|---|---|---|---|
| FIFA World Cup | 2022 | M | 64 | **64** |
| Women's World Cup | 2023 | F | 64 | **64** |
| UEFA Euro | 2020 | M | 51 | **51** |
| UEFA Euro | 2024 | M | 51 | **51** |
| La Liga | 2020/2021 | M | 35 | **35** |
| 1. Bundesliga | 2023/2024 | M | 34 | **34** |
| Ligue 1 | 2022/2023 | M | 32 | **32** |
| UEFA Women's Euro | 2022 | F | 31 | **31** |
| UEFA Women's Euro | 2025 | F | 31 | **31** |
| Ligue 1 | 2021/2022 | M | 26 | **26** |
| Major League Soccer | 2023 | M | 6 | **6** |
| African Cup of Nations | 2023 | M | 52 | **1** |
| **total** | | | | **426** |

Note for the ManU scope: there is **no Premier League 360**. The free PL seasons are 2015/16 (380
matches) and 2003/04 (38), events only. The closest club-football 360 to an EPL broadcast is
Bundesliga 2023/24 (34), Ligue 1 2021/22 + 2022/23 (58) and La Liga 2020/21 (35) — 127 club matches
in total, the rest is international tournament football.

Biggest event-only holdings (no 360): La Liga 2015/16 380, Premier League 2015/16 380, Serie A
2015/16 380, Ligue 1 2015/16 377, Liga F 2023/24 240, NWSL 2023 137, FA WSL 2023/24 132, Frauen
Bundesliga 2023/24 132, Indian Super League 2021/22 115, all 18 Champions League finals 1 each.

### 1.2 Data types

Three per match: `events` (full event feed, player ids and names, x/y locations, freeze frames for
shots), `lineups` (players, positions, cards), `three-sixty` (only for the 426). There is **no
tracking data of any kind** in the open repo — no continuous player trajectories, no ball track.

---

## 2. What a 360 freeze frame actually contains (measured)

Deep measurement on two matches downloaded in full — `3895292` Union Berlin vs Bayer Leverkusen
(Bundesliga 2023/24) and `3930163` Serbia vs England (Euro 2024) — plus four more sampled for
visible-count statistics only.

### 2.1 Schema, verbatim from the data

```
360 record keys        : ['event_uuid', 'freeze_frame', 'visible_area']
freeze-frame entry keys: ['actor', 'keeper', 'location', 'teammate']   (these four, always, 102 402
                          entries checked across the two matches — nothing else ever appears)
sample entry           : {"teammate": true, "actor": false, "keeper": false,
                          "location": [32.919890433375514, 44.179555360971285]}
visible_area           : flat list [X1 Y1 X2 Y2 ... X1 Y1], closed loop, 5-8 distinct vertices
```

**There is no player id and no player name.** A point is only flagged `teammate` (same team as the
event's actor), `keeper`, `actor`. Confirmed by the official spec (`doc/Open Data 360 Frames
v1.0.0.pdf`): *"these freeze frames will not contain player identification, beyond their team
(except for the player performing the current event who will be marked as the actor)"*. Measured
consequence: **6.05-6.73 % of freeze-frame points carry a recoverable identity** (the actor, and only
because the linked event names the player). The other 93 % are anonymous dots.

### 2.2 Coordinate system

- StatsBomb 120 x 80 units. 1 x-unit = 0.875 m, 1 y-unit = 0.85 m on a 105 x 68 pitch.
- Locations are **already in attacking coordinates**: the actor's team attacks towards x = 120.
  Verified from keeper positions — own keeper (teammate=True) median x = 5.00 / 5.25, opponent keeper
  median x = 116.60 / 116.60 in the two matches.
- Player locations may lie **off the pitch**: measured x range -3.63..122.09, y range -8.53..88.07.
  The `visible_area` polygon is clipped to 0..120 x 0..80.
- **The frame is rotated 180 degrees per (acting team, period), and this matters.** Measuring the
  polygon's x-span at y = 10 minus its x-span at y = 70 (a perspective asymmetry that must be fixed
  in the physical world, since the camera never changes touchline):

  | match | team | period 1 | period 2 |
  |---|---|---|---|
  | 3930163 | England | -26.50 | **+20.11** |
  | 3930163 | Serbia | **+26.04** | -20.37 |
  | 3895292 | Bayer Leverkusen | +12.34 | **-18.90** |
  | 3895292 | Union Berlin | -8.28 | **+9.45** |

  Perfectly antisymmetric in both team and period. So the y axis flips with the x axis: it is a 180
  degree rotation of the world, not a mirror. **Anyone reusing `visible_area` as a camera footprint
  must undo it** with `(x, y) -> (120 - x, 80 - y)` for the flipped (team, period) cells. In the
  de-rotated frame the visible region is narrow at the near touchline and wide at the far one — the
  same trapezoid SkillCorner reports (11.0 m near half-width, 24.1 m far, `results/B4_TRANSFER_M3.md`
  §0).

### 2.3 How much of the team you get

Six matches, spanning four competitions:

| match | competition | 360 frames | mean visible / 22 | p10 | max | frames >= 20 | frames = 22 | team with all 11 | both keepers |
|---|---|---|---|---|---|---|---|---|---|
| 3895292 | Bundesliga 23/24 | 3 195 | **14.86 (0.675)** | 10 | 21 | 3.54 % | 0 | — | — |
| 3930163 | Euro 2024 | 3 324 | **16.53 (0.751)** | 13 | 21 | 11.55 % | 0 | — | — |
| 3857276 | World Cup 2022 | 2 873 | **13.19 (0.600)** | 9 | 21 | 1.71 % | 0.00 % | 1.57 % | 0.00 % |
| 3998853 | Women's Euro 2025 | 2 531 | **12.69 (0.577)** | 8 | 21 | 0.08 % | 0.00 % | 1.19 % | 0.00 % |
| 3773386 | La Liga 20/21 | 3 670 | **16.27 (0.740)** | 12 | 21 | 9.89 % | 0.00 % | 3.84 % | 0.00 % |
| 3838017 | Ligue 1 22/23 | 3 266 | **17.14 (0.779)** | 13 | 22 | 25.72 % | **0.03 %** (1 frame) | 5.86 % | 0.00 % |

Per-frame breakdown for 3895292 / 3930163: teammates 7.48 +/- 1.78 / 8.16 +/- 1.41 of 11; opponents
7.37 +/- 2.11 / 8.36 +/- 1.71 of 11; keeper flagged in 43 % / 26 % of frames (max **one** keeper per
frame); actor flagged in 100 % of frames.

Three hard facts follow. **All 22 players appear in 1 frame out of 18 859 measured.** **Both
keepers never appear together, in any of the six matches.** **A single complete team (11 of 11)
appears in 1.2-5.9 % of frames.** SoccerCPD's seeding requirement — one moment per half with all ten
outfielders measured — is satisfied only rarely, and never for both teams at once.

Match-to-match spread of the mean is large (12.69 to 17.14 of 22), i.e. 0.58-0.78 of the squad. For
comparison our own pipeline measures 11.57/22 (0.526) and DeepMind's simulated broadcast camera
12.76 +/- 3.70 (0.580). StatsBomb sits at or above the top of that range, partly because human
annotators place some players manually (see 2.5).

### 2.4 The visible-area polygon, quantified

| statistic (StatsBomb units; metres in brackets) | 3895292 | 3930163 |
|---|---|---|
| vertices per polygon | 5.81 mean (5-8) | 5.62 mean (5-8) |
| polygon area | 2 376 mean | 2 814 mean |
| polygon area / full 120x80 pitch | **0.247** | **0.293** |
| x-span at pitch centre (y = 40) | 38.33 (**33.5 m**) | 40.42 (**35.4 m**) |
| polygon y-extent | 64.96 (55.2 m) | 69.74 (59.3 m) |
| polygon x-centre minus event(ball) x | +0.48 mean, +1.75 median, p10 -11.28, p90 +13.42 | +1.49 mean, +3.60 median, p10 -10.17, p90 +14.08 |
| player points falling outside their own polygon | 854 / 47 465 = **1.80 %** | 869 / 54 937 = **1.58 %** |

The x-span at pitch centre — **33.5 and 35.4 m** — is an independent confirmation of the censoring
window this project already uses: `synthesizer/imputation.py::visibility_mask` with the tuned
**33.8 m** half-width rectangle (`results/B4_TRANSFER_M3.md`), which was fitted to hit 11.8 visible
players on Metrica and had nothing to do with StatsBomb. The two disagree on the cross-pitch axis:
our rectangle is full pitch width (68 m), the real polygons cover 55-59 m of it and taper.

The ball-centring assumption also checks out: the polygon's x-centre sits within ~1.5 units (~1.3 m)
of the ball on average, with a p10-p90 spread of about +/- 11 units (~10 m).

Ball bias, measured directly on the visible set: mean distance from the event location is 22.24 /
24.29 units (19.5 / 21.3 m); 23.3 % / 17.8 % of visible players are within 10 units (8.75 m) of the
ball, 76.2 % / 69.7 % within 30 units (26 m).

### 2.5 Event coverage and sampling rate

360 frames exist for **81.5 % / 84.8 %** of all events (3 195 of 3 921; 3 324 of 3 920). Coverage by
type is uniformly high for in-play actions (Pass 875/1080, Ball Receipt 895/1041, Carry 789/905,
Pressure 233/278, Shot 29/29) and exactly zero for administrative events (Half Start/End, Starting
XI, Substitution, Tactical Shift, Player On/Off, Injury Stoppage, Referee Ball-Drop).

Sampling in time: median gap between consecutive 360 frames **1.0 s**, mean 1.74-1.95 s, p90 3.0 s,
max 145-223 s (stoppages). So it is an **event-triggered, irregular ~0.5 Hz-1 Hz snapshot**, not a
tracking feed. There is no continuity between frames — no player can be followed from one frame to
the next, because there are no identities to follow.

The spec's own caveats, quoted because they explain the 1.6-1.8 % of points outside the polygon:
*"StatsBomb 360 data is currently collected, for the most part, from broadcast video... Player
locations may be outside the visible area where these were manually placed."* So a small minority of
points are annotator reconstructions, not observations, and they are not flagged as such.

### 2.6 What the linked event data adds

The 426 matches also carry: `Starting XI` with a formation code and eleven position labels per team;
`Tactical Shift` events with a timestamp, a new formation code and a new position list (measured:
2-4 shifts per match — Serbia 3421 -> 352 at 31', -> 3412 at 60'; England 4231 -> 4141 at 85'); 8
substitutions per match with timestamps; and full per-player event locations with names. That is
StatsBomb's own formation label and its own change-point times, free.

---

## 3. The key question

### (a) Does the visible-area polygon give a labelled censoring boundary?

**Yes, and this is the one genuinely valuable thing here.** It is an explicit per-frame polygon in
pitch coordinates, produced by the provider, on ~1.3 million frames across 426 real broadcast
matches (426 x ~3 100). After de-rotating (2.2) it is directly comparable with:
- our simulator's rectangle (`visibility_mask`, 33.8 x 68 m) — agrees on the along-pitch axis to
  within 5 %, disagrees on the cross-pitch axis and on the taper;
- the SkillCorner `image_corners_projection` footprint (one match, `results/B4_TRANSFER_M3.md` §0) —
  same trapezoid, and this gives 426 more matches of it;
- our own measured broadcast footprint from the ManU corpus.

Caveat that must be carried: it is the polygon **StatsBomb annotators recorded**, not a projected
camera frustum, it is absent for some frames, and 1.6-1.8 % of the points they placed are outside it.
It is a good empirical description of a broadcast view, not a physical camera model.

### (b) Can we compute shape metrics on the visible subset and study the bias?

We can compute the metrics; we cannot study the bias. Shape metrics need only an unordered point set
per team, and 360 gives that (teammate/opponent split, GK flagged, already in attacking
coordinates) — so centroid, length, width, stretch index, hull area and template matching all run
on 360 frames without modification. Two blockers:

1. **No identities means no time-averaged per-player quantity.** Role occupancy, per-player mean
   position, the `fingerprint/roles.py` estimator over a window, EFPI role assignment across frames —
   all of it needs the same player to be recognisable across frames. In 360 it is not. Only
   *instantaneous* configuration metrics are computable. That kills arm A (IPW on per-player
   observations) as anything measurable on this data.
2. **No truth to compare against.** Below.

For the record, the shrinkage signature is plainly visible inside 360 itself. Outfield shape of the
visible subset, by how many of that team are visible (metres, 105 x 68):

| n visible | 3895292 length / width / stretch | 3930163 length / width / stretch |
|---|---|---|
| 4 | 14.87 / 24.24 / 10.31 | 14.99 / 26.21 / 10.70 |
| 6 | 18.87 / 29.44 / 11.05 | 21.15 / 34.03 / 12.63 |
| 8 | 21.49 / 32.56 / 11.58 | 24.06 / 40.71 / 13.96 |
| 10 | 21.74 / 31.49 / 10.88 | 23.83 / 39.40 / 13.18 |

corr(n_visible, length) = +0.269 / +0.237; corr(n_visible, width) = +0.185 / +0.198. Measured team
length even at 10 visible outfielders (21.7 / 23.8 m) sits **far below** the Rico-Gonzalez full-truth
reference range of 31-46 m. That is exactly the compression the proposal predicts — but with no truth
in this dataset, it cannot be attributed between censoring, annotation and situation. It is a
demonstration of the problem, not a measurement of it.

### (c) Is there any ground truth for the players who are not visible?

**No. None. Not a location, not an identity, not even a flag saying a player is missing.** A
freeze frame is a list of the points the annotator could see; absent players are simply absent from
the array. The only thing recoverable is the *count* of missing players (11 minus the visible count
per side, adjusted for red cards) — measured at **5.47 / 7.14 omitted per frame**, up to 19 — and
that is a censoring indicator, not truth.

Concretely: StatsBomb 360 is produced *from broadcast video*, by the same physical process that
limits us. It is a second sample of the same censored distribution, collected by better annotators
with more time. Using it to validate a correction for censoring would be circular — the correction's
target does not exist in the file.

There is no back door either. Event locations do name players, but only at the moment they touch the
ball, which is the most ball-biased sample of all; a player's mean event location is a ball-biased
estimate, not a position truth.

---

## 4. Where it sits against Metrica and SkillCorner

| property | Metrica (on disk) | SkillCorner opendata (on disk) | **StatsBomb 360** |
|---|---|---|---|
| matches | 2 | 1 (match 1886347) | 426 |
| all 22 players present | yes, every frame | yes (13.04 of 22 detected, 22 reported) | **never** (1 frame in 18 859) |
| ground truth for hidden players | **yes** | partial (SkillCorner's own extrapolation, flagged) | **no** |
| player identity | yes, persistent tracks | yes, persistent tracks | **no** (actor only, 6 %) |
| temporal continuity | 25 Hz continuous | 10 Hz continuous | event-triggered, median gap 1.0 s, no track continuity |
| explicit camera footprint | no (we simulate one) | yes, per frame, 37 530 frames, 1 match | **yes, per frame, ~1.3 M frames, 426 matches** |
| event/tactical labels | basic events | none | full events + formation labels + tactical-shift times |

Metrica is the only source of truth-behind-the-censoring we hold and remains the backbone of the
week-1 to week-5 harness. SkillCorner is the only source that pairs a real camera footprint with real
detected/extrapolated flags, but it is one match. **StatsBomb 360's unique contribution is scale on
exactly one axis: the shape and ball-relative placement of the visible region, across 426 broadcasts,
four leagues and two tournaments.** It adds nothing on truth, identity, or continuity — the three
things the correction arms actually need.

---

## 5. Verdict

**Use it for the censoring operator, not as a validation set. Budget one day, not a work package.**

The masking harness in weeks 1-5 of `docs/GEOMETRIC_TACTICS_PROPOSAL.md` rests on a censoring
operator that is currently a ball-centred 33.8 x 68 m rectangle tuned on a single scalar (11.8
visible/22) and sanity-checked against one SkillCorner match. Every bias number the project reports
inherits that shape. StatsBomb 360 turns that single-match check into a 426-match empirical
distribution, for free, at CPU cost, and it already agrees on the along-pitch axis (33.5 / 35.4 m
against our 33.8 m) while disagreeing on the cross-pitch axis (55-59 m of taper against our flat
68 m). That disagreement is worth resolving before the bias table is computed, not after.

### First experiment (about one day, ~230 MB of downloads)

1. Download `three-sixty` + `events` for ~30 matches spread across the four club competitions
   (Bundesliga 23/24, Ligue 1 21/22 + 22/23, La Liga 20/21) — 7.5 MB + 3.3 MB each. Delete after
   extraction; keep only the extracted polygons (a few MB of npz).
2. De-rotate every `visible_area` into a single physical frame using the acting team and period
   (2.2), then re-express it in **ball-relative** coordinates using the actor's event location.
3. Fit the empirical footprint: distribution of x-span at the near touchline / centre / far
   touchline, y-extent, ball offset in x and y, and the fraction of frames with no polygon.
4. Swap that empirical footprint into `synthesizer.imputation.visibility_mask` (sample a real polygon
   per frame instead of the rectangle) and re-run the week-1/2 masked-Metrica bias measurement.
5. **Pre-registered decision:** if the formation-label disagreement and the shape-bias magnitudes
   move by less than ~10 % relative between the rectangle and the real polygon, the rectangle is
   vindicated and this becomes a one-paragraph robustness note with a citation. If they move more,
   the real footprint becomes the default censoring operator for the whole project and every
   subsequent bias number is computed under it.

Two secondary uses, both cheap, neither on the critical path:

- **A distributional reality check on the simulator.** Compare masked-Metrica point clouds against
  real 360 point clouds on three statistics: visible count per team, distance-to-ball distribution,
  and visible-subset length/width. If the simulator's partial observations look distributionally
  unlike real ones, the bias curves are being computed on the wrong censoring. This is the honest
  external check that the whole week-2 result rests on, and it is the only one available.
- **Free formation change-point labels.** 426 matches with `Tactical Shift` timestamps and formation
  codes, plus substitution times. Weak labels only — cross-provider formation agreement is 30 % — but
  it is the only free labelled source of "the shape changed here" for a week-13 change-point
  detector, and it comes with the partial observations attached.

**What NOT to do with it, explicitly:** do not use 360 to validate any bias correction; do not use it
to score formation estimates against truth; do not attempt role assignment or per-player occupancy on
it. There are no identities and there is no truth. Anything of that shape would be measuring our
censored sample against their censored sample and calling the agreement validation.

---

## 6. Licensing and attribution

From `LICENSE.pdf` in the repo (StatsBomb Public Data User Agreement) and the README:

- **Attribution is mandatory.** README: *"If you publish, share or distribute any research, analysis
  or insights based on this data, please state the data source as StatsBomb and use our logo,
  available in our Media Pack."* Agreement 1.4: *"The User is required to accredit any publication of
  analysis formed from StatsBomb Data with the StatsBomb brand logo."* The logo is in the repo at
  `img/SB - Icon Lockup - Colour positive.png`. The thesis and any slide deck using these numbers
  must carry it.
- **No redistribution.** 1.2.1: the user may not *"edit, distort, distribute, reproduce, sell or in
  any way provide the data to any external or third party"*. Practical consequence for this repo:
  **do not commit StatsBomb JSON, and do not commit derived per-frame dumps that reconstitute it.**
  Aggregated statistics (the footprint distribution, the tables above) are analysis, not the data.
  Keep raw pulls in a gitignored cache, exactly as `data/soccernet` is handled.
- **No commercial exploitation** of the data or of any analysis derived from it (1.2.2). Fine for a
  BTP; would need renegotiating for anything monetised.
- Provided "as is", no warranty of accuracy or completeness (3.2, 3.4); StatsBomb may withdraw the
  service at any time (2.1). Users are asked to register name and email at
  statsbomb.com/resource-centre (2.2) — a request, not a technical gate.

---

## Reproduction

Measurement scripts were run from scratch and are not library code (they were throwaway
one-shot probes against downloaded JSON). Everything above can be regenerated with:
competitions from `statsbombpy.sb.competitions()`; file inventory from
`https://api.github.com/repos/statsbomb/open-data/git/trees/master?recursive=1`; per-match JSON from
`https://raw.githubusercontent.com/statsbomb/open-data/master/data/{events,three-sixty,lineups}/{match_id}.json`.
Matches measured: 3895292, 3930163 (full anatomy), 3857276, 3998853, 3773386, 3838017 (visible-count
statistics). Data used for this audit was deleted after measurement; nothing was added to the repo.

*Data source: StatsBomb. Analysis in this document is not the opinion of StatsBomb.*
