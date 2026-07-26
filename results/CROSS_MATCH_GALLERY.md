# Attacking REACHABILITY: cross-match pooled gallery, faces, and the harvest assessment

Follow-up to `results/CARRIER_CONSTRAINED_v2.md`, which closed per-player carrier attribution on the
grounds that the matcher term (0.609) was already at its gallery-LOTO ceiling and named REACHABILITY
(0.719) as the only term with headroom. This report attacks that term three ways.

Code: `tools/cross_match_gallery.py` (stages `embed` / `pool` / `drift` / `score` / `paired` /
`selftest`), `tools/face_harvest_probe.py` (stages `carrier_face` / `closeup_face` / `harvest` /
`selftest`). Artifacts: `results/cross_match_gallery/`.

## Verdict first

**Reachability moved. End-to-end moved a little, in a way that is real but small and does not
reopen the closure at the coverage we need.** Faces are dead on this footage — measured, not assumed.

| term | before (per-match gallery) | after (kit-conditioned pooled) | after (fully pooled) |
|---|---:|---:|---:|
| gallery has ANY entry for the true carrier (66 good-box moments) | 0.682 (45/66) | 0.712 (47/66) | **0.742 (49/66)** |
| **REACHABILITY** — true player is a candidate \| assigned (all 3) | **0.719** (23/32) | 0.793 (23/29) | **0.800** (20/25) |
| REACHABILITY, held-out only | 0.737 (14/19) | 0.833 (15/18) | 0.812 (13/16) |
| **held-out END-TO-END** (all hard constraints, frozen v1 point) | **0.400** (10/25) | **0.522** (12/23) | 0.476 (10/21) |
| dev end-to-end (tuning split, do not quote) | 0.333 (5/15) | 0.312 (5/16) | 0.312 (5/16) |
| assignments over all 3020 PASS events | 425 (14.1%) | 425 (14.1%) | 420 (13.9%) |

Reachability rose **+0.07 to +0.08**. Held-out end-to-end rose **+0.12** at unchanged coverage.
But the 95% Wilson intervals are 0.23-0.59 (before) and 0.33-0.71 (after) — they overlap almost
entirely — and the per-moment analysis below shows the gain is **six moments, all of them at the
abstention boundary**, with the dev split showing nothing.

**The closure should be softened, not revoked.** The specific v2 sentence that is now wrong is
"reachability is the cheaper deficit to attack" — it was attacked, it moved, and the end-to-end
number moved less than the reachability number did. The v2 headline (per-player attribution is not
defensible at useful coverage) survives: 0.522 held-out with a 0.33-0.71 interval at 14% coverage
still means one in two named passes is wrong.

---

## Experiment 1 — cross-match pooled gallery

### Setup

The gallery was per match: PRTreID crops sampled from tracks the identity chain named *in that
match*. The pooled gallery is keyed by **(club, player)** across the whole corpus of 9 registry
matches that carry a PRTreID named-tracks artifact:

`brighton_manutd, fulham_manutd, liverpool_manutd, manutd_brighton, manutd_liverpool, manutd_palace,
manutd_southampton, manutd_tottenham, tottenham_manutd`

Six of these had no cached gallery embeddings; those were built in one GPU job
(`--stage embed`, gallery crops only, queries untouched). Corpus total: **9646 usable gallery crops,
93 players, 9 matches**.

Three modes, scored like-for-like:

* `per_match` — the shipped baseline. **Wiring check: reproduces v1/v2 exactly** — all-3
  `carrier-selection error 0.154 (12/78), identification 0.438 (n=32), end-to-end 0.350 (n=40)`,
  held-out `0.360` baseline / `0.400` all-hard, reachability `0.719 (23/32)`. Every number in the
  v2 report is reproduced by this module before anything is changed.
* `pooled` — every crop of that player from every corpus match of his club.
* `pooled_kit` — pooled, restricted to matches where the club had the same **home/away** status,
  i.e. wore the same kit. (Home/away is used as the kit key because it is a fact about the fixture,
  not an inference from pixels. It is a proxy: an away side occasionally wears its home kit.)

Leave-one-track-out is preserved in every mode — the query's own `(match, chunk, track_id)` is
dropped. Crops from other matches are by construction not the query's track, so pooling cannot leak.

### Per-player gallery: per-match vs pooled

Full table: `results/cross_match_gallery/per_player_gallery.csv` (93 players x 9 matches).
Summary for the three labelled matches (`results/cross_match_gallery/pool_summary.csv`):

| match | club | players (per-match) | players (pooled) | players (pooled_kit) | **new players, pooled** | new, kit | crops (per-match) | crops (pooled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| manutd_liverpool | Man Utd | 10 | 19 | 15 | **+9** | +5 | 453 | 3972 |
| manutd_liverpool | Liverpool | 8 | 13 | 8 | **+5** | +0 | 824 | 1250 |
| manutd_tottenham | Man Utd | 11 | 19 | 15 | **+8** | +4 | 468 | 3972 |
| manutd_tottenham | Tottenham | 13 | 17 | 13 | **+4** | +0 | 1155 | 2158 |
| manutd_brighton | Man Utd | 8 | 19 | 15 | **+11** | +7 | 425 | 3972 |
| manutd_brighton | Brighton | 12 | 15 | 12 | **+3** | +0 | 552 | 864 |

**Players going from zero samples to non-zero: 3 to 11 per side, 40 across the six (match, club)
pairs.** Man Utd gains most (8.4x more crops; 3972 pooled vs 425-468 per match) because it appears in
all 9 corpus matches. Note `pooled_kit` adds **nothing** for the opponents: each opponent appears in
exactly two corpus fixtures (home and away), so same-kit pooling degenerates to the per-match
gallery for them. Only Man Utd benefits from kit-conditioned pooling.

### The passing volume that is still unreachable

Weighting the roster by the Sofascore `totalPass` oracle turns "players in the gallery" into
"share of real passing volume the closed set can represent":

| match | per-match | pooled | pooled_kit | biggest zero-sample passers (oracle passes) |
|---|---:|---:|---:|---|
| manutd_liverpool | 0.639 | **0.777** | 0.671 | van Dijk (73), de Ligt (55), Onana (51), Alisson (33) |
| manutd_tottenham | 0.859 | **0.887** | 0.876 | Vicario (44), de Ligt (36), Onana (26) |
| manutd_brighton | 0.786 | **0.867** | 0.846 | de Ligt (54), Verbruggen (36), Onana (26) |

Unreachable passing volume falls from **14-36% to 11-22%**. Pooling closes roughly a third to a half
of the volume gap — and then stops dead, because the residual is the same four or five names in
every match.

**The hard structural finding: `Matthijs de Ligt`, `Andre Onana`, `Virgil van Dijk` and
`Guglielmo Vicario` are named ZERO times across all 1664 named tracks in the entire 9-match corpus.**
This is not thin sampling that more matches would fix; it is a systematic identity-chain failure.
Two of the four are goalkeepers, and goalkeepers are barely tracked at all (1105 `goalkeeper` rows
against 122732 `player` rows in one match) because a ball-following broadcast camera rarely frames
them. Those two GKs plus the two centre-backs are among the **highest-volume passers on the pitch**
(van Dijk 73 passes, de Ligt 36-55, Onana 26-51). No amount of cross-match pooling reaches them.

### Appearance drift across matches (the risk the brief flagged — it is real and large)

Mean same-player cosine similarity, within a match vs across matches
(`results/cross_match_gallery/appearance_drift.csv`, 35 players with >= 2 corpus matches):

| | mean cosine |
|---|---:|
| same player, **within** one match | **0.777** |
| same player, **across** matches | **0.641** |
| across matches, same home/away kit | 0.638 |
| across matches, different kit | 0.640 |

Per club the picture is sharper:

| club | within-match | cross-match | note |
|---|---:|---:|---|
| Liverpool | 0.78-0.83 | **0.43-0.53** | red at Anfield vs white away kit at Old Trafford |
| Man Utd | 0.71-0.79 | 0.56-0.71 | red home vs away kits, 9 fixtures |
| Tottenham | 0.75-0.83 | 0.70-0.78 | |
| Brighton | 0.76-0.88 | 0.71-0.83 | |

Liverpool's cross-match same-player similarity **collapses to 0.44-0.53** — far below the 0.88
operating threshold. Pooling Liverpool's Anfield crops into a query at Old Trafford adds crops that
can never win an argmax, and can only add noise to the runner-up. That is exactly the failure mode
the brief warned about, measured.

For Man Utd, `cross_same_kit` (0.638) and `cross_diff_kit` (0.640) are indistinguishable, so the
drift is **not** primarily kit — it is match-to-match session drift (lighting, encoder, camera,
hair, boots). Kit is the extreme case, not the mechanism.

This is why `pooled_kit` beats `pooled` on the held-out split despite adding fewer players: it keeps
the Man Utd gain and refuses to import the Liverpool white-kit crops.

### End-to-end, at the frozen v1 operating point (min_sim 0.88, min_margin 0.01)

The operating point is the one v1 froze on the dev match by a label-free rule; it was **not
re-tuned** for this experiment, so the comparison is clean. Full grid:
`results/cross_match_gallery/mode_scores.csv`.

| mode | config | split | n assigned | end-to-end | ident | reachability | assignments / 3020 PASS |
|---|---|---|---:|---:|---:|---:|---:|
| per_match | baseline | held | 25 | 0.360 | 0.500 | 0.778 (14/18) | 394 |
| per_match | all hard | held | 25 | **0.400** | 0.526 | 0.737 (14/19) | 425 |
| pooled | all hard | held | 21 | 0.476 | 0.625 | 0.812 (13/16) | 420 |
| pooled_kit | baseline | held | 20 | 0.500 | 0.667 | 0.867 (13/15) | 393 |
| **pooled_kit** | **all hard** | **held** | **23** | **0.522** | **0.667** | 0.833 (15/18) | **425** |
| per_match | all hard | dev | 15 | 0.333 | 0.357 | 0.643 (9/14) | — |
| pooled_kit | all hard | dev | 16 | 0.312 | 0.357 | 0.714 (10/14) | — |

Wilson 95% intervals on held-out end-to-end: per_match `0.234-0.593`, pooled `0.283-0.676`,
pooled_kit `0.330-0.708`. **They overlap almost completely.**

I also ran a dev-only re-selection of the threshold for the pooled modes (the pooled similarity
distribution is different, so the frozen point is not obviously right for them). It picked
`min_sim 0.92 / min_margin 0.005` on 8 dev assignments and **did not transfer**: held-out
0.522 -> 0.438 (`pooled_kit`), 0.476 -> 0.412 (`pooled`). Same overfit-to-15-dev-moments lesson as
v2's hubness correction. Grids: `results/cross_match_gallery/dev_grid_*.csv`. The quoted numbers use
the untouched v1 point.

### What actually changed, moment by moment (`--stage paired`)

This is the number that decides the verdict. Restricting to held-out moments where **both** the
baseline and `pooled_kit` assigned a name:

| | value |
|---|---:|
| moments both arms assigned | 21 |
| **predictions that differ on those 21** | **0** |
| baseline correct / pooled_kit correct on those 21 | 10 / 10 |
| McNemar discordant pairs | **0** |

**The pooled gallery did not fix a single identification.** On every moment both arms answer, they
answer *identically*. The whole delta is six moments where the assign/abstain decision changed:

| truth | baseline | pooled_kit |
|---|---|---|
| Noussair Mazraoui | *abstain* | **Noussair Mazraoui** |
| Lisandro Martinez | *abstain* | **Lisandro Martinez** |
| Manuel Ugarte | Bruno Fernandes (wrong) | *abstain* |
| WRONG_PLAYER_BOXED | Bruno Fernandes (wrong) | *abstain* |
| Toby Collyer | Diogo Dalot (wrong) | *abstain* |
| Rasmus Hojlund | Bruno Fernandes (wrong) | *abstain* |

All six move in the right direction, and the mechanism is coherent rather than lucky: pooling gives
the *true* player enough extra crops that his best match clears 0.88 (Mazraoui, Martinez), and it
also raises the similarity of the hub attractor's rivals, which narrows the top-1/top-2 margin and
converts three Bruno-Fernandes hub errors into abstentions. Bruno has 887 pooled crops — the largest
gallery in the corpus — and three of the four suppressed errors were his.

But it is six moments out of 49 judged, the dev split shows the opposite sign (0.333 -> 0.312), and
no single identification improved. Treat +0.12 as **suggestive, not established**.

---

## Experiment 2 — face feasibility, measured

No new dependency: OpenCV's shipped Haar cascades (`haarcascade_frontalface_alt2` +
`haarcascade_profileface`), crops upscaled 3x first so Haar's 20x20 minimum window can fire, detected
heights scaled back to original-crop pixels. `insightface` is not installed and was not added.

### At carrier moments (the moments attribution actually needs)

Geometry over **all 1095 resolvable carriers** in the three labelled matches, using the pipeline's
own `estimate_player_box`; head = 1/8 of standing height, face (chin to hairline) = 1/9.

| | min | p10 | p25 | **median** | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| player box height (px) | 64 | 74 | 80 | **89** | 101 | 114 | 134 |
| head height (px) | 8.0 | 9.2 | 10.0 | **11.1** | 12.6 | 14.2 | 16.8 |
| face height (px) | 7.1 | 8.2 | 8.9 | **9.9** | 11.2 | 12.7 | 14.9 |

**Fraction of carrier moments clearing the ~50 px face bar: 0.0000 (0 / 1095).** The single tallest
carrier in three matches has a 14.9 px face — a third of what face recognition needs.

Real detector on 1093 decoded carrier crops: **17 detections (1.6%)**, detected "face" heights
7.3-29.7 px, **0 at >= 50 px**. Since the geometric ceiling on head height is 16.8 px, every
detection above ~17 px is a false positive on kit texture or turf. Faces do not exist at carrier
moments on this footage. `results/cross_match_gallery/carrier_head_geometry.csv`,
`carrier_face_detection.csv`.

### On close-up crops (where jersey numbers are read)

3570 crops of `brighton_manutd h1_chunk_000` are still on disk from the original close-up probe and
are row-aligned with that chunk's rows of `results/closeup_anchor_probe/closeup_reads.csv`, so the
jersey-read outcome is known per crop for free (alignment asserted: JPEG height == recorded `box_h`).

| | value |
|---|---:|
| crops | 3570 |
| box height px (median / p90 / max) | 113 / 143 / 1019 |
| geometric face height px (median / max) | 12.6 / 113.2 |
| any face detected (Haar) | 6.3% (226/3570) |
| **face >= 50 px** | **0.56% (20/3570)** |

Split by whether the crop produced a jersey anchor (`conf >= 0.70`, not `illegible`):

| | no face | face | total |
|---|---:|---:|---:|
| number read FAILED | 2871 | 191 | 3062 |
| number read = ANCHOR | 473 | 35 | 508 |

**Answer to the honest question: number-read failures with a usable (>= 50 px) face: 20 out of 3062
(0.65%).** So in roughly ten minutes of match footage, a perfect face recogniser would recover at
most ~20 crops — and those 20 sit inside a handful of camera shots, so the distinct *identities*
recovered is smaller still, likely single digits. Meanwhile the same chunk already produced 508
jersey anchors. Faces would add well under 5% to the anchor supply, in exchange for a face
recognition system and a reference-face database we do not have.

**Do not build face recognition.** Not "not yet" — the pixels are not there at carrier moments
(0/1095), and where the pixels do exist (close-ups) the jersey reader is already 25x more productive.

---

## Experiment 3 — harvesting the footage we already own

Assessed from `results/closeup_anchor_probe/closeup_reads.csv`, which already logs **every** eligible
close-up person crop of a full match (`brighton_manutd`, 43020 crops over 11 chunks) with its box
height and the recogniser's confidence. No new footage, no new runs.

| box height band | crops | anchors | non-anchor | median geometric face px |
|---|---:|---:|---:|---:|
| 100-150 px | 32199 | 4150 | 28049 | 12.6 |
| 150-250 px | 3982 | 381 | 3601 | 18.9 |
| 250-400 px | 1690 | 161 | 1529 | 34.8 |
| **400+ px** | **5149** | 866 | **4283** | **74.2** |

Grouped into camera shots (new shot when the sampled-frame gap exceeds 25):

| | count |
|---|---:|
| close-up shots in the match | 669 |
| shots containing a >= 250 px person crop | 176 — **60 yield zero anchors** |
| shots containing a >= 400 px person crop | 150 — **53 yield zero anchors** |
| shots containing a >= 600 px person crop | 134 — **51 yield zero anchors** |

### Assessment

**There is a real unmined pool, and it is smaller than it looks.** ~5100 crops per match sit at
>= 400 px (true close-ups: warm-up, celebrations, substitutions, replays, bench), of which 4283
produce no anchor. But crops are not opportunities — they collapse into **150 close-up shots per
match, of which 53 (35%) currently yield nothing**. Across the 9-match corpus that is roughly
**450-500 shots**, and each shot is one player-appearance at best. Realistically this is tens of new
`(player, match)` anchors, not thousands.

Two things must be said plainly about what that pool can and cannot do:

1. **These crops cannot become gallery crops directly.** The current design is deliberately
   domain-matched: gallery crops are ~90 px wide-shot crops from the same camera as the ~90 px
   queries. A 400-1000 px close-up embedded with PRTreID and matched against a 90 px query is a
   large domain gap, and Experiment 1 just measured that even a *same-domain* cross-match gap
   (0.777 -> 0.641 cosine) is enough to hurt. Harvested close-ups have to route through the existing
   path — read a number, name a track, then sample that track's wide-shot frames — which is exactly
   what `tools/closeup_anchor_probe.py` already does.
2. **So the harvest is bounded by the number reader, not by the footage.** The 53 zero-anchor shots
   per match are shots where the reader already looked and failed (numbers occluded, side-on,
   celebration, blurred). Relaxing the 0.70 anchor gate is the cheap lever, and v2 already measured
   what the reader's low-confidence tail is worth: on carrier crops, dropping the confidence floor
   from 0.60 to 0.05 raised the read rate from 11% to 44% at flat 0.00-0.06 precision. There is no
   reason to expect the close-up tail to behave better without evidence.

The one genuinely promising item in the pool is **pre-match line-up / warm-up footage**, where
players stand still, face the camera, and are shown with name captions. That is not in
`closeup_reads.csv` because the shot classifier's close-up frames are sampled from in-play chunks.
Quantifying it needs a pass over the pre-kickoff portion of the source files, which was out of scope
here — flagged as the one harvest lead worth an hour, and it targets the exact failure mode
Experiment 1 found (de Ligt, Onana, van Dijk, Vicario never named anywhere).

---

## Verdict on the closure

**Revise it partially. Do not reopen per-player attribution as shippable.**

What v2 got right and still stands:
* The matcher term is at its ceiling. Confirmed again: on 21 shared held-out moments the pooled
  gallery produced **identical predictions** to the per-match gallery. More reference crops do not
  make PRTreID a better identifier at ~90 px.
* Per-player attribution is not defensible at useful coverage. 0.522 held-out with a 0.33-0.71
  interval at 14% of passes means about half the named passes are wrong.

What v2 got wrong:
* "Reachability is the cheaper deficit to attack" — half-right. Reachability *did* move
  (0.719 -> 0.79-0.80; unreachable passing volume 14-36% -> 11-22%) and end-to-end moved with it, but
  by less, through abstention rather than through better identification.
* Treating the gallery deficit as a sampling problem. It is not. **Four of the highest-volume
  passers in the labelled matches are named zero times in 1664 named tracks across 9 matches.**
  Pooling nine matches did not name Onana once. That is a naming-pipeline defect (goalkeepers, and
  #4 centre-backs), not a data-volume defect, and it is the binding constraint on reachability now.

What to ship, if anything:
* **`pooled_kit` + hard constraints is free and weakly better.** Same operating point, same coverage
  (425 of 3020 passes, 14.1%), held-out 0.400 -> 0.522, and its six changed moments all move the
  right way. It costs one GPU embed pass per new match and no new logic. Ship it as the gallery
  default; **do not claim the precision gain** (n=6, dev split negative, intervals overlap).
* Do **not** pool across different kits. Liverpool's cross-fixture same-player similarity is
  0.43-0.53; importing those crops is strictly noise.
* Do **not** build face recognition. 0/1095 carrier moments clear 50 px; 20/3062 failed close-up
  reads have a usable face.
* The next lever, if Sid wants one more swing, is **naming goalkeepers and the missing centre-backs**
  — pre-match/line-up footage with name captions, which is footage we already own and have never
  touched. That is a reachability fix aimed at the 20-30% of passing volume that is *provably*
  unreachable today, and it is the only remaining path that is not a better matcher.

## Caveats

* 90 labels; 49 judged and 23-25 assigned on the held-out split. Every precision here carries a
  +-0.19 interval. Deltas below 0.15 are unmeasured, and +0.12 is below that bar.
* The mode comparison (`per_match` / `pooled` / `pooled_kit`) was read off the held-out split.
  That is three looks at a 25-moment set. The dev split, which is the only split with any
  pre-registration claim, shows **no** pooled gain (0.333 -> 0.312).
* `pooled_kit` uses home/away as the kit key. It is a fixture fact, not a pixel measurement; an away
  side wearing its home kit would be mis-grouped. The Liverpool drift numbers say the grouping is
  doing real work, but it was not verified kit by kit.
* Cross-match pooling assumes player names are spelled identically across matches (all names come
  from the same Sofascore-backed lineup assignment, and matching is `norm_name`-normalised).
* The face measurement uses Haar, which is a weak detector. It understates detection. But the
  *geometric* ceiling — 16.8 px maximum head height over 1095 carrier moments — is detector
  independent, and it is the number the verdict rests on.
* The harvest assessment is one match (`brighton_manutd`) because it is the only one with a full
  `closeup_reads.csv`. Shot counts are per-match and were multiplied by 9 for the corpus estimate.
* Everything inherits v2's caveats: 15.4% carrier-selection error before identification starts, a
  per-track team label that is wrong ~17% of the time, and named tracks whose own read precision is
  marked UNVERIFIED upstream.
