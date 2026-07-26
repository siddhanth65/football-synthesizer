# Carrier attribution probe -- closed-set player id at pass moments (FEASIBILITY, not production)

Tests one idea: stop naming tracks globally (open set, 750-1045 track ids per chunk for 22 players,
1-4% attribution -- `results/PLAYER_ANALYSIS_v2.md`) and instead identify only **the carrier at pass
moments** against a **closed set** built from the 22-player roster.

Code: `tools/carrier_attribution_probe.py` (staged: `funnel` / `embed` / `select` / `score` /
`labelpack`). Artifacts: `results/carrier_attr/`. Matches: `manutd_liverpool` (dev/threshold),
`manutd_tottenham`, `manutd_brighton` (hold-outs). Embedder: PRTreID (`tools/prtreid_probe.py`),
the same one the identity chain uses. Carrier rule: `tools.event_ledger._contact_frame` +
`nearest_carrier`, **reused unchanged, not forked**.

## Verdict first

**Worth building properly -- but not for the reason the idea was proposed, and not before Sid labels
the pack.**

- Attribution coverage rises from a measured **0.9-2.0%** baseline to **11.6-15.3%** of all
  operating-point PASS events: a **~8-13x coverage lift**, and ~95% of it lands on tracks the
  current mechanism could never name.
- After discounting by the two measurable error sources (incomplete closed set, matcher precision)
  the *estimated correct* coverage is **6.5-9.0%** -- still a **4-8x** lift, but that number is an
  estimate, not a measurement. Precision is unmeasured until the labelling pack comes back.
- **The stated premise is false.** "At pass moments the camera is centred on the ball so the carrier
  is well framed" does not hold: carrier crops at pass moments are the *same size* as a random
  tracked player (median box height 86-92 px vs 86-92 px, table below). There is no appearance
  advantage at pass moments. The lift comes entirely from the **closed set** (8-13 candidates instead
  of ~900 tracks), which is a different and weaker claim than the one proposed.
- **Binding constraint = carrier resolution, not identification.** Only 31.3-41.9% of PASS events
  have a resolvable carrier at all, and the dominant loss (37-57% of all events) is *no ball+player
  sample in the kick window* -- a ball/live-play coverage problem, not an identity problem.

## The funnel, per match

| stage | manutd_liverpool | manutd_tottenham | manutd_brighton |
|---|---:|---:|---:|
| operating-point PASS events (BAS, frozen 0.40 + 1 s NMS) | 981 | 1079 | 960 |
| -- with a ball+player sample in the kick window | 475 (48.4%) | 468 (43.4%) | 600 (62.5%) |
| -- carrier within 3 m (**carrier-resolvable**) | **355 (36.2%)** | **338 (31.3%)** | **402 (41.9%)** |
| median carrier distance (m) | 0.91 | 1.00 | 1.04 |
| gallery: named tracks / crops / players | 221 / 1277 / 18 | 282 / 1623 / 24 | 166 / 977 / 20 |
| gallery coverage by oracle pass volume | **63.9%** | **85.9%** | **78.6%** |
| assigned at the frozen operating point | 122 | 125 | 147 |
| assignment rate among resolvable carriers | 34.4% | 37.0% | 36.6% |
| abstention rate among resolvable carriers | 65.6% | 63.0% | 63.4% |
| **ATTRIBUTION COVERAGE (of all PASS events)** | **12.4%** | **11.6%** | **15.3%** |
| baseline mechanism on the same events (carrier's own track already named) | 0.9% | 2.0% | 1.2% |

The baseline row is the honest apples-to-apples control: it is the *existing* architecture measured
inside this same funnel, and it reproduces `PLAYER_ANALYSIS_v2`'s 1-4% independently. Of the closed-
set assignments, **117/122, 115/125, 144/147** are on carrier tracks the existing mechanism has no
name for -- the lift is genuinely new attribution, not a restatement.

### Where the events die (stage 2 is the ceiling)

| loss | manutd_liverpool | manutd_tottenham | manutd_brighton |
|---|---:|---:|---:|
| no ball+player sample within the kick window | 506 (51.6%) | 611 (56.6%) | 360 (37.5%) |
| sample exists but nearest player > 3 m | 120 (12.2%) | 130 (12.0%) | 198 (20.6%) |

The first row is the whole game. It is replay/close-up segments (no calibration, no tracking) plus
gaps in the post-`link_ball` usable ball track -- not an identity failure. Fixing identification
perfectly caps attribution at 31-42%; fixing ball/live coverage is the only way past that.

## Gallery (stage 1) -- who exists in the closed set

Gallery = PRTreID embeddings of up to 6 frames per track that the identity chain already named
(`outputs/identity/<match>_named_tracks_both2_prtreid.parquet`), cropped from the **same wide-shot
domain** as the queries. Deliberately not the close-up anchor crops: those are a different scale
domain, and cross-domain matching is exactly what already limits `wire_anchors`.

Per-team gallery population (of the ~14-16 players per team who touch the ball):

| match | team 0 players / crops | team 1 players / crops |
|---|---|---|
| manutd_liverpool | 10 / 453 | 8 / 824 |
| manutd_tottenham | 11 / 468 | 13 / 1155 |
| manutd_brighton | 8 / 425 | 12 / 552 |

Per-player sample counts are extremely uneven (dev match, team 0: 151, 83, 66, 42, 31, 28, 24, 12,
10, 6 crops) -- the hero-shot concentration from the anchor stage propagates straight into the
gallery. **Several players have zero gallery samples, and that is the finding the task anticipated:**
weighted by the oracle's `totalPass`, the gallery can represent only **63.9% / 85.9% / 78.6%** of the
real passing volume. Every pass by an absent player is *guaranteed* to be misassigned or abstained --
the "closed" set is not closed.

A second, smaller defect: the aligned table's per-frame `team` flips within a track, so 7-9% of
crops would carry the wrong club under a naive per-frame label. The probe uses the per-track majority
team for queries and the roster-resolved team for the gallery.

## Frozen operating point -- and the selection rule that failed

The pre-declared rule (written into the module before any score was computed) was: *max assignment
rate subject to same-track identity consistency >= 0.90 on >= 10 tracks, dev match only.*

**That rule is unmeasurable on real data, and the reason is itself a result.** Carrier tracks are
one pass long: 355 resolvable pass events on the dev match come from **344 distinct carrier tracks**;
333 of them carry exactly one pass, only 11 carry two. There is essentially no "same track at
consecutive pass moments" to check. (This also explains the 1-4% baseline from the other side: a
track that lives one pass long can never be named globally and then reused.)

Stated fallback, using a pre-existing pre-committed bar rather than a new one
(`tools.prtreid_probe.PRECISION_BAR = 0.80`): **max retention subject to dev-match gallery-LOTO
pseudo-truth precision >= 0.80 on >= 50 crops.** Chosen on `manutd_liverpool` only, written to
`results/carrier_attr/frozen_threshold.json`, applied verbatim to both hold-outs:

```
min_sim = 0.88, min_margin = 0.01   (dev gallery precision 0.812, retention 0.567)
```

The gallery-LOTO task -- re-identify each gallery crop against the same-team gallery with its own
track removed -- is the same 8-16-way closed-set problem the queries face, so its precision curve is
the only pre-label evidence available. Dev-match curve (`results/carrier_attr/dev_gallery_sweep.csv`):

| min_sim \ min_margin | 0.00 | 0.01 | 0.02 | 0.04 |
|---|---|---|---|---|
| 0.80 | .649 (n=1272) | .776 (771) | .819 (453) | .895 (190) |
| 0.88 | .683 (1181) | **.812 (724)** | .853 (429) | .913 (184) |
| 0.92 | .714 (984) | .853 (597) | .914 (338) | .955 (154) |

Unfiltered top-1 accuracy on this task is **0.647** -- PRTreID at 90 px crop scale is far from a
solved identifier; the gate buys precision by discarding 43% of crops. Note the pseudo-truth labels
come from the Koshkina/PRTreID chain, whose own read precision is marked UNVERIFIED, so these
precisions are optimistic.

Hold-out gallery-LOTO precision at the frozen point: **0.794** (tottenham), **0.752** (brighton) --
both below the 0.812 dev value, the expected off-dev drop, and brighton falls under the 0.80 bar.

## Indirect validation (no manual labels yet)

**(a) Per-player pass counts vs the Sofascore oracle.** Spearman on the joined players, with a
2000-draw permutation null:

| match | Spearman | n players | permutation p | top-5 overlap (of 5) |
|---|---:|---:|---:|---:|
| manutd_liverpool | **0.534** | 15 | 0.022 | 3 |
| manutd_tottenham | 0.374 | 20 | 0.058 | 1 |
| manutd_brighton | 0.304 | 18 | 0.105 | 2 |

Positive in all three, significant in one. This is what ~50-70% precision plus a truncated gallery
looks like -- consistent with the architecture working, nowhere near proof that it does.

**(b) Same-track consistency: structurally unmeasurable** (1 multi-assignment track per match). Not
a pass, not a fail -- the validator does not exist at this track length. Reported as dead.

**(c) Roster / substitution sanity.** Off-roster assignments: **0 / 0 / 0** (the closed set cannot
emit a name that did not play -- true by construction, so this is a wiring check, not evidence).
The sharper check is substitution windows. Of assignments to players with <= 40 minutes (the only
ones where a half is provably impossible): **2 of 13 are impossible** -- both are Christian Eriksen
(18 min, second-half sub) assigned in h1 of `manutd_tottenham`. Small n, but it is a direct,
label-free observation of the error mode: when the true carrier is missing from the gallery, the
matcher hands the pass to whoever is nearest in embedding space.

Where the windows *are* satisfiable, they check out: Casemiro (45 min, dev match) 100% h1; Bruno
Fernandes (42 min, red card, tottenham) 100% h1; Toby Collyer, Darwin Nunez, Lucas Bergvall, Pape
Matar Sarr, Garnacho-as-sub, Solly March all 0% h1.

## The premise that did not survive

| match | carrier crop height p25/p50/p75 (px) | random tracked player p25/p50/p75 (px) |
|---|---|---|
| manutd_liverpool | 81 / 89 / 101 | 80 / 91 / 104 |
| manutd_tottenham | 82 / 92 / 101 | 81 / 92 / 104 |
| manutd_brighton | 75 / 86 / 100 | 75 / 86 / 99 |

Identical. The broadcast keeps the ball near the *centre* of frame but does not zoom in for it: the
carrier at a pass moment is the same ~90 px tall figure as anyone else on the pitch. The proposed
"conditions are best at exactly those instants" advantage is not present in this footage.

What *is* real: the candidate set shrinks from ~900 open-set track ids to 8-13 named players on the
carrier's tracked team. That is the entire mechanism, and it is enough for an 8-13x coverage lift.

## Honest precision arithmetic (why the headline needs the labels)

`P(correct | assigned) <= P(true carrier is in the gallery) x P(matcher picks right | in gallery)`,
using the oracle-pass-weighted gallery share and the gallery-LOTO precision at the frozen point:

| match | gallery pass share | gallery-LOTO precision | est. precision bound | est. *correct* coverage |
|---|---:|---:|---:|---:|
| manutd_liverpool | 0.639 | 0.812 | ~0.52 | ~6.5% |
| manutd_tottenham | 0.859 | 0.794 | ~0.68 | ~7.9% |
| manutd_brighton | 0.786 | 0.752 | ~0.59 | ~9.0% |

Both factors are optimistic (gallery crops come from tracks good enough to have been named; the
pseudo-truth labels are themselves unverified), and query crops at the kick are, if anything, harder
than gallery crops. Treat 50-70% as an upper band, not a measurement. Against a baseline whose own
precision is also unmeasured, the coverage lift is the defensible claim; the precision claim waits
for labels.

## Labelling pack (ready for Sid)

`results/carrier_attr/labelpack/` -- **90 pass moments**, 30 per match, stratified by half x carrier
depth band (far / mid / near) so it is not all easy close-ups. Per moment:

- `<id>_crop.jpg` -- the carrier crop at the kick moment (the exact pixels the model embedded);
- `<id>_ctx-2..+2.jpg` -- five full frames around the kick (stride 10) with the carrier boxed yellow,
  so the labeller can use motion and neighbouring shirt numbers;
- a row in `labels.csv` with `player_name`, `confidence_1to3`, `notes` blank.

**Model predictions are deliberately absent from the pack.** `WRONG_PLAYER_BOXED` is an allowed
`player_name` value -- it separates carrier-selection error from identification error, which the
funnel currently cannot. Regenerate with
`python -m tools.carrier_attribution_probe --stage labelpack`.

## What to build (and what not to)

1. **Do not build this as-is.** Score the 90 labels first; the whole precision story is an estimate.
2. **The highest-value fix is not identification.** 37-57% of pass events have no ball+player sample
   at the kick. That is ball coverage and live-play classification, and it caps everything.
3. **Close the closed set.** 14-36% of passing volume belongs to players with no gallery entry. A
   gallery seeded from *any* confident evidence (kit + role + position priors, not only close-up
   number reads) matters more than a better embedder.
4. **Read numbers on the carrier crop.** In the labelpack frames, back numbers are legible on several
   players at exactly this zoom (`20`, `37`, `38`, `11`, `8`, `10`, `17`, `4` in one sampled frame).
   A recognizer run on the ~350-400 carrier crops per match is a far cheaper and more direct route to
   a name than 90 px appearance ReID, and it fails loudly (unreadable) instead of silently.

## Caveats

- 3 matches, one embedder, one carrier heuristic. No manual labels yet -- every precision number here
  is either pseudo-truth (identity-chain labels) or an arithmetic bound.
- The gallery inherits every bias of the anchor stage: hero-shot players are over-sampled, and the
  identity chain's read precision is marked UNVERIFIED upstream.
- The 3 m carrier radius, the -0.6..+0.2 s contact window and the BAS 0.40 operating point are all
  inherited frozen from `tools/event_ledger.py` and `tools/bas_validate.py`; none were re-tuned here.
- Attribution coverage is quoted against *operating-point PASS events*, not against Sofascore's
  attempted-pass totals. On the dev match those events are ~981 vs 971 oracle attempts, so the two
  denominators are close, but they are not the same denominator.
