# GT audit — the SoccerNet-GSR `team` attribute on the clips we lose to side flips

Act v8-W0. The WACV 2026 Broadcast2Pitch paper (GS-HOTA 61.48 test, the leader) states that test
samples **SNGS-126, SNGS-131 and SNGS-197** *"have incorrect team annotations in the SoccerNet-GSR
test set"*. Three of those are in our audited v6 flip set (`results/gsr_benchmark/
gsr_v6_legitimacy_audit.json`: SNGS-126, **130**, 131, 197), each costing ~-33 GS-HOTA on its clip.
This act tests their statement with our own instruments, on the laptop, on CPU.

**Headline: their statement is confirmed, exactly and independently — and it is incomplete. All 164
sequences of SoccerNet-GSR were scanned. Every (game, half) group is unanimous about which kit wears
the `left` label except five clips: SNGS-092 (valid), SNGS-111 (train), SNGS-126, SNGS-131, SNGS-197
(test). Every one of the five is a *second-half* clip carrying the *first-half* team-to-side
convention. The three test clips are precisely the three Broadcast2Pitch names; the two dev clips are
new. SNGS-130 — our fourth flip, which they do not name — is NOT a GT error: its GT is correct by
0.11 m of mean pitch-x, and losing it is our own estimation noise. So 3 of our 4 test flips are
unreachable by any correct method, worth ~-2.4 GS-HOTA of our 53.09, and the leader eats a larger
absolute penalty (~-3.6) on the same three clips.**

---

## 1. Instruments

Three, in increasing order of authority. All ground-truth only; no prediction of ours is involved.

| # | instrument | what it can prove | what it is blind to |
|---|---|---|---|
| 1 | **`gk_self` on GT positions** (`tools/gsr_teamside.side_scores`) — "a keeper's own team defends the goal he stands in", the *definition* of the GSR `left`/`right` attribute; 113/113 on GT across the dev split | a keeper labelled against his own position | whether the *outfield* labels agree with the keeper |
| 2 | **kit clustering vs the GT split** (`generator.teams.jersey_color` → KMeans(2)) — median CIELAB of non-grass torso pixels per crop | individual players filed under the wrong team | a *whole-clip* swap, which leaves both groups perfectly coherent |
| 3 | **cross-clip consistency** (`game_consistency`) — teams change ends once, at half time, so every clip of one (game, half) must give the same kit the same side | a whole-clip swap | nothing relevant; it needs >= 2 clips per game-half, which GSR always has (6-11) |

Instrument 3 is the decisive one and it is the reason this audit reaches a different answer from the
one instruments 1 and 2 give on their own. A clip whose entire `team` attribute is swapped is
*internally flawless*: the keeper is labelled by his own goal, both kit groups are pure, nothing
inside the file contradicts anything else inside the file. Only the outside world contradicts it.

The kit handle used by instrument 3 is a **per-game KMeans(2) over each clip's per-side median
CIELAB**, i.e. the two kits of that match, with the assignment margin recorded. An earlier version
used "the redder side" (higher `a*`) and produced one false positive, SNGS-160, whose two kits differ
in lightness but not at all in `a*` (L\* 42.7 vs 77.3, `a*` 2.0 vs 2.0). The prototype version calls
it correctly; the false positive is recorded here because it is the only difference between the two
runs and a reader should know the weaker rule existed.

Runnable check: `python -m tools.gsr_gt_audit --demo` (asserts the `gk_self` sign flip, the verdict
ordering, and that the kit handle survives an `a*`-tied kit pair).

## 2. Per-sequence verdicts

Sign convention throughout: GT `left` is re-coded as cluster 0, so **negative is correct** — a
positive score is a rule that points against the label with exact inputs.

| clip | split | game/half | GT `meanx` | GT `gk_self` | GT `gk_anchor` | kit crop / track agree | kit separability<sup>1</sup> | cross-clip | **verdict** |
|---|---|---|---|---|---|---|---|---|---|
| **SNGS-126** | test | 7 / 2nd | **+3.99** | -48.02 | **+2.76** | 0.980 / 1.00 | 6.5 | **minority (6 of 8 say the other way)** | **GT SIDE-SWAPPED** |
| **SNGS-131** | test | 7 / 2nd | **+3.78** | -48.68 | **+3.77** | 0.973 / 1.00 | 7.1 | **minority (same group)** | **GT SIDE-SWAPPED** |
| **SNGS-197** | test | 11 / 2nd | **+5.37** | -45.95 | **+1.51** | 0.992 / 1.00 | 5.1 | **minority (5 of 6 say the other way)** | **GT SIDE-SWAPPED** |
| SNGS-130 | test | 7 / 2nd | -0.11 | -47.75 | -0.73 | 0.959 / 1.00 | 6.6 | majority | **GT CONSISTENT** |
| SNGS-116 (control) | test | 7 / 1st | -4.84 | -44.07 | -1.62 | 0.950 / 1.00 | 5.3 | majority | GT CONSISTENT |
| SNGS-129 | test | 7 / 2nd | -9.34 | -101.17 | +1.17 | 0.915 / 1.00 | 4.9 | majority | GT CONSISTENT |
| SNGS-195 | test | 11 / 2nd | +0.69 | -46.46 | +1.20 | 0.993 / 1.00 | 6.0 | majority | GT CONSISTENT (depth inverts) |
| SNGS-038 | valid | 2 / 2nd | +0.95 | -49.40 | +0.18 | 0.960 / 1.00 | 5.6 | majority | GT CONSISTENT (depth inverts) |
| **SNGS-092** | valid | 5 / 2nd | **+3.76** | -85.39 | +0.85 | 0.959 / 1.00 | 5.2 | **minority (10 of 11)** | **GT SIDE-SWAPPED** |
| **SNGS-111** | train | 6 / 2nd | **+8.81** | -95.27 | -7.54 | 0.950 / 1.00 | 3.0 | **minority (8 of 9)** | **GT SIDE-SWAPPED** |

<sup>1</sup> within-clip crop-clustering separability (centroid gap / mean within-cluster distance), a
different quantity from the per-game kit separation in §3.

Read the columns in order and the trap is visible.

* **`gk_self` is negative on all ten** — and on all 49 test clips (checked). Every annotated keeper
  in the test split stands in the goal his own label names. Instrument 1 alone would have returned
  "GT is fine everywhere", which is what it did on the first pass of this audit.
* **Kit coherence is total on all ten**: track-level agreement 1.00 everywhere, crop-level 0.915 to
  0.993 on kits separated by 3.0-7.1 units of within-cluster distance. Not one player is filed under
  the wrong team in any clip. There is no `GT MIXED` case in this pool. The contact sheets
  (`results/gsr_benchmark/gt_audit/*_contact.png`) show it directly: in SNGS-126 and SNGS-131 every
  `left` crop is the white/green kit and every `right` crop is the red kit, without exception.
* **`gk_anchor`** — which outfield group sits nearer the annotated keeper — is positive on the three
  test swaps (+2.76, +3.77, +1.51) and negative on the healthy controls SNGS-130 (-0.73) and
  SNGS-116 (-1.62). That is the internal fingerprint of the swap: the keeper is physically
  surrounded by the *opposite*-labelled outfield group. It is corroboration, not proof — `gk_anchor`
  has a GT ceiling of only 0.690 (`results/GSR_TEAMSIDE.md` §3) and is positive on two healthy clips
  here too (SNGS-129, SNGS-195).
* **The cross-clip column is what decides.**

## 3. The decisive scan: all 164 sequences

`python -m tools.gsr_gt_audit --games` over every sequence in train+valid+test. 164 scanned, 164 with
a usable two-kit signature, **0 degenerate** (no clip where both sides land on the same kit).

| game / half | n clips | kit separation | kit-of-`left` counts | minority |
|---|---|---|---|---|
| 2 / 1st, 2nd | 8, 10 | 52.0 | 8-0, 10-0 | — |
| 3 / 1st, 2nd | 10, 11 | 46.7 | 10-0, 11-0 | — |
| 4 / 1st, 2nd | 9, 9 | 8.1 | 9-0, 9-0 | — |
| 5 / 1st | 8 | 51.4 | 8-0 | — |
| **5 / 2nd** | **11** | 51.4 | **10-1** | **SNGS-092** |
| 6 / 1st | 10 | 33.9 | 10-0 | — |
| **6 / 2nd** | **9** | 33.9 | **8-1** | **SNGS-111** |
| 7 / 1st | 8 | 54.1 | 8-0 | — |
| **7 / 2nd** | **8** | 54.1 | **6-2** | **SNGS-126, SNGS-131** |
| 8 / 1st, 2nd | 9, 10 | 47.5 | 9-0, 10-0 | — |
| 9 / 1st, 2nd | 9, 11 | 43.4 | 9-0, 11-0 | — |
| 11 / 1st | 8 | 39.4 | 8-0 | — |
| **11 / 2nd** | **6** | 39.4 | **5-1** | **SNGS-197** |

**159 of 164 clips (96.9%) obey the rule; 5 do not.** Assignment margins for the five dissenters are
29.3 to 52.1 against kit separations of 33.9 to 54.1 — i.e. each is a confident assignment, not a
coin flip. Game 4 is the one weakly-separated match (kit separation 8.1, two similar kits) and it is
unanimous in both halves anyway.

**The error has a signature.** Every group flips its kit-to-side mapping between the halves, which is
correct — teams change ends. All five dissenters are **second-half** clips, and each carries its own
game's **first-half** mapping. Zero first-half clips dissent. That is what "the annotator's
team-to-side convention was not updated after the ends changed" looks like, and it is a far more
specific fingerprint than "some labels are wrong".

### Why the clip-internal instruments cannot see it

The five clips are self-consistent because the swap is applied to the *whole* clip. In SNGS-126 the
`left` label is on the white/green team and their keeper is annotated at pitch x = -48.0 (the left
goal). In SNGS-130 — same game, same half, 16 minutes later — the `left` label is on the red team and
the white/green keeper is annotated at +47.8 (the right goal). Both files are internally coherent;
they cannot both be right about which physical goal the white/green team defends.

Two escape hatches were tested and both are closed:

1. **A 180-degree rotated pitch frame** would make both consistent. It does not happen: the per-frame
   correlation between GT `bbox_pitch.x_bottom_middle` and GT `bbox_image.x_center` is **positive on
   all 49 test clips** (0.37 to 0.96, SNGS-126 = 0.63, SNGS-131 = 0.44, SNGS-197 = 0.82). GT +x is
   image-right in every clip.
2. **A reverse-angle camera** would make image-left a different physical goal. It does not happen:
   the per-frame correlation between GT `bbox_pitch.y_bottom_middle` and image foot-y is **positive
   on all 49** (0.79 to 0.97). Every clip is shot from the same touchline.

With the frame and the camera side fixed within a game, the kit-to-side mapping is a fact, and the
five dissenters contradict it.

## 4. SNGS-130 vs SNGS-195 — the discrepancy in our list against theirs

Broadcast2Pitch names 126, 131, 197. Our v6 flip set is 126, **130**, 131, 197. The two sets differ
by one clip in each direction, and each difference has a different explanation.

**SNGS-130 is not a GT error.** It sits with the majority of game 7's second half (6 of 8), its
keeper, its kits and its cross-clip mapping all agree, and its GT `meanx` is **-0.11 m** — the label
is correct, by eleven centimetres of mean pitch-x separation between the two teams over 750 frames.
That is the tightest margin in the entire test split (next tightest correct clip: SNGS-191 at
-2.30 m). Our estimator, running the same rule on our own noisy positions, landed on the other side
of a 0.11 m knife edge. **SNGS-130 is ours to lose and ours to fix**, and it is not fixable by any
better side rule — `results/GSR_TEAMSIDE.md` establishes that `meanx` is already at its own GT-oracle
ceiling; at 0.11 m of signal the clip is simply below the noise floor of the positional family, and
only the named upgrade path (a keeper-to-team link that is not kit colour) reaches it.

**SNGS-195 is the mirror image.** It is a GT-consistent clip whose depth statistic inverts by +0.69 m
(broadcast framing, exactly like SNGS-038 at +0.95 m). The GT-oracle `meanx` gets it *wrong*; our
estimator gets it *right*, by luck. That is why `GSR_TEAMSIDE.md` §7 records the oracle and our
estimator both scoring 45/49 while disagreeing about which clips: the oracle misses {126, 131, 195,
197} and we miss {126, 130, 131, 197}. This audit now separates that shared 45/49 into two different
kinds of failure — three annotation errors, one framing artefact, one noise-floor coin flip.

**A correction to our own record.** `results/GSR_TEAMSIDE.md` §3 explains SNGS-092 as a framing
artefact ("a spell where the camera frames one attacking phase and the defending team's own deep line
is off-screen"). That explanation is now retracted: SNGS-092 is a **GT side-swap**, the same error as
the three test clips. SNGS-038 keeps the framing explanation (majority-consistent). SNGS-111, listed
there as the third dev inversion, is also a GT side-swap. So of the three dev "ground-truth geometry
inversions" that act reported, **two were annotation errors and one was framing** — and the act's
central conclusion is unaffected, because a positional rule cannot reach either kind.

## 5. Implications — how much of the loss is unreachable by anyone

No action is proposed here and nothing was changed in any shipped path.

**Our side.** `results/GSR_DELEAK.md` §"Where the 4.02 goes" is the only place these clips have a
measured correct-side counterfactual (the v4-era leaky arm vs the GT-free arm):

| clip | leaky (side correct) | GT-free (side flipped) | leak-adjusted flip cost |
|---|---|---|---|
| SNGS-126 | 53.12 | 5.36 | -45.83 |
| SNGS-131 | 35.15 | 6.78 | -26.44 |
| SNGS-197 | 56.16 | 6.64 | -47.58 |

(The leaky arm also carries a roster leak worth -1.94 on the 45 non-flipped clips; the last column
subtracts it, so it is flip-cost only.) Summed over the 49-clip mean: **-2.45 GS-HOTA**. That is the
part of our loss attributable to the three annotation errors, and **no correct method can recover
it** — recovering it means predicting the wrong side on purpose. Our v6 test score is 53.09; against
a corrected ground truth it would be **~55.5**. SNGS-130 has no v6 counterfactual measured; at the
same per-clip scale it is worth a further -0.5 to -1.0, and *that* one is genuinely ours.

So of the -2.4 to -3.1 flip loss the brief costed, **roughly 2.45 of it (about 80%) is benchmark
noise and about 0.6-1.0 is real headroom.**

**Their side.** The leader's 61.48 is scored on the same 49 clips including the same three. A flipped
clip does not degrade, it annihilates — ours score 5.36 to 6.78 when flipped, and a stronger
pipeline's would score similarly, because GS-HOTA's identity is (team, jersey) and a swapped team
invalidates every association in the clip. If they lose all three, their 46 healthy clips average
**~65.1**, and the three cost them **~-3.6 GS-HOTA** — *more* than the -2.45 it costs us, because the
penalty scales with how good the clip would otherwise have been.

That inverts the comfortable reading. **This finding does not close the gap to the leader; it widens
it slightly** — 53.09 vs 61.48 is a gap of 8.39 on the published numbers and ~9.6 on annotation-
corrected ones. The honest framing is that both headline numbers understate both systems, ours by
~2.5 and theirs by ~3.6.

The alternative reading has to be stated because it cannot be excluded from here: if their pipeline
somehow *reproduces* the annotation error on those three clips, then ~3.6 of the 8.39 gap is
unreachable-by-correct-methods and the true capability gap is ~4.8. We have no evidence for that,
and the fact that they published the observation at all is weak evidence against it — a team names a
ground-truth bug in its paper when the bug is costing it points. Their repo is public
(`docs/WINNER_REPO_RECON.md`); whether their submission emits the physically-correct side on
SNGS-126/131/197 is checkable there, and that check is not part of this act.

**Dev-pool consequence.** Two of the 115 dev sequences (SNGS-092 valid, SNGS-111 train) carry the
same error. Any side rule graded on the dev pool has a hard ceiling of 113/115 = 0.9826 for that
reason alone, which is *below* the 0.98 target the N1b brief set. `GSR_TEAMSIDE.md` reported
`meanx` at 112/115 on GT; 2 of those 3 misses are now known to be unreachable, so the rule's true
error rate on well-annotated dev clips is **1 in 113**, not 3 in 115.

## 6. Negatives and limits

1. **The obvious instrument was insufficient.** `gk_self` — 113/113 on dev, the definitional rule —
   returns "consistent" on all five swapped clips, because the keeper's label is swapped along with
   everyone else's and the rule only ever compares a keeper with his own position. The first pass of
   this audit returned "GT looks fine, their claim is not reproduced" on exactly that basis. The
   cross-clip test is what changed the answer.
2. **No `GT MIXED` case exists in this pool.** Kit-level agreement is 1.00 at track level in all ten
   audited clips. Whatever "incorrect team annotations" means in their paper, it is not scattered
   per-player mislabelling — it is a clean whole-clip swap.
3. **The cross-clip test cannot grade a clip in isolation**, and it assumes an annotator error is a
   minority event within a game-half. If a whole game-half were annotated with a swapped convention,
   the majority vote would adopt the error. Nothing here detects that; it would need a physical
   anchor outside the annotation (e.g. reading the scoreboard, or the goal direction from a scoring
   event).
4. **One instrument version was wrong and is recorded.** The `a*`-only "redder side" handle produced
   a false positive on SNGS-160 (kits with identical `a*`). Fixed by per-game LAB prototypes; the
   number quoted everywhere in this document is from the fixed run.
5. **The -2.45 and ~-3.6 figures are estimates**, the first from a v4-era counterfactual and the
   second from an assumption about a system we cannot run. The *fact* established here is the
   annotation error itself; the GS-HOTA arithmetic downstream of it is arithmetic, not measurement.
6. **No submission implication.** Nothing was changed, no package rebuilt, no GT edited, no
   prediction re-scored. Emitting a deliberately wrong side on three clips to match a buggy label is
   not proposed and would not survive a legitimacy audit.

## 7. Files

- `tools/gsr_gt_audit.py` — the whole audit (`--run --games --demo`).
- `results/gsr_benchmark/gt_audit/gt_audit.json` — per-clip oracle rules, kit clustering, verdicts.
- `results/gsr_benchmark/gt_audit/gt_game_consistency.json` — the 164-sequence cross-clip scan.
- `results/gsr_benchmark/gt_audit/SNGS-*_contact.png` — 10 contact sheets, GT `left` row above GT
  `right` row, each crop captioned with its GT team, jersey, track and frame.
- `docs/WINNER_REPO_RECON.md` — the leader's public repository, recon (companion act).
