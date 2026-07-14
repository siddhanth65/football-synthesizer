# France v Senegal -- CV-primary tactical read

*Body written entirely from our computer-vision metrics (the [CV] sections). FIFA PMSR appears only in the validation appendix [FIFA], as an oracle we check ourselves against -- never as content.*

**Ball-evidence gate:** post-link coverage 46% (>= 40%), pass-recall proxy 55.0% (>= 50%) -> **PASS**.  Body guardrail precision: 100% (77/77 numbers CV-backed).

*Gate inputs read from `outputs/eval/france_senegal_ball_eval.json` (oracle: FIFA PMSR).*

## [CV] How France set up

Our shape classifier reads France in a **3-5-2** (nominal 4-2-3-1). The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **30 m** up the pitch (the raw broadcast line is censoring-inflated; the de-biased value is validated to within about **5 m** of FIFA's per-phase lines); they build from a base line around **52 m**; the block is **36 m** wide and **19 m** deep; nearest-team-mate compactness holds near **12.6** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **47%** -- the dominant channel
- centre **31%**
- wings **22%** -- comparatively thin, a structural handle for opponents

## [CV] In possession

France move as a unit -- velocity synchrony **71%** on a 0-1 scale (this is a position-only measure and always renders).

By our own phase classifier France spend **48%** of in-possession time in build-up, **32%** in progression and **20%** in the final third -- a side that wants to carry the ball forward, not launch it.

With the ball, goalward directness (verticality) measures **10%** of maximum; tracking caught **86 defensive-line breaks**; ball progression was worth **6.7 xG-equivalent** expected threat (xT) over **3598** tracked advances.

Passing volume in the usable ball track: **680 passes** at a short mean of **7.2 m**, with a PPDA proxy of **0.3** opponent build-up passes per defensive action (801 pressures tracked) -- an aggressive, high-tempo profile.

## [CV] Out of possession

France defend from a high starting point -- the visibility-corrected line at **30 m** is a front-foot posture (position-only, always rendered).

On the ball-tracked chunks their pressing intensity -- a 0-1 time-to-intercept index -- runs at **48%** of maximum.

Their counterpress decays predictably: after a loss France regain the ball within 3 s **70%** of the time, **81%** within 5 s and **86%** within 8 s -- the curve flattens after the 5 s window.

They commit hard to it -- counterpress rate **91%** of losses at a mean re-engagement of **0.8 s** -- but convert only **130 high regains** from **677 losses**, so much of the pressure delays rather than wins the ball high.

## [CV] Counter-structure seams -- how to play against France

Each seam pairs a concrete idea with the exact CV number behind it and a one-line condition that would close it. Seams resting on ball-tracked families abstain when the match fails the evidence gate.

- **Seam.** Retain through the first press and play forward within about 5 s. France recover **81%** of their losses inside 5 s but the curve flattens to **86%** by 8 s -- beating the initial counterpress buys a clean progression window.
  - *Backing:* counterpress regain curve 70% (3 s) -> 81% (5 s) -> 86% (8 s), over 677 tracked losses.
  - *Closes if:* France's 5 s regain rate rises well above 81% -- a counterpress that no longer plateaus.

- **Seam.** Attack and switch into the wide lanes. France concentrate centrally and in the half-spaces (**47%** half-space, **31%** centre) with only **22%** in the wings -- isolating the full-backs stretches a compact block.
  - *Backing:* lane occupation: half-space 47%, centre 31%, wing 22% (France, whole match, position-only).
  - *Closes if:* France's wing occupation climbs well above 22% toward their central load.

- **Seam.** Target the space behind the defensive line. Visibility-corrected, France's line sits **30 m** up the pitch behind a base line near 52 m; early, direct balls in behind exploit that depth before the block resets.
  - *Backing:* de-biased defensive line 30 m (validated to within about 5 m of FIFA per-phase; position-only).
  - *Closes if:* France drop the line well below 30 m toward their own half.

- **Seam.** Commit to the counter the instant you win it back. France pour into the counterpress (**91%** rate, **0.8 s** mean re-engagement) yet convert only **130** of **677** losses into high regains -- survive first contact and the pitch opens.
  - *Backing:* counterpress rate 91%, mean recovery 0.8 s, 130 high regains vs 677 losses.
  - *Closes if:* France's high regains rise toward a third of their losses -- a counterpress that wins the ball high rather than just delaying.

- **Seam.** Deny territory with a compact mid-block, not a deep retreat. Our pooled opponent model (pooled 8-obs model -- directional) shows France push further up the more you drop off; against this opponent's mid-depth line France's attacking-third share held at **24%**.
  - *Backing:* attacking-third share 24% (C5 opponent model, pooled 8-obs -- directional).
  - *Closes if:* France's attacking-third share stays near 24% no matter how deep you defend -- i.e. the opponent term flattens.

## [FIFA] Validation against FIFA PMSR -- oracle only, not used above

None of the numbers below feed the analysis above. FIFA's official PMSR is used purely to check our CV output where the two measure the same thing.

*Possession*

| metric | CV (ours) | FIFA | note |
| --- | --- | --- | --- |
| possession share | 48% (space-control proxy) | 49.4% | proxy, not a like-for-like count |

*In-possession phase split (FIFA normalised over the same three buckets). Validated C6 mean-abs-error 9.7 pp.*

| phase | CV (ours) | FIFA |  |
| --- | --- | --- | --- |
| build-up | 48% | 64% |  |
| progression | 32% | 22% |  |
| final third | 20% | 14% |  |

*Defensive line (pooled). FIFA per-phase blocks: high_block 49 m, mid_block 38 m, low_block 19 m.*

| metric | CV (ours) | FIFA | error |
| --- | --- | --- | --- |
| de-biased defensive line | 30.0 m | 35.3 m (block mean) | 5.3 m (pooled ~5.2 m) |

*Pass volume in the usable ball track vs FIFA total.*

| metric | CV (ours) | FIFA | recall proxy |
| --- | --- | --- | --- |
| passes | 680 | 584 | 55.0% |

---
*football-synthesizer report v2. Body = CV metrics + roster names. Appendix = FIFA PMSR oracle. C5 tendencies are a pooled 8-observation directional signal.*