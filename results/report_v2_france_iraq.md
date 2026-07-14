# France v Iraq -- CV-primary tactical read

*Body written entirely from our computer-vision metrics (the [CV] sections). FIFA PMSR appears only in the validation appendix [FIFA], as an oracle we check ourselves against -- never as content.*

**Ball-evidence gate:** post-link coverage 36% (>= 40%), pass-recall proxy 22.8% (>= 50%) -> **ABSTAIN on ball families**.  Body guardrail precision: 100% (38/38 numbers CV-backed).

*Gate inputs read from `outputs/eval/france_iraq_ball_eval.json` (oracle: FIFA PMSR).*

## [CV] How France set up

Our shape classifier reads France in a **3-5-2** (nominal 4-3-3). The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **36 m** up the pitch (the raw broadcast line is censoring-inflated; the de-biased value is validated to within about **5 m** of FIFA's per-phase lines); they build from a base line around **58 m**; the block is **34 m** wide and **20 m** deep; nearest-team-mate compactness holds near **13.2** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **48%** -- the dominant channel
- centre **29%**
- wings **23%** -- comparatively thin, a structural handle for opponents

## [CV] In possession

France move as a unit -- velocity synchrony **69%** on a 0-1 scale (this is a position-only measure and always renders).

> **Withheld.** Ball-dependent possession detail (tempo, PPDA, ball-xT, phase split, verticality): insufficient ball-track evidence (coverage 36%, recall proxy 22.8%) -- withheld.

## [CV] Out of possession

France defend from a high starting point -- the visibility-corrected line at **36 m** is a front-foot posture (position-only, always rendered).

> **Withheld.** Ball-dependent pressing detail (pressing intensity, counterpress decay curve, regains): insufficient ball-track evidence (coverage 36%, recall proxy 22.8%) -- withheld.

## [CV] Counter-structure seams -- how to play against France

Each seam pairs a concrete idea with the exact CV number behind it and a one-line condition that would close it. Seams resting on ball-tracked families abstain when the match fails the evidence gate.

> **Withheld.** Seam withheld (counterpress-timing window, ball-derived): insufficient ball-track evidence (coverage 36%, recall proxy 22.8%) -- withheld.

- **Seam.** Attack and switch into the wide lanes. France concentrate centrally and in the half-spaces (**48%** half-space, **29%** centre) with only **23%** in the wings -- isolating the full-backs stretches a compact block.
  - *Backing:* lane occupation: half-space 48%, centre 29%, wing 23% (France, whole match, position-only).
  - *Closes if:* France's wing occupation climbs well above 23% toward their central load.

- **Seam.** Target the space behind the defensive line. Visibility-corrected, France's line sits **36 m** up the pitch behind a base line near 58 m; early, direct balls in behind exploit that depth before the block resets.
  - *Backing:* de-biased defensive line 36 m (validated to within about 5 m of FIFA per-phase; position-only).
  - *Closes if:* France drop the line well below 36 m toward their own half.

> **Withheld.** Seam withheld (transition after beating the press, ball-derived): insufficient ball-track evidence (coverage 36%, recall proxy 22.8%) -- withheld.

- **Seam.** Deny territory with a compact mid-block, not a deep retreat. Our pooled opponent model (pooled 8-obs model -- directional) shows France push further up the more you drop off; against this opponent's mid-depth line France's attacking-third share held at **32%**.
  - *Backing:* attacking-third share 32% (C5 opponent model, pooled 8-obs -- directional).
  - *Closes if:* France's attacking-third share stays near 32% no matter how deep you defend -- i.e. the opponent term flattens.

## [FIFA] Validation against FIFA PMSR -- oracle only, not used above

None of the numbers below feed the analysis above. FIFA's official PMSR is used purely to check our CV output where the two measure the same thing.

*Possession*

| metric | CV (ours) | FIFA | note |
| --- | --- | --- | --- |
| possession share | 49% (space-control proxy) | 49.0% | proxy, not a like-for-like count |

*In-possession phase split (FIFA normalised over the same three buckets). Validated C6 mean-abs-error 16.2 pp.*

| phase | CV (ours) | FIFA |  |
| --- | --- | --- | --- |
| build-up | 32% | 54% |  |
| progression | 33% | 18% |  |
| final third | 35% | 28% |  |

*Defensive line (pooled). FIFA per-phase blocks: high_block 51 m, mid_block 38 m, low_block 18 m.*

| metric | CV (ours) | FIFA | error |
| --- | --- | --- | --- |
| de-biased defensive line | 36.4 m | 35.7 m (block mean) | 0.7 m (pooled ~5.2 m) |

*Pass volume in the usable ball track vs FIFA total.*

| metric | CV (ours) | FIFA | recall proxy |
| --- | --- | --- | --- |
| passes | 297 | 604 | 22.8% |

---
*football-synthesizer report v2. Body = CV metrics + roster names. Appendix = FIFA PMSR oracle. C5 tendencies are a pooled 8-observation directional signal.*