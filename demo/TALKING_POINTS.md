# Demo pack — talking points (10 min)

Open: `demo/demo_pack.html` in any browser (offline, no internet needed). Full-screen it.

## What to open, in order
1. `demo/demo_pack.html` — the single page. Everything hangs off it.
2. Section 02 — press play on the identity video (2:51). Let it run ~60-90s, skip to the
   identity segment (~half way) if short on time.
3. Section 03 — click one report link (Liverpool) live in front of the professor.
4. Sections 04-06 — talk over the validation table, limits, and roadmap.

## Ten bullets to say
1. Thesis: uncertainty-aware tactical inference from broadcast video — we measure what the
   broadcast supports and prove where it stops. One TV camera in, gated tactical data out.
2. Pipeline (Sec 01): six gated stages. Nothing downstream ships unless the stage above it
   validated against ground truth — that gating is the whole contribution.
3. We SEE players (Sec 02): this is OUR CV output from the broadcast feed, not vendor
   tracking. Detect, track, calibrate to a 105x68m pitch, then name by shirt number.
4. Honesty on screen: the video states its own failures — when geometry drops out the
   top-down panel goes dark rather than guessing.
5. Verified reads (montages): identity precision is audited by looking at EVERY gated
   jersey crop, not a sample — that grid IS the audit.
6. We make reports AUTOMATICALLY (Sec 03): open the live Liverpool pack. A numeric
   guardrail refuses any figure not backed by a stored fact.
7. Validation (Sec 04): BAS 10/12 in a frozen band, E2E goals 14/16 with all halftime
   splits correct, GS-HOTA externally graded 14.76 -> 22.85, WP calibration ECE 0.042.
8. We KNOW the limits (Sec 05): ~11.8 of 22 players visible per frame — a broadcast never
   shows the whole pitch, and we never invent the rest.
9. The retraction is the method: two team-flips sat in the corpus and were caught by our
   OWN automated screen, then retracted loudly. verify_team_mapping.py now passes 12/12.
10. The ask (Sec 06): December roadmap — pack demo, forecast v0, B4 imputation behind
    uncertainty gates to lift the visibility floor honestly.

## Things Sid should know BEFORE presenting
- Identity video is Man Utd v BRIGHTON (Jul 16 render). It is the ONLY match with a
  named-tracks parquet, so it could not be regenerated on Liverpool/Southampton without
  running the full identity pipeline (hours of GPU). It is current and correct — use it.
- Reports linked are LIVERPOOL + BRIGHTON v2 (both Jul 19). Deliberate: the Southampton v2
  report is dated Jul 20 but the Jul 23 retraction FIXED Southampton's team-mapping flip
  and the HTML was NOT regenerated — showing it would display the retracted claim. Do NOT
  open the Southampton report live. Liverpool and Brighton were unaffected by both flips.
- The two montages are Southampton (yellow away) and Liverpool (red) audit grids — safe to
  show as images; they are jersey-crop grids, not the team-shape claims that were retracted.
- All links are relative and resolve from demo/demo_pack.html. If you move the demo/ folder,
  keep it inside the repo so ../results/... still resolves.
- Video autoplay is off by design (browsers block it); click play.
