"""Turn our pipeline's artifacts into the prediction table ``eval/footpass_score.py`` grades.

The last link of the from-pixels chain. For every labelled FOOTPASS event frame it answers the one
question PCBAS asks -- *which player did that* -- using only pipeline outputs:

1. map the GT **global** frame to ``(chunk, local grid frame)`` with the measured chunk offsets
   (``tools/footpass_prep.chunk_game``) and the measured annotation-vs-video frame shift
   (``tools/footpass_prep.align_check``);
2. the actor is the tracked player nearest the ball at that frame -- the same carrier rule
   ``tools/event_ledger.py`` uses, reused verbatim (Viterbi ball track, aligned detections filling
   gaps, abstain beyond :data:`tools.event_ledger.TEAM_MAX_M`);
3. its name comes from the Stage-2 identity solver
   (``outputs/identity/solver/<match>_solver_names.parquet``), whose FOOTPASS "names" are
   ``T<team>#<shirt>``, and the dial is the solver's own posterior.

Deliberately **not** tuned: the contact frame is the grid frame nearest the label, with no search
window. ``tools/event_ledger.py`` biases its window before the BAS peak because that detector fires
mid-flight; the FOOTPASS label is already the action instant, and a window scanned for "the frame
where somebody is closest to the ball" would be a knob fitted against the thing being graded.

CLI::

    python -m tools.footpass_predict --games game_18,game_24,game_47
    python -m eval.footpass_score --preds outputs/footpass/preds.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from eval.footpass_score import VAL_GAMES, load_events
from tools import event_ledger as el
from tools.footpass_prep import LINEUP, RESULTS_DIR, chunk_offsets, match_id

logger = logging.getLogger("footpass_predict")

#: Where the identity solver writes its per-track names + posteriors.
NAMES_DIR = Path("outputs/identity/solver")
#: Default destination for the prediction table.
PREDS = Path("outputs/footpass/preds.parquet")
#: FOOTPASS identity key emitted by ``tools/footpass_prep.build_lineup``.
_NAME_RE = re.compile(r"^T(\d+)#(\d+)$")


def _shirt_of(name: str) -> int:
    """Shirt number encoded in a FOOTPASS identity key (``T1#7`` -> 7); ``-1`` if unparseable."""
    m = _NAME_RE.match(str(name))
    return int(m.group(2)) if m else -1


def measured_frame_shift(game: str, *, results_dir: Path = RESULTS_DIR) -> int:
    """Video frame to read for a given annotation frame: ``video = gt + shift``.

    **Measured 2026-07-29: the shift is 0.** Annotation frame ``k`` is mp4 frame ``k``, exact to
    within one frame, proven three ways (``results/FOOTPASS_E2E_PLAN.md``): GT ROI boxes overlaid on
    the decoded frame land on the players; a non-grass-inside-box statistic peaks at shift
    ``{-1, 0, +1}`` and collapses by 40% at ``+-5`` across both halves of all three games; and the
    camera-cut trains agree at a single-frame peak (offset ``+1``/``0``/``+1``, 147-234 exact hits
    against a runner-up of 3-4).

    The residual +-1 is a detector-convention difference, not a data offset, and is absorbed by the
    sampling grid (this run extracted at stride 2). This returns 0 and only *warns* if ``align.json`` reports a cut offset
    larger than one frame -- which would mean something changed and the proof must be re-read.
    """
    path = results_dir / "align.json"
    if not path.exists():
        logger.warning("%s missing -- RUN: python -m tools.footpass_prep --stage align", path)
        return 0
    k = int(json.loads(path.read_text(encoding="utf-8"))[game]["best_offset"])
    if abs(k) > 1:
        logger.warning("%s: cut-train offset %+d exceeds the proven +-1 -- re-check alignment",
                       game, k)
    return 0


def track_teams(df: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Majority team per ``(chunk, track_id)``.

    The aligned table's ``team`` is a per-detection kit call and flips frame to frame; the per-track
    majority is the stable label (the pattern of ``tools.carrier_attribution_probe.track_team``).
    """
    g = (df[df["role"].isin(["player", "goalkeeper"])]
         .groupby(["chunk", "track_id"])["team"].agg(lambda s: int(s.mode().iloc[0])))
    return {(str(c), int(t)): int(v) for (c, t), v in g.items()}


def solver_names(game: str, names_dir: Path) -> dict[tuple[str, int], tuple[int, float]]:
    """Arm (a): the Stage-2 MILP's assignment + its posterior, per ``(chunk, track_id)``."""
    npath = names_dir / f"{match_id(game)}_solver_names.parquet"
    if not npath.exists():
        logger.warning("%s missing -- every event will be unanswered", npath)
        return {}
    nm = pd.read_parquet(npath)
    conf = nm["confidence"] if "confidence" in nm else pd.Series(1.0, index=nm.index)
    return {(str(c), int(t)): (_shirt_of(p), float(k))
            for c, t, p, k in zip(nm["chunk"], nm["track_id"], nm["player_name"], conf)}


def greedy_names(game: str, *, variant: str = "",
                 min_votes: int | None = None) -> dict[tuple[str, int], tuple[int, float]]:
    """Arm (b): the Stage-1 greedy rule fed per-crop reads (``percrop_names``).

    The rule `results/OCR_REALMATCH.md` §amendment and `results/RETEST_ABSTENTION.md` register:
    the frozen GTA connector at ``tau=0.040`` (within-chunk, splitter off), then unanimous
    propagation of the *directly read* names inside each merge group -- fill only unnamed members,
    and on ANY disagreement leave the whole group untouched. `tools/gta_carrier.py` is imported, not
    modified. A read becomes a name only if its number exists on the roster of the team the track
    was assigned, using the measured team map.

    Every confidence is **1.0 by construction** -- this arm has no dial, which is exactly the
    property `RETEST_ABSTENTION.md` §greedy-vs-solver flags. Its frontier is therefore a single
    point, and is reported as such rather than dressed up as a curve.

    Args:
        game: FOOTPASS game id.
        variant: Per-crop OCR cache variant (crop geometry) to consume.
        min_votes: Exploratory override of the frozen rule's vote bar.
    """
    from tools.gta_carrier import merge_match, propagate_names  # noqa: PLC0415
    from tools.identity_match import PARTITION, percrop_reads  # noqa: PLC0415

    mid = match_id(game)
    lu = pd.read_parquet(Path(str(LINEUP).format(match=mid)))
    tmap = {int(r.fp_team): int(r.team) for r in lu.itertuples()}          # fp team -> our team id
    roster = {(int(r.fp_team), int(r.shirt)) for r in lu.itertuples()}
    teams = track_teams(registry.get(mid).load_aligned())

    base: dict[tuple[str, int], str] = {}
    for chunk, by_track in percrop_reads(mid, variant=variant, min_votes=min_votes).items():
        for tid, votes in by_track.items():
            num = Counter(n for n, _c in votes).most_common(1)[0][0]
            our = teams.get((str(chunk), int(tid)), -1)
            for fp, ours in tmap.items():
                if ours == our and (fp, int(num)) in roster:
                    base[(str(chunk), int(tid))] = f"T{fp}#{int(num)}"
                    break
    merged, mstat = merge_match(mid, PARTITION)
    names, nstat = propagate_names(mid, merged, base=base)
    logger.info("%s greedy: %d direct reads -> %d named tracks (%d propagated, %d groups "
                "disagreed); fragments %d -> %d", game, len(base), len(names),
                nstat["n_names_propagated"], nstat["n_groups_name_disagree"],
                mstat["n_fragments_before"], mstat["n_fragments_after"])
    return {k: (_shirt_of(v), 1.0) for k, v in names.items()}


def predict_game(game: str, *, names_dir: Path = NAMES_DIR, frame_shift: int | None = None,
                 carrier_max_m: float = el.TEAM_MAX_M, arm: str = "solver",
                 percrop_variant: str = "", min_votes: int | None = None) -> pd.DataFrame:
    """One prediction row per labelled event of ``game``.

    Args:
        game: FOOTPASS game id.
        names_dir: Where ``<match>_solver_names.parquet`` lives.
        frame_shift: ``video = gt + shift``; ``None`` reads the measured value.
        carrier_max_m: Abstain when the nearest player to the ball is farther than this.
        arm: ``"solver"`` (frozen Stage-2 MILP + its posterior) or ``"greedy"``
            (Stage-1 rule on per-crop reads; every confidence is 1.0).
        percrop_variant: Per-crop OCR cache variant, used by the ``greedy`` arm.
        min_votes: Exploratory vote-bar override, used by the ``greedy`` arm.

    Returns:
        ``game, frame, pred_team, pred_shirt, conf`` (the scorer's schema) plus diagnostics
        ``chunk, local_frame, track_id, carrier_dist_m, ball_found, n_players``.
    """
    mid = match_id(game)
    match = registry.get(mid)
    shift = measured_frame_shift(game) if frame_shift is None else int(frame_shift)
    df = match.load_aligned()
    offsets = chunk_offsets(game)
    teams = track_teams(df)
    named = (greedy_names(game, variant=percrop_variant, min_votes=min_votes) if arm == "greedy"
             else solver_names(game, names_dir))

    events = load_events((game,))
    ends = {c: offsets[c] + int(df[df["chunk"] == c]["frame"].max() or 0) for c in offsets}
    rows: list[dict] = []
    for chunk in sorted(offsets):
        dfc = df[df["chunk"] == chunk]
        if dfc.empty:
            continue
        f0 = offsets[chunk]
        sel = events[(events["frame"] + shift >= f0) & (events["frame"] + shift <= ends[chunk])]
        if sel.empty:
            continue
        pf, by = el._players_by_frame(dfc)
        bf, bxy = el._ball_track(match, chunk, dfc)
        for gt_frame in sel["frame"].astype(int):
            # The sampling grid is measured, not assumed. ``el.bas_to_grid_frame`` rounds to
            # ``el.STEP`` (5), but this run extracted at stride 2 (FOOTPASS_E2E_PLAN A.6), so that
            # rounding would evaluate the carrier up to 2 frames further from the label than
            # necessary. ``_nearest_row`` already snaps to the nearest frame that actually exists,
            # whatever the stride, so the label's own frame is the right target.
            g = int(gt_frame) + shift - f0
            row = {"game": game, "frame": int(gt_frame), "chunk": chunk, "local_frame": g,
                   "pred_team": -1, "pred_shirt": -1, "conf": np.nan, "track_id": -1,
                   "carrier_dist_m": np.nan, "ball_found": False, "n_players": 0}
            pi = el._nearest_row(pf, g, el.BALL_GAP_FR)
            bi = el._nearest_row(bf, g, el.BALL_GAP_FR)
            if pi is not None and bi is not None:
                pos, tms, tracks = by[int(pf[pi])]
                row["n_players"] = int(len(tracks))
                row["ball_found"] = True
                hit = el.nearest_carrier(pos, tms, tracks, (bxy[bi][0], bxy[bi][1]))
                if hit is not None and hit[2] <= carrier_max_m:
                    _t, tid, dist = hit
                    key = (chunk, int(tid))
                    row["track_id"] = int(tid)
                    row["carrier_dist_m"] = round(float(dist), 2)
                    row["pred_team"] = int(teams.get(key, -1))
                    shirt, conf = named.get(key, (-1, float("nan")))
                    row["pred_shirt"] = int(shirt)
                    row["conf"] = float(conf)
            rows.append(row)
    out = pd.DataFrame(rows)
    logger.info("%s: %d events, ball+players found %d, carrier within %.1fm %d, named %d",
                game, len(out), int(out["ball_found"].sum()), carrier_max_m,
                int((out["track_id"] >= 0).sum()), int((out["pred_shirt"] > 0).sum()))
    return out


def _selftest() -> None:
    """Pure-seam checks that need no artifacts."""
    assert _shirt_of("T1#7") == 7 and _shirt_of("T2#99") == 99 and _shirt_of("Bruno") == -1
    # the carrier rule and grid snap we depend on, pinned here so a change upstream is caught.
    # Stride-agnostic: on a stride-2 grid frame 5 must snap to frame 4, not to a multiple of 5.
    assert el._nearest_row(np.array([0, 2, 4, 6]), 5, 25) == 2
    assert el._nearest_row(np.array([0, 5, 10]), 7, 25) == 1
    hit = el.nearest_carrier(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([0, 1]),
                             np.array([5, 6]), (9.0, 0.0))
    assert hit == (1, 6, 1.0), hit
    print("footpass_predict selftest ok")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default=",".join(VAL_GAMES))
    ap.add_argument("--names-dir", type=Path, default=NAMES_DIR)
    ap.add_argument("--frame-shift", type=int, default=None)
    ap.add_argument("--carrier-max-m", type=float, default=el.TEAM_MAX_M)
    ap.add_argument("--arm", default="solver", choices=["solver", "greedy"])
    ap.add_argument("--percrop-variant", default="", help="per-crop cache variant, e.g. _w125")
    ap.add_argument("--min-votes", type=int, default=None,
                    help="EXPLORATORY override of the frozen rule's vote bar (greedy arm)")
    ap.add_argument("--out", type=Path, default=PREDS)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    frames = [predict_game(g, names_dir=args.names_dir, frame_shift=args.frame_shift,
                           carrier_max_m=args.carrier_max_m, arm=args.arm,
                           percrop_variant=args.percrop_variant, min_votes=args.min_votes)
              for g in args.games.split(",")]
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
