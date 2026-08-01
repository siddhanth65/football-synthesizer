"""Prepare the three FOOTPASS VAL games for our from-pixels pipeline (CPU only).

Everything the end-to-end run needs before a single GPU second is spent:

* ``probe``   -- ffprobe each mp4 and check it against the annotations' frame span.
* ``align``   -- **prove** the annotation frame index equals the mp4 frame index, by matching the
  camera-cut train of the video (ffmpeg ``scene`` detector) against the camera-cut train implied by
  the annotations' per-frame visible-player count. A scoring harness with a frame misalignment is
  worthless, so this runs before anything else and emits the measured offset.
* ``chunk``   -- one ffmpeg stream-copy segmentation per game into
  ``matches/footpass_<game>/h{1,2}/chunk_NNN.mp4``, plus the exact per-chunk **global frame offset**
  read from ffmpeg's own segment list (cuts snap to keyframes, so the offsets are measured, never
  assumed).
* ``lineup``  -- the per-half roster the identity solver consumes, derived from the annotations'
  distinct ``(player_id, shirt_number, role_id)`` triples. FOOTPASS is anonymised, so a player's
  "name" is ``T<team>#<shirt>``, which is unique within a game and is exactly the identity PCBAS
  scores.
* ``teammap`` -- resolve the one remaining bit (our kit-anchored team 0/1 vs FOOTPASS team 1/2) from
  the per-crop OCR reads against the two rosters, and rewrite the lineups if it is flipped.

The video lives under ``data/footpass/video/`` (gitignored); chunks under ``matches/`` (also
gitignored -- ``*.mp4``). Nothing here touches the GPU.

CLI (every stage CPU-only except ``extract``, which is the pipeline's GPU entry point)::

    python -m tools.footpass_prep --stage probe
    python -m tools.footpass_prep --stage align            # the frame-alignment proof
    python -m tools.footpass_prep --stage chunk            # segment + measure offsets
    python -m tools.footpass_prep --stage offsets          # re-measure offsets, no re-segmenting
    python -m tools.footpass_prep --stage verify           # frame accounting across the joins
    python -m tools.footpass_prep --stage lineup
    python -m tools.footpass_prep --stage extract          # *** GPU *** detect + track + calibrate
    python -m tools.footpass_prep --stage teammap          # after the OCR pass exists
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry

logger = logging.getLogger("footpass_prep")

#: The three VAL games (the only split whose labels we hold).
GAMES = ("game_18", "game_24", "game_47")
#: Extracted full-match videos (one file spans both halves).
VIDEO_DIR = Path("data/footpass/video")
#: Digests written by ``tools/footpass_digest.py``.
DIGEST_DIR = Path("data/footpass/digest")
#: Chunk root the pipeline reads (``core.registry.VIDEO_ROOT``).
MATCHES_ROOT = Path("matches")
#: Roster artifact ``tools/identity_match.py`` loads.
LINEUP = Path("outputs/identity/{match}_lineup_assign.parquet")
#: Where the measured per-chunk global frame offsets are written.
OFFSETS = Path("outputs/{match}/chunk_offsets.json")
#: Where prep reports land.
RESULTS_DIR = Path("results/footpass")
#: Broadcast frame rate (verified by ffprobe on all three files).
FPS = 25.0
#: Chunk length in seconds -- the pipeline's convention (``tools/chunk_video.py``).
CHUNK_S = 600
#: ffmpeg ``scene`` score above which a frame is called a camera cut.
SCENE_THRESHOLD = 0.25
#: Minimum jump in the annotations' visible-player count to call a frame a camera cut.
MIN_VISIBLE_JUMP = 6
#: Offsets searched by the alignment proof (frames).
ALIGN_SEARCH = 60
#: FOOTPASS ``role_id`` of the goalkeeper (verified: exactly two per half, one per team).
GK_ROLE_ID = 1

_META_RE = re.compile(r"pts_time:([0-9.\-]+)")


def match_id(game: str) -> str:
    """Registry match id for a FOOTPASS game (``game_18`` -> ``footpass_game_18``)."""
    return f"footpass_{game}"


def ffbin(name: str) -> str:
    """Locate ``ffmpeg``/``ffprobe`` (PATH, then the user's Desktop build)."""
    onpath = shutil.which(name)
    if onpath:
        return onpath
    for p in Path.home().glob(f"OneDrive/Desktop/ffmpeg-*/bin/{name}.exe"):
        return str(p)
    raise FileNotFoundError(f"{name} not found on PATH")


def video_path(game: str) -> Path:
    """Full-match mp4 for a game."""
    return VIDEO_DIR / f"{game}.mp4"


def load_digest(game: str, half: str) -> dict:
    """One half's digest arrays (``half`` is ``H1``/``H2``)."""
    return dict(np.load(DIGEST_DIR / f"val_{game}_{half}.npz"))


# === probe =======================================================================================
def probe(game: str) -> dict:
    """Container facts of one game's mp4 against the annotations' frame span.

    Args:
        game: FOOTPASS game id.

    Returns:
        ``nb_frames``, ``fps``, ``start_time``, the annotated frame range per half and the tail of
        unannotated frames at the end of the file.
    """
    out = subprocess.run(
        [ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames,r_frame_rate,start_time,width,height", "-of", "json", str(video_path(game))],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (float(x) for x in st["r_frame_rate"].split("/"))
    halves = {}
    for half in ("H1", "H2"):
        d = load_digest(game, half)
        halves[half.lower()] = [int(d["window"][:, 0].min()), int(d["window"][:, 1].max())]
    last = max(v[1] for v in halves.values())
    return {"game": game, "nb_frames": int(st["nb_frames"]), "fps": num / den,
            "start_time": float(st.get("start_time", 0.0)),
            "size": [int(st["width"]), int(st["height"])],
            "annotated_frames": halves, "last_annotated_frame": last,
            "unannotated_tail_frames": int(st["nb_frames"]) - 1 - last}


# === alignment proof =============================================================================
def video_cut_frames(game: str, *, threshold: float = SCENE_THRESHOLD,
                     duration: float | None = None) -> np.ndarray:
    """Camera-cut frame indices of the mp4, via ffmpeg's ``scene`` detector (CPU, ~25x realtime).

    Args:
        game: FOOTPASS game id.
        threshold: ``scene`` score above which a frame starts a new shot.
        duration: Analyse only the first N seconds (smoke tests); ``None`` = whole file.

    Returns:
        Ascending int array of 0-based frame indices, each the FIRST frame of a new shot.
    """
    start = probe(game)["start_time"]
    cmd = [ffbin("ffmpeg"), "-hide_banner", "-v", "error"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(video_path(game)), "-an", "-vf",
            f"scale=192:108,select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    times = [float(m.group(1)) for m in _META_RE.finditer(res.stdout)]
    return np.unique(np.rint((np.asarray(times) - start) * FPS).astype(np.int64))


def annotation_cut_frames(game: str, *, min_jump: int = MIN_VISIBLE_JUMP) -> np.ndarray:
    """Camera-cut frames implied by the annotations' per-frame visible-player count.

    ``roi_*`` is NaN exactly when a player is off-screen, so the number of players carrying an ROI
    steps sharply at a camera cut. The digest stores this as visibility runs.

    Args:
        game: FOOTPASS game id.
        min_jump: Minimum change in the visible count to call a frame a cut.

    Returns:
        Ascending int array of annotation frame indices at which the count jumped.
    """
    cuts: list[np.ndarray] = []
    for half in ("H1", "H2"):
        d = load_digest(game, half)
        runs = d["runs"]
        last = int(d["window"][:, 1].max())
        delta = np.zeros(last + 2, np.int32)
        for _p, s, e in runs:
            delta[int(s)] += 1
            delta[int(e) + 1] -= 1
        cnt = np.cumsum(delta)
        jump = np.abs(np.diff(cnt.astype(np.int64)))
        cuts.append(np.flatnonzero(jump >= min_jump) + 1)
    return np.unique(np.concatenate(cuts))


def align_offset(video_cuts: np.ndarray, ann_cuts: np.ndarray, *, search: int = ALIGN_SEARCH
                 ) -> dict:
    """Offset ``k`` maximising exact matches of ``video_cut + k`` against the annotation cuts (pure).

    Exact (zero-tolerance) matching is deliberate: a peak one frame wide is a proof, a peak smeared
    over a tolerance window is not.

    Args:
        video_cuts: Frame indices of camera cuts detected in the video.
        ann_cuts: Frame indices of cuts implied by the annotations.
        search: Half-width of the offset search, in frames.

    Returns:
        ``best_offset``, ``n_matched``, ``n_video_cuts``, ``n_ann_cuts``, ``runner_up`` (the second
        best offset's match count) and the full ``histogram`` over the searched offsets.
    """
    ann = set(int(x) for x in ann_cuts)
    hist = {k: sum(1 for v in video_cuts if int(v) + k in ann) for k in range(-search, search + 1)}
    order = sorted(hist.items(), key=lambda kv: (-kv[1], abs(kv[0])))
    best, n = order[0]
    runner = next((c for k, c in order if abs(k - best) > 1), 0)
    return {"best_offset": int(best), "n_matched": int(n), "n_video_cuts": int(video_cuts.size),
            "n_ann_cuts": int(ann_cuts.size), "runner_up": int(runner),
            "histogram": {str(k): v for k, v in hist.items() if v}}


def align_check(game: str, *, duration: float | None = None) -> dict:
    """Run the full frame-alignment proof for one game."""
    v = video_cut_frames(game, duration=duration)
    a = annotation_cut_frames(game)
    res = align_offset(v, a)
    res["game"] = game
    res["duration_analysed_s"] = duration
    logger.info("%s: offset %+d matched %d/%d video cuts (runner-up %d)", game,
                res["best_offset"], res["n_matched"], res["n_video_cuts"], res["runner_up"])
    return res


# === chunking ====================================================================================
def half_boundary_frame(game: str) -> int:
    """Midpoint of the unannotated gap between the two halves (frames).

    The FOOTPASS mp4 is H1 and H2 concatenated with half-time removed, and the annotation frame
    index runs continuously across the join, so the boundary is the middle of the gap between
    ``H1``'s last annotated frame and ``H2``'s first.
    """
    h1 = load_digest(game, "H1")["window"][:, 1].max()
    h2 = load_digest(game, "H2")["window"][:, 0].min()
    return int((int(h1) + int(h2)) // 2)


def chunk_game(game: str, *, seconds: int = CHUNK_S, overwrite: bool = False) -> dict:
    """Stream-copy segment one game into per-half chunks and record measured frame offsets.

    Cuts are forced at every ``seconds`` boundary **and** at the half boundary; with ``-c copy``
    ffmpeg snaps each to the next keyframe, so the actual start times are read back from its own
    segment list rather than assumed. ``chunk_NNN`` keeps the GLOBAL segment index, so chunk keys
    stay unique across halves and map 1:1 onto the offset table.

    Args:
        game: FOOTPASS game id.
        seconds: Nominal chunk length.
        overwrite: Re-segment even if chunks already exist.

    Returns:
        ``{chunk_key: global_start_frame}`` plus a ``_meta`` entry, also written to
        ``outputs/<match>/chunk_offsets.json``.
    """
    mid = match_id(game)
    dest = MATCHES_ROOT / mid
    off_path = Path(str(OFFSETS).format(match=mid))
    if off_path.exists() and not overwrite:
        logger.info("%s: chunks already prepared -> %s", game, off_path)
        return json.loads(off_path.read_text(encoding="utf-8"))

    info = probe(game)
    boundary = half_boundary_frame(game)
    total_s = info["nb_frames"] / FPS
    times = sorted({round(t, 3) for t in
                    [*(float(s) for s in range(seconds, int(total_s), seconds)), boundary / FPS]})
    work = dest / "_seg"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    listing = work / "segments.csv"
    subprocess.run(
        [ffbin("ffmpeg"), "-hide_banner", "-loglevel", "warning", "-i", str(video_path(game)),
         "-c", "copy", "-map", "0:v:0", "-f", "segment",
         "-segment_times", ",".join(f"{t:.3f}" for t in times),
         "-segment_list", str(listing), "-segment_list_type", "csv",
         "-reset_timestamps", "1", str(work / "seg_%03d.mp4")], check=True)

    rows = list(csv.reader(listing.read_text(encoding="utf-8").splitlines()))
    for idx, (name, start, _end) in enumerate(rows):
        half = "h1" if float(start) * FPS < boundary else "h2"
        (dest / half).mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / Path(name).name), str(dest / half / f"chunk_{idx:03d}.mp4"))
    shutil.rmtree(work, ignore_errors=True)
    return write_offsets(game)


def nb_frames(path: Path) -> int:
    """Container frame count of a video (no decode)."""
    out = subprocess.run(
        [ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def write_offsets(game: str) -> dict:
    """Recompute the per-chunk global frame offsets from the chunks themselves.

    The offsets are **cumulative frame counts**, not ``segment_list`` start times: two of the three
    files carry a non-zero container ``start_time`` (0.0230 s), which makes ``round(start * fps)``
    land one frame late from the second chunk on. Cumulative counting is exact by construction and
    is cross-checked here against the source's own frame count.

    Args:
        game: FOOTPASS game id.

    Returns:
        The offsets payload, also written to ``outputs/<match>/chunk_offsets.json``.
    """
    mid = match_id(game)
    dest = MATCHES_ROOT / mid
    chunks = sorted(((p.parent.name, int(p.stem.split("_")[1]), p)
                     for h in ("h1", "h2") for p in (dest / h).glob("chunk_*.mp4")),
                    key=lambda r: r[1])
    offsets: dict[str, int] = {}
    counts: dict[str, int] = {}
    cursor = 0
    for half, idx, path in chunks:
        key = f"{half}_chunk_{idx:03d}"
        offsets[key] = cursor
        counts[key] = nb_frames(path)
        cursor += counts[key]
    info = probe(game)
    payload = {**offsets, "_meta": {
        "game": game, "fps": FPS, "nb_frames": info["nb_frames"],
        "frames_in_chunks": cursor, "frames_lost": info["nb_frames"] - cursor,
        "half_boundary_frame": half_boundary_frame(game), "n_chunks": len(offsets),
        "chunk_frames": counts,
        "note": "global_frame = chunk_offset + local_frame (local_frame = the extract parquet's "
                "'frame' column, which counts source frames of the chunk video from 0); offsets "
                "are cumulative container frame counts, NOT segment_list start times"}}
    off_path = Path(str(OFFSETS).format(match=mid))
    off_path.parent.mkdir(parents=True, exist_ok=True)
    off_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("%s: %d chunks, %d frames in chunks vs %d in source (lost %d) -> %s", game,
                len(offsets), cursor, info["nb_frames"], info["nb_frames"] - cursor, off_path)
    return payload


def chunk_offsets(game: str) -> dict[str, int]:
    """``{chunk_key: global_start_frame}`` written by :func:`chunk_game` (no ``_meta``)."""
    raw = json.loads(Path(str(OFFSETS).format(match=match_id(game))).read_text(encoding="utf-8"))
    return {k: int(v) for k, v in raw.items() if not k.startswith("_")}


def verify_chunks(game: str, *, decode: bool = False) -> dict:
    """Check the chunk set covers every source frame exactly once, with no join gaps.

    Args:
        game: FOOTPASS game id.
        decode: Count frames by full decode (``-count_frames``, minutes per chunk) instead of
            trusting the container header. The header and the decode agreed on all three files.
    """
    offs = chunk_offsets(game)
    order = sorted(offs.items(), key=lambda kv: kv[1])
    counted = []
    for key, f0 in order:
        path = MATCHES_ROOT / match_id(game) / key[:2] / f"chunk_{key.split('_chunk_')[1]}.mp4"
        if decode:
            out = subprocess.run(
                [ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=True).stdout.strip()
            n = int(out)
        else:
            n = nb_frames(path)
        counted.append((key, f0, n))
    gaps = [(counted[i][0], counted[i][1] + counted[i][2] - counted[i + 1][1])
            for i in range(len(counted) - 1)]
    total = sum(c for _k, _f, c in counted)
    src = probe(game)["nb_frames"]
    return {"game": game, "n_chunks": len(counted), "total_frames": total,
            "nb_frames_source": src, "frames_lost": src - total, "decoded": decode,
            "bad_joins": {k: int(g) for k, g in gaps if g != 0},
            "per_chunk": [{"chunk": k, "offset": f, "frames": c} for k, f, c in counted]}


# === lineups =====================================================================================
def build_lineup(game: str, *, team_map: dict[int, int] | None = None) -> pd.DataFrame:
    """Per-half roster in the schema ``tools/identity_match.py`` reads.

    FOOTPASS carries no names, so a player's key is ``T<footpass_team>#<shirt>`` -- unique within a
    game and exactly the ``(team, jersey)`` identity PCBAS scores. ``on_h1``/``on_h2`` come from
    which half-datasets the player appears in, which is the only substitution signal available.

    Args:
        game: FOOTPASS game id.
        team_map: FOOTPASS team (1/2) -> pipeline team id (0/1). Defaults to ``{1: 0, 2: 1}``;
            :func:`resolve_team_map` measures the truth once the OCR pass exists.

    Returns:
        The lineup table (also written to ``outputs/identity/<match>_lineup_assign.parquet``).
    """
    tmap = team_map or {1: 0, 2: 1}
    seen: dict[tuple[int, int], dict] = {}
    for half in ("H1", "H2"):
        d = load_digest(game, half)
        for team, shirt, role in zip(d["team"], d["shirt"], d["role"]):
            key = (int(team), int(shirt))
            rec = seen.setdefault(key, {"fp_team": int(team), "shirt": int(shirt),
                                        "role_id": int(role), "on_H1": False, "on_H2": False})
            rec[f"on_{half}"] = True
            rec["role_id"] = int(role)
    rows = []
    for (fp_team, shirt), rec in sorted(seen.items()):
        rows.append({
            "team": int(tmap[fp_team]), "name": f"T{fp_team}#{shirt}", "shirt": shirt,
            "position": "G" if rec["role_id"] == GK_ROLE_ID else "O",
            "is_sub": not rec["on_H1"], "on_h1": rec["on_H1"], "on_h2": rec["on_H2"],
            "assigned": False, "confidence": 0.0, "method": "footpass_annotation",
            "vote_mass": 0, "n_fragments": 0, "mean_u": float("nan"), "track_ids": "",
            "team_name": f"FOOTPASS team {fp_team}", "fp_team": int(fp_team),
            "role_id": int(rec["role_id"]),
        })
    df = pd.DataFrame(rows)
    out = Path(str(LINEUP).format(match=match_id(game)))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("%s: %d roster slots (%d h1, %d h2, %d GK) -> %s", game, len(df),
                int(df["on_h1"].sum()), int(df["on_h2"].sum()),
                int((df["position"] == "G").sum()), out)
    return df


def percrop_reads_table(game: str) -> pd.DataFrame:
    """``pipeline_team, number`` for every jersey read the per-crop OCR pass emitted.

    Joins the frozen aggregation rule's reads onto the aligned table's per-track majority team, so
    :func:`resolve_team_map` sees exactly the ``(our team id, number)`` pairs the pipeline believes.
    """
    from tools.identity_match import percrop_reads  # noqa: PLC0415
    from tools.footpass_predict import track_teams  # noqa: PLC0415

    mid = match_id(game)
    teams = track_teams(registry.get(mid).load_aligned())
    rows = [{"game": game, "pipeline_team": teams.get((chunk, int(tid)), -1), "number": int(num)}
            for chunk, by_track in percrop_reads(mid).items()
            for tid, votes in by_track.items() for num, _c in votes]
    out = pd.DataFrame(rows)
    return out[out["pipeline_team"] >= 0] if len(out) else out


def resolve_team_map(game: str, reads: pd.DataFrame) -> dict[int, int]:
    """Decide which pipeline team id is FOOTPASS team 1, from jersey reads alone.

    Each read is a ``(pipeline_team, number)`` pair; a number that exists on exactly one FOOTPASS
    roster votes for that pairing. Uses the rosters (a declared input) and the OCR (our own output) --
    never the event labels.

    Args:
        game: FOOTPASS game id.
        reads: ``pipeline_team, number`` rows (one per emitted jersey read).

    Returns:
        ``{1: pipeline_team_for_footpass_team_1, 2: ...}``.
    """
    lu = pd.read_parquet(Path(str(LINEUP).format(match=match_id(game))))
    by_fp = {t: set(g["shirt"].astype(int)) for t, g in lu.groupby("fp_team")}
    only1 = by_fp[1] - by_fp[2]
    only2 = by_fp[2] - by_fp[1]
    votes = {0: 0, 1: 0}
    for pt, num in zip(reads["pipeline_team"].astype(int), reads["number"].astype(int)):
        if num in only1:
            votes[pt] += 1
        elif num in only2:
            votes[pt] -= 1
    logger.info("%s: team-map votes %s (unique shirts: team1 %d, team2 %d)", game, votes,
                len(only1), len(only2))
    return {1: 0, 2: 1} if votes[0] >= votes[1] else {1: 1, 2: 0}


# === CLI =========================================================================================
def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["probe", "align", "chunk", "offsets", "verify", "lineup", "extract",
                             "teammap"])
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--duration", type=float, default=None, help="align: analyse only N seconds")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sample-every", type=int, default=2,
                    help="extract: frame stride (2 = the FOOTPASS_E2E_PLAN A.6 pick)")
    ap.add_argument("--calib-period", type=int, default=62,
                    help="extract: PnLCalib every N SAMPLED frames; 62 at stride 2 "
                         "holds the cadence at one calibration per 5 s of video")
    ap.add_argument("--tracker", default="botsort",
                    help="extract: botsort (A.5 pick) or bytetrack")
    ap.add_argument("--reads", type=Path, default=None,
                    help="teammap: parquet with pipeline_team, number")
    args = ap.parse_args()
    games = tuple(args.games.split(","))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "extract":
        from tools.batch_match import run_batch  # noqa: PLC0415  (GPU stack, imported on demand)

        for g in games:
            for half in ("h1", "h2"):
                chunks_dir = MATCHES_ROOT / match_id(g) / half
                if not any(chunks_dir.glob("chunk_*.mp4")):
                    continue
                logger.info("EXTRACT %s %s (GPU) stride=%d calib_period=%d tracker=%s",
                            g, half, args.sample_every, args.calib_period, args.tracker)
                run_batch(chunks_dir, Path("outputs") / match_id(g) / half / "match",
                          sample_every=args.sample_every, calib_period=args.calib_period,
                          calib_drift=2.0, detector="football", tracker=args.tracker,
                          skip_existing=True, limit=None)
        return

    if args.stage == "teammap":
        out = {}
        for g in games:
            reads = pd.read_parquet(args.reads) if args.reads else percrop_reads_table(g)
            if "game" in reads:
                reads = reads[reads["game"] == g]
            tmap = resolve_team_map(g, reads)
            build_lineup(g, team_map=tmap)
            out[g] = {"map": {str(k): v for k, v in tmap.items()}, "n_reads": int(len(reads))}
        (RESULTS_DIR / "team_map.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    runner = {"probe": probe, "align": lambda g: align_check(g, duration=args.duration),
              "chunk": lambda g: chunk_game(g, overwrite=args.overwrite),
              "offsets": write_offsets, "verify": verify_chunks,
              "lineup": lambda g: build_lineup(g).to_dict("list")}[args.stage]
    payload = {g: runner(g) for g in games}
    if args.stage in ("probe", "align", "verify"):
        path = RESULTS_DIR / f"{args.stage}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(json.dumps(payload, indent=2, default=str)[:4000])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
