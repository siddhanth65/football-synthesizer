"""Match-consistent team identity via jersey colour (cross-chunk / cross-half anchoring).

The per-chunk jersey KMeans labels teams arbitrarily, and aligning by defended goal flips at half-time
(teams swap ends). Jersey **colour** is invariant the whole match, so we anchor on it: sample each
player track's torso colour from the video, then cluster **all tracks across all chunks together** into
two teams. That single global split is consistent across chunks *and* halves.

The positions parquet stores only the foot point (no bounding box), so we estimate a torso box from the
foot point scaled by vertical image position (a crude depth proxy) and reuse
:func:`generator.teams.jersey_color`. Distinct kits (e.g. red vs sky-blue) cluster cleanly despite the
approximate box.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from generator.teams import jersey_color

PLAYER_ROLES = ("player", "goalkeeper")
# Player pixel height as a fraction of frame height: larger near the bottom of frame (closer to camera).
_H_FRAC_NEAR, _H_FRAC_FAR = 0.13, 0.035
_W_TO_H = 0.45  # torso/player width-to-height ratio


def estimate_player_box(image_x: float, image_y: float, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    """A rough ``(x1, y1, x2, y2)`` player box from the foot point, depth-scaled by image row."""
    t = float(np.clip(image_y / max(frame_h, 1), 0.0, 1.0))  # 0 top (far) .. 1 bottom (near)
    h_px = frame_h * (_H_FRAC_FAR + (_H_FRAC_NEAR - _H_FRAC_FAR) * t)
    w_px = h_px * _W_TO_H
    x1, x2 = image_x - w_px / 2, image_x + w_px / 2
    y1, y2 = image_y - h_px, image_y
    return (max(int(x1), 0), max(int(y1), 0), min(int(x2), frame_w), min(int(y2), frame_h))


def chunk_track_colors(positions_chunk: pd.DataFrame, video_path: str, *, sample_frames: int = 120,
                       min_conf: float = 0.4) -> dict[int, np.ndarray]:
    """Mean torso CIELAB colour per player track in one chunk, sampled from its video."""
    import cv2  # noqa: PLC0415

    pl = positions_chunk[positions_chunk["role"].isin(PLAYER_ROLES)].dropna(subset=["image_x", "image_y"])
    if "conf" in pl.columns:
        pl = pl[pl["conf"].fillna(1.0) >= min_conf]
    if pl.empty:
        return {}
    cap = cv2.VideoCapture(video_path)
    fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = np.sort(pl["frame"].unique())
    pick = frames[np.linspace(0, len(frames) - 1, min(sample_frames, len(frames))).astype(int)]
    by_frame = {fr: g for fr, g in pl[pl["frame"].isin(pick)].groupby("frame")}
    acc: dict[int, list] = defaultdict(list)
    for fr in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, bgr = cap.read()
        if not ok:
            continue
        for r in by_frame.get(fr, pd.DataFrame()).itertuples(index=False):
            x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
            crop = bgr[y1:y2, x1:x2]
            if crop.size:
                acc[int(r.track_id)].append(jersey_color(crop))
    cap.release()
    return {tid: np.mean(labs, axis=0) for tid, labs in acc.items() if labs}


def anchor_teams(positions: pd.DataFrame, video_map: dict[str, str], *, sample_frames: int = 120,
                 random_state: int = 0, chunkwise: bool = True) -> pd.DataFrame:
    """Relabel player ``team`` to a **match-consistent** id by clustering jersey colour.

    Args:
        positions: Combined match positions (needs a ``chunk`` column).
        video_map: ``chunk`` value -> video path.
        sample_frames: Frames sampled per chunk for colour.
        chunkwise: use the robust chunkwise-then-match anchoring (:func:`apply_chunkwise_team_labels`);
            ``False`` falls back to pooling every track into one global KMeans
            (:func:`apply_global_team_labels`).

    Returns:
        A copy of ``positions`` with player/GK ``team`` set to the global cluster (0/1); non-player rows
        keep ``team = -1``. Track ids without colour evidence are left as -1.
    """
    track_colors: dict[tuple[str, int], np.ndarray] = {}
    for chunk, g in positions.groupby("chunk"):
        if chunk not in video_map:
            continue
        for tid, lab in chunk_track_colors(g, video_map[chunk], sample_frames=sample_frames).items():
            track_colors[(chunk, int(tid))] = lab
    label = apply_chunkwise_team_labels if chunkwise else apply_global_team_labels
    return label(positions, track_colors, random_state=random_state)


def apply_chunkwise_team_labels(positions: pd.DataFrame, track_colors: dict[tuple[str, int], np.ndarray],
                                *, random_state: int = 0) -> pd.DataFrame:
    """Robust cross-chunk team anchoring: split each chunk into two teams, then match teams across chunks.

    Pooling every track's colour into one global KMeans is fragile -- noisy per-track colours (grass
    contamination, motion blur) cross the team boundary, so on a *follow-play* broadcast a whole chunk can
    land on the wrong team. Instead we (1) KMeans(2) **within each chunk**, where the two kits separate
    cleanly under consistent lighting, then (2) KMeans(2) the resulting **chunk-cluster centroids** (two
    denoised points per chunk) to map each chunk's two clusters onto the two global teams. Far steadier
    when the two kits are only moderately distinct.

    Pure (no video): the testable core. Player/GK rows get the global team; tracks without colour and all
    non-player rows become ``team = -1``.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    by_chunk: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for (chunk, tid), lab in track_colors.items():
        by_chunk[chunk][int(tid)] = lab
    track_local: dict[tuple[str, int], tuple[str, int]] = {}
    centroids: dict[tuple[str, int], np.ndarray] = {}
    for chunk, tcs in by_chunk.items():
        tids = list(tcs)
        if len(tids) < 2:
            continue
        x = np.stack([tcs[t] for t in tids])
        local = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(x)
        for t, lab_i in zip(tids, local):
            track_local[(chunk, t)] = (chunk, int(lab_i))
        for lab_i in (0, 1):
            sel = x[local == lab_i]
            if len(sel):
                centroids[(chunk, int(lab_i))] = sel.mean(axis=0)
    if len(centroids) < 2:
        raise ValueError(f"too few chunk clusters to anchor ({len(centroids)})")
    ckeys = list(centroids)
    global_lab = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(
        np.stack([centroids[k] for k in ckeys]))
    global_of = {k: int(v) for k, v in zip(ckeys, global_lab)}
    team_of = {kt: global_of[loc] for kt, loc in track_local.items()}
    out = positions.copy()
    is_player = out["role"].isin(PLAYER_ROLES)
    out.loc[is_player, "team"] = [
        team_of.get((c, int(t)), -1)
        for c, t in zip(out.loc[is_player, "chunk"], out.loc[is_player, "track_id"])
    ]
    out.loc[~is_player, "team"] = -1
    return out


def apply_global_team_labels(positions: pd.DataFrame, track_colors: dict[tuple[str, int], np.ndarray],
                             *, random_state: int = 0) -> pd.DataFrame:
    """Cluster ``(chunk, track_id) -> LAB`` colours into two global teams and relabel player rows.

    Pure (no video): the testable core of :func:`anchor_teams`. Player/GK rows get the global cluster
    id; tracks without colour and all non-player rows become ``team = -1``.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    if len(track_colors) < 2:
        raise ValueError(f"too few coloured tracks to anchor ({len(track_colors)})")
    keys = list(track_colors)
    labels = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(
        np.stack([track_colors[k] for k in keys]))
    team_of = {k: int(v) for k, v in zip(keys, labels)}
    out = positions.copy()
    is_player = out["role"].isin(PLAYER_ROLES)
    out.loc[is_player, "team"] = [
        team_of.get((c, int(t)), -1)
        for c, t in zip(out.loc[is_player, "chunk"], out.loc[is_player, "track_id"])
    ]
    out.loc[~is_player, "team"] = -1
    return out


def global_team_map(cent: dict[tuple[str, int], np.ndarray], *, dark_is_team0: bool = True,
                    random_state: int = 0) -> dict[tuple[str, int], int]:
    """Map each ``(chunk, per-chunk-team)`` LAB centroid to a global team (0/1), anchored by brightness.

    Clusters the centroids into two global teams, then makes the **darker-kit** cluster global ``0`` (when
    ``dark_is_team0``), so identity is consistent across chunks/halves without a which-cluster ambiguity.
    Pure (no video) -- the testable core of :func:`align_teams_by_color`.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    if len(cent) < 2:
        raise ValueError(f"too few chunk-teams to align ({len(cent)})")
    keys = list(cent)
    gl = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(
        np.stack([cent[k] for k in keys]))
    glob = {k: int(v) for k, v in zip(keys, gl)}
    mean_L = {c: float(np.mean([cent[k][0] for k in keys if glob[k] == c])) for c in (0, 1)}
    dark = min(mean_L, key=mean_L.get)
    remap = {dark: (0 if dark_is_team0 else 1), 1 - dark: (1 if dark_is_team0 else 0)}
    return {k: remap[glob[k]] for k in keys}


def align_teams_by_color(positions: pd.DataFrame, video_map: dict[str, str], *, sample_frames: int = 150,
                         dark_is_team0: bool = True, random_state: int = 0) -> pd.DataFrame:
    """Make per-chunk team labels globally consistent + identity-anchored, keeping ALL labelled players.

    batch_match labels every player-detection but numbers the two teams arbitrarily per chunk. This takes
    those complete per-chunk labels and (1) samples each per-chunk team's mean jersey colour, (2) clusters
    the per-(chunk, team) centroids into two global teams (so a chunk's "team 0" maps to whichever global
    team its colour matches), and (3) **anchors identity by brightness** -- the darker-kit global team
    becomes ``0`` (set ``dark_is_team0``), removing the which-cluster-is-which ambiguity. Unlike
    :func:`apply_chunkwise_team_labels` it relabels in place (per-detection), so brief track fragments stay
    labelled rather than dropping to ``-1``.

    Returns a copy of ``positions`` with player/GK ``team`` globally consistent (0/1); non-players ``-1``.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    cent: dict[tuple[str, int], np.ndarray] = {}
    for chunk, g in positions.groupby("chunk"):
        if chunk not in video_map:
            continue
        cols = chunk_track_colors(g, video_map[chunk], sample_frames=sample_frames)  # track -> LAB
        gp = g[g["role"].isin(PLAYER_ROLES)]
        team_of_track = dict(zip(gp["track_id"].astype(int), gp["team"].astype(int)))
        for team in (0, 1):
            labs = [cols[t] for t in cols if team_of_track.get(int(t)) == team]
            if labs:
                cent[(chunk, team)] = np.mean(labs, axis=0)
    final = global_team_map(cent, dark_is_team0=dark_is_team0, random_state=random_state)
    out = positions.copy()
    is_pl = out["role"].isin(PLAYER_ROLES)
    out.loc[is_pl, "team"] = [
        final.get((c, int(t)), -1)
        for c, t in zip(out.loc[is_pl, "chunk"], out.loc[is_pl, "team"])
    ]
    out.loc[~is_pl, "team"] = -1
    return out


def main() -> None:
    import argparse  # noqa: PLC0415
    import logging  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="combined match positions parquet (with 'chunk')")
    ap.add_argument("--chunks-dir", required=True, help="dir of <chunk>.mp4 videos")
    ap.add_argument("--out", default=None, help="write the team-anchored positions parquet here")
    ap.add_argument("--sample-frames", type=int, default=120)
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    video_map = {c: str(Path(args.chunks_dir) / f"{c}.mp4") for c in pos["chunk"].unique()}
    logging.info("anchoring %d chunks by jersey colour...", len(video_map))
    anchored = anchor_teams(pos, video_map, sample_frames=args.sample_frames)
    if args.out:
        anchored.to_parquet(args.out, index=False)
        logging.info("wrote %s", args.out)
    from fingerprint.team_style import match_style_vector  # noqa: PLC0415

    z = match_style_vector(anchored)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    cols = ["team", "frames", "avg_players", "width", "length", "compactness", "surface_area",
            "lane_centre", "buildup_height", "def_line_height", "attacking_third_share", "wing_share",
            "top_speed_kmh"]
    print("\nFULL-MATCH team-style fingerprint (colour-anchored, per-chunk direction):\n")
    print(z[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

