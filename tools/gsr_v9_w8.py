"""v9 W8: the ATTACH oracle -- can a discarded detection inherit an existing track's identity?

v9 W7 established two facts that frame this session:

* 66.1% of the DEV-20 GS-DetA miss bucket (13,167 rows) is a **confident** detector box the chain
  discards after the detector, most of it inside ``supervision.ByteTrack`` (kb v9-w7-001);
* recovering those boxes as **new** tracks is worth nothing (-10 fully-correct rows for 3,113
  recovered misses, kb v9-w7-004) because the identity machinery never names a fresh short track.

The untested mechanism is ATTACH: give the discarded box to an **existing** track, so it inherits
that track's role/team/jersey. This module prices that mechanism on CPU, before any GPU arm:

1. **discards** -- confident detector boxes (``lowconf`` dump at ``DETECT_CONF``) with no row in the
   control's ``positions`` parquet for the same frame (tolerance :data:`MATCH_TOL_PX`);
2. **classification** -- each discard is ``gt_miss`` (its GT row has no submission row within 5 m),
   ``gt_covered`` (its GT row is already matched, so an attach is a duplicate) or ``no_gt``;
3. **attachability** -- for a ``gt_miss`` discard, does a submission track carrying the SAME GT
   identity exist within ``gap`` timesteps and have no row at this timestep;
4. **host quality** -- are that host's own attributes (role, team, jersey) equal to the GT row's;
5. **attach precision** -- what the geometry-only candidate rule (pan-compensated nearest gap track)
   actually picks, and how often that is the wrong identity.

Nothing here touches GT except to *measure*: the counterfactual submissions it writes copy
attributes from our own tracks and project the box with a homography fitted to our own rows.

Stages::

    python -m tools.gsr_v9_w8 --stage oracle     # census + attach tables (CPU, no scoring)
    python -m tools.gsr_v9_w8 --stage score      # build counterfactual submissions and score them
    python -m tools.gsr_v9_w8 --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_v9_deta import effective_attrs, load_gt, load_pred, match_positions, similarity
from tools.gsr_v9_w7 import match_feet

logger = logging.getLogger("gsr_v9_w8")

#: Person roles the detector emits (the ball is never attached).
PERSON_ROLES = ("player", "goalkeeper", "referee")
#: The shipped detector confidence floor.
DETECT_CONF = 0.20
#: Shipped ByteTrack settings, replayed to reproduce the discard set (kb v9-w7-005).
MIN_HITS = 3
LOST_BUFFER = 60
#: Temporal gaps (timesteps) the attach oracle reports.
GAPS = (1, 3, 10, 30)
#: Minimum separation (box heights) from the nearest KEPT box, swept as a GT-free purity gate.
SEPS = (0.0, 0.3, 0.6, 1.0)
#: Track-id offset for rows written by a counterfactual arm (never collides with a real id).
ATTACH_ID_TAG = "att"


# === pure helpers =================================================================================
def sub_foot(row: dict) -> tuple[float, float]:
    """Image-space bottom-middle point of a submission row (pure)."""
    b = row["bbox_image"]
    return float(b["x_center"]), float(b["y"] + b["h"])


def sub_pitch(row: dict) -> tuple[float, float]:
    """Pitch bottom-middle point of a submission row (pure)."""
    b = row["bbox_pitch"]
    return float(b["x_bottom_middle"]), float(b["y_bottom_middle"])


def fit_homography(img: np.ndarray, pitch: np.ndarray):
    """Least-squares image->pitch homography from a frame's own rows (pure); ``None`` if degenerate.

    The submission carries both the image foot point and the pitch point of every row it wrote, and
    those are related by exactly the homography the extraction used, so four rows recover it with no
    GT contact and no re-calibration.
    """
    if len(img) < 4:
        return None
    import cv2  # noqa: PLC0415

    h, _mask = cv2.findHomography(img.astype(float), pitch.astype(float), 0)
    return h


def apply_homography(h, pts: np.ndarray) -> np.ndarray:
    """Project image points to pitch metres with ``h`` (pure)."""
    import cv2  # noqa: PLC0415

    return cv2.perspectiveTransform(np.asarray(pts, float).reshape(-1, 1, 2), h).reshape(-1, 2)


def pan_shift(prev: dict[int, tuple[float, float]],
              cur: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """Global frame-to-frame image shift as the median displacement of shared tracks (pure).

    This is the cheap stand-in for camera motion: when the camera pans, every track moves together,
    so the median of the per-track displacement is the pan and the residual is real player motion.
    """
    common = [k for k in cur if k in prev]
    if not common:
        return 0.0, 0.0
    d = np.array([[cur[k][0] - prev[k][0], cur[k][1] - prev[k][1]] for k in common], float)
    return float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def _bucket(value: float, edges: tuple[float, ...]) -> str:
    """Name the half-open bin ``value`` falls in (pure)."""
    for lo, hi in zip((0.0, *edges), (*edges, float("inf"))):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}" if hi != float("inf") else f">={lo:g}"
    return "?"


# === per-sequence state ===========================================================================
class SeqState:
    """Everything one sequence's attach analysis needs, derived once.

    Attributes:
        gt_rows: Per-timestep GT annotations.
        pr_rows: Per-timestep control submission rows.
        frame_of_ts: Timestep -> 0-based image index.
        discards: ``{timestep: [(foot_xy, box_xyxy, conf, role), ...]}`` -- confident detector boxes
            the control's positions parquet never received.
        track_ts: ``{track id: {timestep: row}}`` for the control submission.
        track_gt: ``{track id: GT track id}`` by majority over matched timesteps.
        gt_covered: ``{timestep: set of GT indices matched by a submission row}``.
        gt_of_discard: ``{(timestep, discard index): GT index}`` from the image-space match.
        cum: ``(T, 2)`` cumulative global image shift, timestep 0 at the origin.
        pan: ``(T,)`` per-timestep pan magnitude in px.
        homog: ``{timestep: 3x3}`` image->pitch homography fitted on the submission's own rows.
    """

    def __init__(self, seq_dir: Path, pred_path: Path, positions: Path, lowconf: Path):
        self.name = seq_dir.name
        ts, self.gt_rows = load_gt(seq_dir)
        self.pr_rows = load_pred(pred_path, ts, len(self.gt_rows))
        self.frame_of_ts = {k: int(img_id[-6:]) - 1 for img_id, k in ts.items()}
        self.id_of_ts = {v: k for k, v in ts.items()}
        self._build_tracks()
        self._build_discards(positions, lowconf)
        self._build_matches()
        self._build_motion()

    # --- construction -----------------------------------------------------------------------
    def _build_tracks(self) -> None:
        self.track_ts: dict[int, dict[int, dict]] = defaultdict(dict)
        for t, rows in enumerate(self.pr_rows):
            for r in rows:
                self.track_ts[int(r["track_id"])][t] = r
        self.track_ts = dict(self.track_ts)
        self.track_span = {k: (min(v), max(v)) for k, v in self.track_ts.items()}

    def _build_discards(self, positions: Path, lowconf: Path) -> None:
        """Split this sequence's confident detector boxes into kept and ByteTrack-discarded.

        The split comes from REPLAYING ``supervision.ByteTrack`` over the cached detector dump, not
        from matching the dump against the shipped ``positions`` parquet: the dump is a separate
        inference pass whose boxes differ from the pipeline's by up to a few pixels, and a tolerance
        join at 2 px leaves 22,210 of 225,926 DEV-20 positions rows unmatched, which would
        misclassify 9.8% of kept boxes as discards. The replay is self-consistent by construction and
        is validated to 0.0004 of the extraction's recall (kb v9-w7-005).
        """
        import supervision as sv  # noqa: PLC0415

        from generator.tracking import untracked_mask  # noqa: PLC0415

        pos = pd.read_parquet(positions / f"{self.name}.parquet")
        self.n_pos = int(len(pos[pos["role"].isin(PERSON_ROLES)]))
        det = pd.read_parquet(lowconf / f"{self.name}.parquet")
        det = det[det["role"].isin(PERSON_ROLES) & (det["conf"] >= DETECT_CONF)]
        self.n_det = int(len(det))
        det_by_frame = {int(f): g for f, g in det.groupby("frame")}
        bt = sv.ByteTrack(minimum_consecutive_frames=MIN_HITS, lost_track_buffer=LOST_BUFFER)
        drop_by_frame: dict[int, np.ndarray] = {}
        kept_by_frame: dict[int, np.ndarray] = {}
        for frame in range(int(det["frame"].max()) + 1) if len(det) else ():
            g = det_by_frame.get(frame)
            if g is None or not len(g):
                continue  # extract.py skips the tracker update on frames with no detection
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            out = bt.update_with_detections(sv.Detections(
                xyxy=boxes, confidence=g["conf"].to_numpy(float),
                class_id=np.zeros(len(g), int)))
            drop_by_frame[frame] = untracked_mask(boxes, out.xyxy)
            kept_by_frame[frame] = out.xyxy
        self.discards: dict[int, list[tuple]] = {}
        self.kept: dict[int, np.ndarray] = {}
        for t in range(len(self.gt_rows)):
            frame = self.frame_of_ts[t]
            g = det_by_frame.get(frame)
            if g is None or not len(g):
                continue
            foot = g[["image_x", "image_y"]].to_numpy(float)
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            confs = g["conf"].to_numpy(float)
            roles = g["role"].to_numpy(str)
            drop = drop_by_frame.get(frame, np.zeros(len(foot), bool))
            k = kept_by_frame.get(frame, np.zeros((0, 4)))
            self.kept[t] = (np.column_stack([(k[:, 0] + k[:, 2]) / 2, k[:, 3]]) if len(k)
                            else np.zeros((0, 2)))
            self.discards[t] = [(foot[i], boxes[i], float(confs[i]), str(roles[i]))
                                for i in range(len(foot)) if drop[i]]
        self._det_by_frame = det_by_frame

    def _build_matches(self) -> None:
        self.gt_covered: dict[int, set[int]] = {}
        self.gt_of_discard: dict[tuple[int, int], int] = {}
        gt_hits: dict[int, Counter] = defaultdict(Counter)
        for t, (gts, prs) in enumerate(zip(self.gt_rows, self.pr_rows)):
            sim = similarity(_pitch_xy_gt(gts), _pitch_xy_pr(prs))
            pairs = match_positions(sim)
            self.gt_covered[t] = {i for i, _j in pairs}
            for i, j in pairs:
                gt_hits[int(prs[j]["track_id"])][int(gts[i]["track_id"])] += 1
            # image-space match of GT to the FULL confident detector set, so a discard cannot claim
            # a GT row that a kept box already explains
            frame = self.frame_of_ts[t]
            g = self._det_by_frame.get(frame)
            disc = self.discards.get(t, [])
            if g is None or not len(g) or not gts:
                continue
            foot = g[["image_x", "image_y"]].to_numpy(float)
            hit = match_feet([x["bbox_image"] for x in gts], foot)
            disc_foot = {tuple(np.round(d[0], 6)): k for k, d in enumerate(disc)}
            for gi, di in hit.items():
                k = disc_foot.get(tuple(np.round(foot[di], 6)))
                if k is not None:
                    self.gt_of_discard[(t, k)] = gi
        self.track_gt = {k: (c.most_common(1)[0][0] if c else None) for k, c in gt_hits.items()}
        self.track_gt.update({k: self.track_gt.get(k) for k in self.track_ts})
        self.gt_track_tracks: dict[int, list[int]] = defaultdict(list)
        for tid, gid in self.track_gt.items():
            if gid is not None:
                self.gt_track_tracks[gid].append(tid)

    def _build_motion(self) -> None:
        n = len(self.gt_rows)
        self.cum = np.zeros((n, 2))
        self.pan = np.zeros(n)
        self.homog: dict[int, object] = {}
        prev: dict[int, tuple[float, float]] = {}
        for t in range(n):
            cur = {int(r["track_id"]): sub_foot(r) for r in self.pr_rows[t]}
            if t:
                dx, dy = pan_shift(prev, cur)
                self.cum[t] = self.cum[t - 1] + (dx, dy)
                self.pan[t] = float(np.hypot(dx, dy))
            if cur:
                prev = cur
            rows = self.pr_rows[t]
            if len(rows) >= 4:
                h = fit_homography(np.array([sub_foot(r) for r in rows]),
                                   np.array([sub_pitch(r) for r in rows]))
                if h is not None:
                    self.homog[t] = h

    # --- queries ----------------------------------------------------------------------------
    def predict_track(self, tid: int, t: int, gap: int) -> tuple[np.ndarray, float, int] | None:
        """Pan-compensated image prediction of a gap track at timestep ``t``.

        Returns ``(xy, box height, |dt|)`` for the temporally nearest row of ``tid`` inside
        ``[t - gap, t + gap]``, or ``None`` when the track has a row at ``t`` or none in range.
        """
        seen = self.track_ts.get(tid)
        if not seen or t in seen:
            return None
        cand = [u for u in range(max(t - gap, 0), min(t + gap, len(self.gt_rows) - 1) + 1)
                if u in seen]
        if not cand:
            return None
        u = min(cand, key=lambda x: abs(x - t))
        r = seen[u]
        xy = np.array(sub_foot(r)) + (self.cum[t] - self.cum[u])
        return xy, float(r["bbox_image"]["h"]), abs(u - t)

    def track_attrs(self, tid: int) -> tuple:
        """Majority effective attributes of a submission track (pure over its own rows)."""
        c: Counter = Counter()
        for r in self.track_ts.get(tid, {}).values():
            a = r["attributes"] or {}
            c[effective_attrs(a.get("role"), a.get("team"), a.get("jersey"))] += 1
        return c.most_common(1)[0][0] if c else (None, None, None)


def _pitch_xy_gt(rows: list[dict]) -> np.ndarray:
    """GT bottom-middle pitch points as ``(N, 2)`` (pure)."""
    if not rows:
        return np.zeros((0, 2))
    return np.array([[r["bbox_pitch"]["x_bottom_middle"], r["bbox_pitch"]["y_bottom_middle"]]
                     for r in rows], float)


_pitch_xy_pr = _pitch_xy_gt


# === the oracle ===================================================================================
def analyse_sequence(st: SeqState, gaps: tuple[int, ...] = GAPS,
                     taus: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)) -> dict:
    """Census + attach tables for one sequence.

    Args:
        st: Prepared sequence state.
        gaps: Temporal gaps (timesteps) to report attachability at.
        taus: Candidate-rule acceptance radii, in units of the discarded box's height.

    Returns:
        Counters keyed by class, gap, pan bucket and tau.
    """
    c: Counter = Counter()
    c["n_det"] = st.n_det
    c["n_pos_ctl"] = st.n_pos
    for t, gts in enumerate(st.gt_rows):
        c["gt"] += len(gts)
        c["gt_kept_hit"] += len(match_feet([g["bbox_image"] for g in gts],
                                           st.kept.get(t, np.zeros((0, 2)))))
        c["n_kept"] += len(st.kept.get(t, ()))
    for t, disc in st.discards.items():
        pan = _bucket(float(st.pan[t]), (1.0, 3.0, 8.0))
        for k, (foot, box, _conf, role) in enumerate(disc):
            c["discard"] += 1
            c[f"discard::pan::{pan}"] += 1
            gi = st.gt_of_discard.get((t, k))
            if gi is None:
                cls = "no_gt"
            elif gi in st.gt_covered[t]:
                cls = "gt_covered"
            else:
                cls = "gt_miss"
            c[f"cls::{cls}"] += 1
            c[f"cls::{cls}::pan::{pan}"] += 1
            box_h = float(box[3] - box[1])
            gid, gattr = None, None
            if gi is not None:
                g = st.gt_rows[t][gi]
                gid = int(g["track_id"])
                ga = g["attributes"] or {}
                gattr = effective_attrs(ga.get("role"), ga.get("team"), ga.get("jersey"))
            if cls == "gt_miss":
                hosts = st.gt_track_tracks.get(gid, [])
                for gap in gaps:
                    ok = [tid for tid in hosts if st.predict_track(tid, t, gap) is not None]
                    busy = [tid for tid in hosts if t in st.track_ts.get(tid, {})]
                    if not ok:
                        c[f"{'host_busy' if busy else 'no_host'}::g{gap}"] += 1
                        continue
                    c[f"attachable::g{gap}"] += 1
                    c[f"attachable::g{gap}::pan::{pan}"] += 1
                    ha = st.track_attrs(min(ok, key=lambda tid: st.predict_track(tid, t, gap)[2]))
                    c[f"host_role_ok::g{gap}"] += ha[0] == gattr[0]
                    c[f"host_team_ok::g{gap}"] += ha[:2] == gattr[:2]
                    if ha == gattr:
                        c[f"host_named::g{gap}"] += 1
                        c[f"host_named::g{gap}::pan::{pan}"] += 1
            # candidate rule: pan-compensated nearest gap track, scale-normalised radius. Runs on
            # EVERY discard, because a GT-blind rule cannot tell a miss from a duplicate.
            for gap in gaps:
                pick = candidate_host(st, t, foot, box_h, gap, max(taus))
                if pick is None:
                    c[f"rule::g{gap}::nopick"] += 1
                    c[f"rule::g{gap}::nopick::{cls}"] += 1
                    continue
                tid, dist = pick
                if cls != "gt_miss":
                    outcome = f"onto_{cls}"
                elif st.track_gt.get(tid) != gid:
                    outcome = "wrong_identity"
                else:
                    outcome = ("correct" if st.track_attrs(tid) == gattr
                               else "right_id_wrong_attrs")
                sep = separation(st, t, foot, box_h)
                for tau in taus:
                    if dist > tau:
                        continue
                    c[f"rule::g{gap}::t{tau}::pick"] += 1
                    c[f"rule::g{gap}::t{tau}::{outcome}"] += 1
                    c[f"rule::g{gap}::t{tau}::pan::{pan}::pick"] += 1
                    c[f"rule::g{gap}::t{tau}::pan::{pan}::{outcome}"] += 1
                    for smin in SEPS:
                        if sep >= smin:
                            c[f"rule::g{gap}::t{tau}::s{smin}::pick"] += 1
                            c[f"rule::g{gap}::t{tau}::s{smin}::{outcome}"] += 1
    return dict(c)


def separation(st: SeqState, t: int, foot: np.ndarray, box_h: float) -> float:
    """Distance from a discarded box to the nearest KEPT box of the same frame, in box heights.

    A discarded box sitting on top of a box the tracker kept is a double detection: attaching it can
    only add a duplicate row. The gate is GT-free -- it reads the tracker's own output.
    """
    k = st.kept.get(t)
    if k is None or not len(k):
        return float("inf")
    return float(np.min(np.linalg.norm(k - foot, axis=1)) / max(box_h, 1.0))


def candidate_host(st: SeqState, t: int, foot: np.ndarray, box_h: float, gap: int,
                   tau: float) -> tuple[int, float] | None:
    """Geometry-only host pick: the pan-compensated nearest track absent at ``t`` (pure-ish).

    Args:
        st: Sequence state.
        t: Timestep of the discarded box.
        foot: Its image foot point.
        box_h: Its box height (the distance radius is normalised by it, so the rule is scale free).
        gap: How far back/forward a track may be missing and still be a candidate.
        tau: Acceptance radius in box heights (only used to bound the search; the caller thresholds).

    Returns:
        ``(track id, distance in box heights)`` or ``None``.
    """
    best: tuple[int, float] | None = None
    for tid in st.track_ts:
        span = st.track_span[tid]
        if t < span[0] - gap or t > span[1] + gap:
            continue
        p = st.predict_track(tid, t, gap)
        if p is None:
            continue
        xy, h_host, _dt = p
        if not (0.5 <= box_h / max(h_host, 1.0) <= 2.0):
            continue
        d = float(np.linalg.norm(xy - foot) / max(box_h, 1.0))
        if d <= tau and (best is None or d < best[1]):
            best = (tid, d)
    return best


# === counterfactual submissions ===================================================================
def _attached_row(st: SeqState, t: int, tid: int, foot: np.ndarray, box: np.ndarray,
                  conf: float, n: int) -> dict | None:
    """Build the submission row an attach would write (host attributes, own homography)."""
    h = st.homog.get(t)
    if h is None:
        return None
    px, py = apply_homography(h, foot[None, :])[0]
    role, team, jersey = st.track_attrs(tid)
    image_id = st.id_of_ts[t]
    return {
        "id": f"{ATTACH_ID_TAG}{image_id}{n:05d}", "image_id": image_id, "track_id": int(tid),
        "supercategory": "object", "category_id": 1,
        "attributes": {"role": role, "team": team, "jersey": jersey},
        "bbox_pitch": {"x_bottom_left": float(px), "y_bottom_left": float(py),
                       "x_bottom_middle": float(px), "y_bottom_middle": float(py),
                       "x_bottom_right": float(px), "y_bottom_right": float(py)},
        "bbox_image": {"x": float(box[0]), "y": float(box[1]),
                       "x_center": float((box[0] + box[2]) / 2),
                       "y_center": float((box[1] + box[3]) / 2),
                       "w": float(box[2] - box[0]), "h": float(box[3] - box[1])},
        "confidence": float(conf),
    }


def build_attach_arm(st: SeqState, mode: str, gap: int, tau: float, sep_min: float = 0.0,
                     pan_min: float = 0.0,
                     cap: int | None = None) -> tuple[list[dict], Counter]:
    """Rows a counterfactual attach would add, plus a row audit.

    Args:
        st: Sequence state.
        mode: ``"oracle"`` (attach only true misses onto the true-identity host) or ``"rule"``
            (attach whatever the geometry-only candidate rule picks, GT-blind).
        gap: Temporal gap allowed between the discarded box and the host's nearest row.
        tau: Acceptance radius in box heights (``rule`` mode only).
        sep_min: Minimum separation from the nearest kept box, in box heights (``rule`` mode only).
        pan_min: Only attach on frames whose global image shift is at least this many px/frame
            (``rule`` mode only) -- the pan regime is where the tracker's Kalman prediction drifts.
        cap: Optional maximum attaches per host track.

    Returns:
        ``(rows, audit counter)``.
    """
    rows: list[dict] = []
    audit: Counter = Counter()
    per_track: Counter = Counter()
    for t in sorted(st.discards):
        if mode != "oracle" and st.pan[t] < pan_min:
            continue
        # the official evaluator refuses a submission that emits one track id twice in a timestep,
        # so at most one box may attach to a given host per frame -- the nearest one wins
        picks: dict[int, tuple[float, int]] = {}
        for k, (foot, box, _conf, _role) in enumerate(st.discards[t]):
            box_h = float(box[3] - box[1])
            gi = st.gt_of_discard.get((t, k))
            if mode == "oracle":
                if gi is None or gi in st.gt_covered[t]:
                    continue
                gid = int(st.gt_rows[t][gi]["track_id"])
                ok = [(st.predict_track(x, t, gap), x) for x in st.gt_track_tracks.get(gid, [])]
                ok = [(p[2], x) for p, x in ok if p is not None]
                if not ok:
                    continue
                score, tid = min(ok)
            else:
                if separation(st, t, foot, box_h) < sep_min:
                    continue
                pick = candidate_host(st, t, foot, box_h, gap, tau)
                if pick is None:
                    continue
                tid, score = pick[0], pick[1]
            if tid not in picks or score < picks[tid][0]:
                if tid in picks:
                    audit["dropped_host_conflict"] += 1
                picks[tid] = (score, k)
            else:
                audit["dropped_host_conflict"] += 1
        for tid, (_score, k) in picks.items():
            foot, box, conf, _role = st.discards[t][k]
            gi = st.gt_of_discard.get((t, k))
            miss = gi is not None and gi not in st.gt_covered[t]
            if cap is not None and per_track[tid] >= cap:
                audit["capped"] += 1
                continue
            row = _attached_row(st, t, tid, foot, box, conf, len(rows))
            if row is None:
                audit["no_homography"] += 1
                continue
            per_track[tid] += 1
            rows.append(row)
            if gi is None:
                audit["attached_onto_gt_null"] += 1
            elif not miss:
                audit["attached_onto_covered_gt"] += 1
            elif st.track_gt.get(tid) != int(st.gt_rows[t][gi]["track_id"]):
                audit["attached_wrong_identity"] += 1
            else:
                g = st.gt_rows[t][gi]
                ga = g["attributes"] or {}
                want = effective_attrs(ga.get("role"), ga.get("team"), ga.get("jersey"))
                audit["attached_correct" if st.track_attrs(tid) == want
                      else "attached_right_id_wrong_attrs"] += 1
    return rows, audit


def write_arm(st: SeqState, src: Path, dest: Path, rows: list[dict]) -> None:
    """Write the control submission plus the attached rows to ``dest``."""
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["predictions"] = payload["predictions"] + rows
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")


# === stages =======================================================================================
def _states(arm_dir: Path, data_dir: Path, positions: Path, lowconf: Path,
            seqs: list[str]):
    """Yield prepared sequence states one at a time (they are large)."""
    data = arm_dir / "predictions" / "data"
    for i, s in enumerate(seqs):
        logger.info("[%d/%d] %s", i + 1, len(seqs), s)
        yield s, SeqState(data_dir / s, data / f"{s}.json", positions, lowconf)


def run_oracle(arm_dir: Path, data_dir: Path, positions: Path, lowconf: Path, out: Path) -> dict:
    """Stage 1: the attach census over every sequence the control holds."""
    seqs = sorted(p.stem for p in (arm_dir / "predictions" / "data").glob("*.json"))
    per_seq: dict[str, dict] = {}
    total: Counter = Counter()
    for s, st in _states(arm_dir, data_dir, positions, lowconf, seqs):
        cts = analyse_sequence(st)
        per_seq[s] = cts
        total.update(cts)
        logger.info("  det=%d pos=%d discard=%d miss=%d attach@g10=%d named@g10=%d",
                    st.n_det, st.n_pos, cts.get("discard", 0), cts.get("cls::gt_miss", 0),
                    cts.get("attachable::g10", 0), cts.get("host_named::g10", 0))
    payload = {"control": str(arm_dir), "positions": str(positions), "lowconf": str(lowconf),
               "seqs": seqs, "total": dict(total), "per_seq": per_seq}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info("wrote %s", out)
    return payload


#: Counterfactual arms priced by ``--stage score``: ``(name, mode, gap, tau, cap)``.
SCORE_ARMS: tuple[tuple[str, str, int, float, float, float, int | None], ...] = (
    ("oracle_g10", "oracle", 10, 0.0, 0.0, 0.0, None),
    ("oracle_g30", "oracle", 30, 0.0, 0.0, 0.0, None),
    ("rule_g3_t0.25", "rule", 3, 0.25, 0.0, 0.0, None),
    ("rule_g10_t0.5", "rule", 10, 0.5, 0.0, 0.0, None),
    ("rule_g10_t0.5_s0.6", "rule", 10, 0.5, 0.6, 0.0, None),
    ("rule_g30_t1.0_s0.3", "rule", 30, 1.0, 0.3, 0.0, None),
    ("rule_g3_t0.1_s1.0", "rule", 3, 0.1, 1.0, 0.0, None),
    ("rule_g30_t1.0_s0.3_pan8", "rule", 30, 1.0, 0.3, 8.0, None),
    ("rule_g30_t1.0_pan8", "rule", 30, 1.0, 0.0, 8.0, None),
)


def run_score(arm_dir: Path, data_dir: Path, positions: Path, lowconf: Path, work: Path,
              out: Path, arms: tuple = SCORE_ARMS, states: tuple[str, ...] = ("off", "on")) -> dict:
    """Stage 2: build every counterfactual submission and score it officially against the control.

    Both flag states of the shipped post-processing stack (v9 W3 vote + v9 W4 gk_side) are applied
    to the control and to every arm, exactly as :func:`tools.gsr_v9_w7.score_pair` does.
    """
    from scipy.stats import wilcoxon  # noqa: PLC0415

    from eval.gsr_score import EVAL_CONFIGS, gs_hota  # noqa: PLC0415
    from tools.gsr_v9_w7 import FLAGS_OFF, FLAGS_ON, _prepare  # noqa: PLC0415

    src = arm_dir / "predictions" / "data"
    seqs = sorted(p.stem for p in src.glob("*.json"))
    audits: dict[str, Counter] = {name: Counter() for name, *_ in arms}
    for s, st in _states(arm_dir, data_dir, positions, lowconf, seqs):
        for name, mode, gap, tau, sep_min, pan_min, cap in arms:
            rows, audit = build_attach_arm(st, mode, gap, tau, sep_min, pan_min, cap)
            audits[name].update(audit)
            audits[name]["rows"] += len(rows)
            write_arm(st, src / f"{s}.json", work / "raw" / name / f"{s}.json", rows)
    res: dict[str, dict] = {"seqs": seqs, "audit": {n: dict(audits[n]) for n, *_ in arms}}
    for state in states:
        flags = FLAGS_ON if state == "on" else FLAGS_OFF
        shutil.rmtree(work / f"{state}_control", ignore_errors=True)
        _prepare(src, work / f"{state}_control", seqs, flags)
        base = gs_hota(work / f"{state}_control", data_dir, seq_info={s: 0 for s in seqs},
                       **EVAL_CONFIGS["gs_hota_full"])
        out_state: dict[str, dict] = {"control": {"combined": base["combined"],
                                                  "per_seq": base["per_seq"]}}
        logger.info("[%s] control GS-DetA %.4f GS-HOTA %.4f", state,
                    base["combined"]["GS-DetA"], base["combined"]["GS-HOTA"])
        for name, *_ in arms:
            shutil.rmtree(work / f"{state}_{name}", ignore_errors=True)
            _prepare(work / "raw" / name, work / f"{state}_{name}", seqs, flags)
            r = gs_hota(work / f"{state}_{name}", data_dir, seq_info={s: 0 for s in seqs},
                        **EVAL_CONFIGS["gs_hota_full"])
            paired = {}
            for key in ("GS-HOTA", "GS-DetA", "GS-AssA"):
                d = np.array([r["per_seq"][s][key] - base["per_seq"][s][key] for s in seqs])
                paired[key] = {"mean": float(d.mean()), "helped": int((d > 0).sum()),
                               "hurt": int((d < 0).sum()),
                               "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
            out_state[name] = {
                "combined": r["combined"], "per_seq": r["per_seq"],
                "delta": {k: r["combined"][k] - base["combined"][k]
                          for k in base["combined"] if isinstance(base["combined"][k],
                                                                  (int, float))},
                "paired": paired}
            logger.info("[%s] %-16s DetA %+.4f HOTA %+.4f rows %d %s", state, name,
                        out_state[name]["delta"]["GS-DetA"], out_state[name]["delta"]["GS-HOTA"],
                        audits[name]["rows"], dict(audits[name]))
        res[state] = out_state
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    logger.info("wrote %s", out)
    return res


# === self-check ===================================================================================
def demo() -> None:
    """Assert the pure seams: foot points, tolerance matching, pan estimate, homography."""
    r = {"bbox_image": {"x": 10.0, "y": 20.0, "x_center": 18.0, "y_center": 40.0, "w": 16.0,
                        "h": 40.0},
         "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0}}
    assert sub_foot(r) == (18.0, 60.0)
    assert sub_pitch(r) == (1.0, 2.0)
    # pure pan: every track moves by the same vector -> that vector is the estimate
    prev = {1: (0.0, 0.0), 2: (10.0, 10.0), 3: (5.0, 7.0)}
    cur = {1: (3.0, 1.0), 2: (13.0, 11.0), 3: (8.0, 8.0)}
    assert pan_shift(prev, cur) == (3.0, 1.0)
    assert pan_shift({}, cur) == (0.0, 0.0)
    # one outlier (a real runner) must not move the median
    cur[3] = (60.0, 8.0)
    assert pan_shift(prev, cur) == (3.0, 1.0)
    img = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    pitch = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    h = fit_homography(img, pitch)
    assert h is not None
    assert np.allclose(apply_homography(h, np.array([[50.0, 25.0]])), [[5.0, 2.5]], atol=1e-6)
    assert fit_homography(img[:3], pitch[:3]) is None
    assert _bucket(0.5, (1.0, 3.0)) == "0-1"
    assert _bucket(9.0, (1.0, 3.0)) == ">=3"
    print("gsr_v9_w8 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("oracle", "score"), default="oracle")
    ap.add_argument("--arm-dir", type=Path,
                    default=Path("outputs/gsr/v9_w7/deleak_v6dev_w7ctl_clip_e0.3r1w0.5a0.3"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--positions", type=Path, default=Path("outputs/gsr/v9_w7/positions_w7ctl"))
    ap.add_argument("--lowconf", type=Path, default=Path("outputs/gsr/v9_w7/lowconf_s4b"))
    ap.add_argument("--work", type=Path, default=Path("outputs/gsr/v9_w8/cf"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/gsr_benchmark/gsr_v9_w8_oracle.json"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if args.stage == "score":
        run_score(args.arm_dir, args.data_dir, args.positions, args.lowconf, args.work, args.out)
        return
    run_oracle(args.arm_dir, args.data_dir, args.positions, args.lowconf, args.out)


if __name__ == "__main__":
    main()
