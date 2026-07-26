"""Face feasibility + unused-footage harvest assessment for player attribution.

Two measurements, no system built:

**FACE.** Faces are the obvious "extra source of identity" nobody has measured on this footage.
Face recognition generally needs roughly :data:`FACE_USABLE_PX` pixels of face height. This module
measures, rather than assumes, three things:

1. the head-region pixel height at CARRIER moments, from the box geometry the pipeline already uses
   (:func:`generator.team_anchor.estimate_player_box`, head ~= :data:`HEAD_FRAC` of standing height);
2. the actual face-detection rate of an already-installed detector (OpenCV Haar frontal + profile --
   no new dependency) on real carrier crops decoded from the chunk videos;
3. the same detector on real CLOSE-UP crops, split by whether the jersey-number reader succeeded --
   i.e. "would a face add anchors the number reader misses?".

For (3) the close-up crops of ``brighton_manutd h1_chunk_000`` are still on disk from the original
probe run (``results/closeup_anchor_probe/_tmp_spot``) and are row-aligned with that chunk's rows of
``results/closeup_anchor_probe/closeup_reads.csv``, so the pass/fail of the number read is known per
crop for free. Alignment is asserted (crop JPEG height == recorded ``box_h``).

**HARVEST.** How much additional reference material sits in footage we already own. Every eligible
close-up crop of a full match is already logged in ``closeup_reads.csv`` with its box height and the
recogniser's confidence, so the count of high-legibility close-ups that fall OUTSIDE the current
anchor gate is a table lookup, not a new run.

Run (CPU; stage ``carrier_face`` decodes chunk videos, so do not run it under a GPU job)::

    python -m tools.face_harvest_probe --stage closeup_face
    python -m tools.face_harvest_probe --stage carrier_face
    python -m tools.face_harvest_probe --stage harvest
    python -m tools.face_harvest_probe --stage selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tools.carrier_attribution_probe import OUT_DIR, PROBE_MATCHES, _video_for

REPORT_DIR = Path("results/cross_match_gallery")
CLOSEUP_DIR = Path("results/closeup_anchor_probe")
#: Row-aligned leftovers of the original close-up probe: every eligible crop of one chunk.
SPOT_CROPS = CLOSEUP_DIR / "_tmp_spot"
SPOT_MATCH, SPOT_CHUNK = "brighton_manutd", "h1_chunk_000"
#: Anchor gate used by ``tools.closeup_anchor_probe`` (frozen by its own spot-check).
ANCHOR_CONF = 0.70
#: Crop-eligibility floor of the close-up probe (person-box height in px).
MIN_BOX_H = 100.0
#: Sampled-frame gap above which two close-up frames belong to different camera shots.
SHOT_GAP_FRAMES = 25

#: Head height as a fraction of standing height (~1/8 is the standard adult figure).
HEAD_FRAC = 1.0 / 8.0
#: Face height (chin to hairline) as a fraction of standing height.
FACE_FRAC = 1.0 / 9.0
#: Pixel height a face generally needs before recognition is worth attempting.
FACE_USABLE_PX = 50.0
#: Carrier crops sampled per match for the real-detector measurement.
CARRIER_SAMPLE = 400
#: Haar upscale factor -- tiny crops are enlarged so the 20x20 minimum detector window can fire.
HAAR_UPSCALE = 3


def _cascades() -> list[cv2.CascadeClassifier]:
    """Frontal + profile Haar cascades shipped with the installed OpenCV (no new dependency)."""
    root = Path(cv2.data.haarcascades)
    return [cv2.CascadeClassifier(str(root / n)) for n in
            ("haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml")]


def detect_face(bgr: np.ndarray, cascades: list[cv2.CascadeClassifier]) -> float:
    """Height in ORIGINAL-crop pixels of the largest face found, or ``0.0``.

    The crop is upscaled by :data:`HAAR_UPSCALE` first, because Haar's minimum detection window is
    20x20 px and a face on this footage is often smaller than that; the returned height is scaled
    back down so it is comparable with the geometric estimate.

    Args:
        bgr: crop in BGR.
        cascades: loaded cascades to try in order.

    Returns:
        Largest detected face height in original-crop pixels (``0.0`` when nothing is found).
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, None, fx=HAAR_UPSCALE, fy=HAAR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    grey = cv2.equalizeHist(grey)
    best = 0.0
    for cas in cascades:
        for _, _, _, h in cas.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4,
                                               minSize=(20, 20)):
            best = max(best, float(h) / HAAR_UPSCALE)
    return best


def _pct(series: pd.Series) -> dict:
    q = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {k: round(float(q[k]), 1) for k in ("min", "10%", "25%", "50%", "75%", "90%", "max")}


# === carrier moments =============================================================================
def carrier_box_heights(match_id: str) -> pd.DataFrame:
    """Geometric box / head / face pixel heights for every resolvable carrier of a match."""
    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    ev = pd.read_parquet(OUT_DIR / f"{match_id}_events.parquet")
    q = ev[ev["resolvable"]].reset_index(drop=True)
    chunk = str(q["chunk"].iloc[0])
    cap = cv2.VideoCapture(str(_video_for(match_id, chunk)))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    cap.release()
    box_h = np.array([estimate_player_box(x, y, fh, fw)[3]
                      - estimate_player_box(x, y, fh, fw)[1]
                      for x, y in zip(q["image_x"], q["image_y"])], float)
    return pd.DataFrame({"match": match_id, "chunk": q["chunk"], "frame": q["frame"],
                         "image_x": q["image_x"], "image_y": q["image_y"], "box_h": box_h,
                         "head_px": box_h * HEAD_FRAC, "face_px": box_h * FACE_FRAC})


def stage_carrier_face(sample: int = CARRIER_SAMPLE, seed: int = 7) -> pd.DataFrame:
    """Geometric head size on ALL carriers + real Haar detection on a sample of carrier crops."""
    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    geo = pd.concat([carrier_box_heights(m) for m in PROBE_MATCHES], ignore_index=True)
    geo.to_csv(REPORT_DIR / "carrier_head_geometry.csv", index=False, encoding="utf-8")
    print(f"carrier moments: {len(geo)}")
    print(f"  box height px  : {_pct(geo['box_h'])}")
    print(f"  head height px : {_pct(geo['head_px'])}")
    print(f"  face height px : {_pct(geo['face_px'])}")
    print(f"  face >= {FACE_USABLE_PX:.0f} px : "
          f"{(geo['face_px'] >= FACE_USABLE_PX).mean():.4f} "
          f"({int((geo['face_px'] >= FACE_USABLE_PX).sum())}/{len(geo)})")
    rng = np.random.default_rng(seed)
    cascades = _cascades()
    rows: list[dict] = []
    for mid, g in geo.groupby("match"):
        take = g.iloc[rng.choice(len(g), size=min(sample, len(g)), replace=False)]
        for chunk_key, gg in take.groupby("chunk"):
            cap = cv2.VideoCapture(str(_video_for(str(mid), str(chunk_key))))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            for r in gg.sort_values("frame").itertuples(index=False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
                ok, bgr = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
                crop = bgr[y1:y2, x1:x2]
                rows.append({"match": mid, "chunk": chunk_key, "frame": int(r.frame),
                             "box_h": float(y2 - y1), "face_px_geo": r.face_px,
                             "face_px_det": detect_face(crop, cascades)})
            cap.release()
        print(f"  {mid}: {len(rows)} crops probed")
    det = pd.DataFrame(rows)
    det.to_csv(REPORT_DIR / "carrier_face_detection.csv", index=False, encoding="utf-8")
    hit = det["face_px_det"] > 0
    print(f"carrier crops probed with Haar: {len(det)}")
    print(f"  any face detected     : {hit.mean():.4f} ({int(hit.sum())}/{len(det)})")
    if hit.any():
        print(f"  detected face height  : {_pct(det.loc[hit, 'face_px_det'])}")
        big = det["face_px_det"] >= FACE_USABLE_PX
        print(f"  detected >= {FACE_USABLE_PX:.0f} px  : {int(big.sum())}/{len(det)}")
    return det


# === close-up crops ==============================================================================
def stage_closeup_face() -> pd.DataFrame:
    """Haar faces on the retained close-up crops, split by jersey-number-read pass/fail."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reads = pd.read_csv(CLOSEUP_DIR / "closeup_reads.csv")
    chunk = reads[reads["chunk"] == SPOT_CHUNK].reset_index(drop=True)
    files = sorted(SPOT_CROPS.glob("crop_*.jpg"))
    assert len(files) == len(chunk), f"crop/row misalign {len(files)} vs {len(chunk)}"
    cascades = _cascades()
    rows: list[dict] = []
    for i, path in enumerate(files):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        r = chunk.loc[i]
        assert abs(bgr.shape[0] - float(r["box_h"])) < 1.5, f"row {i} height mismatch"
        rows.append({"i": i, "box_h": float(r["box_h"]), "pred": int(r["pred"]),
                     "conf": float(r["conf"]), "illegible": bool(r["illegible"]),
                     "face_px_det": detect_face(bgr, cascades),
                     "face_px_geo": float(r["box_h"]) * FACE_FRAC})
    det = pd.DataFrame(rows)
    det["anchor"] = (det["conf"] >= ANCHOR_CONF) & ~det["illegible"]
    det["face"] = det["face_px_det"] > 0
    det["face_usable"] = det["face_px_det"] >= FACE_USABLE_PX
    det.to_csv(REPORT_DIR / "closeup_face_detection.csv", index=False, encoding="utf-8")
    print(f"close-up crops ({SPOT_MATCH} {SPOT_CHUNK}): {len(det)}")
    print(f"  box height px        : {_pct(det['box_h'])}")
    print(f"  geometric face px    : {_pct(det['face_px_geo'])}")
    print(f"  any face detected    : {det['face'].mean():.4f} "
          f"({int(det['face'].sum())}/{len(det)})")
    if det["face"].any():
        print(f"  detected face height : {_pct(det.loc[det['face'], 'face_px_det'])}")
    print(f"  face >= {FACE_USABLE_PX:.0f} px       : {det['face_usable'].mean():.4f} "
          f"({int(det['face_usable'].sum())}/{len(det)})")
    tab = pd.crosstab(det["anchor"], det["face"], margins=True)
    print("crops: rows = number read is an ANCHOR (conf >= "
          f"{ANCHOR_CONF}), cols = face detected")
    print(tab.to_string())
    fail = det[~det["anchor"]]
    print(f"  number-read FAILURES with a face      : {int(fail['face'].sum())}/{len(fail)} "
          f"({fail['face'].mean():.4f})")
    print(f"  number-read FAILURES with >= {FACE_USABLE_PX:.0f} px face: "
          f"{int(fail['face_usable'].sum())}/{len(fail)} ({fail['face_usable'].mean():.4f})")
    return det


# === harvest assessment ==========================================================================
def _shot_ids(reads: pd.DataFrame, gap: int = SHOT_GAP_FRAMES) -> pd.Series:
    """Group sampled close-up frames into camera shots (same chunk, frame gap <= ``gap``)."""
    out: list[pd.Series] = []
    for chunk, g in reads.groupby("chunk"):
        frames = np.sort(g["frame"].unique())
        sid = np.cumsum(np.r_[0, np.diff(frames) > gap])
        lookup = dict(zip(frames, sid))
        out.append(pd.Series([f"{chunk}_{lookup[f]}" for f in g["frame"]], index=g.index))
    return pd.concat(out).sort_index()


def stage_harvest() -> pd.DataFrame:
    """How much close-up reference material sits outside the current anchor gate."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reads = pd.read_csv(CLOSEUP_DIR / "closeup_reads.csv")
    reads["anchor"] = (reads["conf"] >= ANCHOR_CONF) & ~reads["illegible"]
    reads["shot"] = _shot_ids(reads)
    shots = reads.groupby("shot").agg(max_box_h=("box_h", "max"),
                                      anchors=("anchor", "sum")).reset_index()
    rows = []
    for band, lo, hi in (("100-150", 100.0, 150.0), ("150-250", 150.0, 250.0),
                         ("250-400", 250.0, 400.0), ("400+", 400.0, np.inf)):
        g = reads[(reads["box_h"] >= lo) & (reads["box_h"] < hi)]
        rows.append({"box_h_band": band, "crops": len(g),
                     "anchors": int(g["anchor"].sum()),
                     "non_anchor": int((~g["anchor"]).sum()),
                     "legible_flagged": int((~g["illegible"]).sum()),
                     "face_px_geo_median": round(float(g["box_h"].median() * FACE_FRAC), 1)
                     if len(g) else None})
    tab = pd.DataFrame(rows)
    tab.to_csv(REPORT_DIR / "harvest_bands.csv", index=False, encoding="utf-8")
    print(f"{SPOT_MATCH}: {len(reads)} eligible close-up crops "
          f"(box_h >= {MIN_BOX_H:.0f} px) over {reads['chunk'].nunique()} chunks")
    print(tab.to_string(index=False))
    print(f"  anchors (conf >= {ANCHOR_CONF})   : {int(reads['anchor'].sum())} "
          f"({reads['anchor'].mean():.3f})")
    print(f"  NON-anchor eligible crops : {int((~reads['anchor']).sum())}")
    print(f"close-up SHOTS: {len(shots)} (gap > {SHOT_GAP_FRAMES} sampled frames = new shot)")
    for lo in (100.0, 250.0, 400.0, 600.0):
        b = shots[shots["max_box_h"] >= lo]
        print(f"  shots containing a >= {lo:.0f} px person crop: {len(b):4d}   "
              f"of which ZERO anchors: {int((b['anchors'] == 0).sum())}")
    gal = pd.read_parquet(OUT_DIR / f"{PROBE_MATCHES[0]}_gallery.parquet")
    print(f"  for scale, wide-shot gallery crops in {PROBE_MATCHES[0]}: {len(gal)}")
    return tab


def _selftest() -> None:
    """Smallest runnable check: the detector wrapper and the band table are sane."""
    cascades = _cascades()
    assert all(not c.empty() for c in cascades), "haar cascades failed to load"
    assert detect_face(np.zeros((0, 0, 3), np.uint8), cascades) == 0.0
    assert detect_face(np.full((120, 60, 3), 128, np.uint8), cascades) == 0.0  # flat grey: no face
    box = 100.0
    assert abs(box * FACE_FRAC - 11.111) < 1e-2
    print("selftest ok")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["carrier_face", "closeup_face", "harvest", "selftest"])
    ap.add_argument("--sample", type=int, default=CARRIER_SAMPLE)
    a = ap.parse_args()
    if a.stage == "selftest":
        _selftest()
    elif a.stage == "carrier_face":
        stage_carrier_face(a.sample)
    elif a.stage == "closeup_face":
        stage_closeup_face()
    elif a.stage == "harvest":
        stage_harvest()


if __name__ == "__main__":
    main()
