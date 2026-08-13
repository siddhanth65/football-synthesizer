"""W1 seam experiment: three tracklet aggregators over ONE per-crop identity source.

Campaign v8, session W1 (`results/GSR_V8_W1.md`). The 61.48 leader names per-crop
``argmax`` + tracklet majority vote as its own bottleneck; this harness grades that aggregation
against our evidential/Bayesian fusion (:func:`generator.evidential_jersey.fuse_tracklet`) and our
votes machinery (:func:`generator.jersey_id.percrop_votes`) on the **same** per-crop probability
vectors, at tracklet level, against GSR jersey ground truth on DEV-20 GT crops.

Two per-crop sources are supported and are graded by identical code:

* **ours** -- the v6 reader (arm-4t PARSeq behind the legibility gate + pose torso RoI), produced by
  ``--reader``; the persisted columns are the standard ``leg`` / ``torso`` / ``p0`` / ``p1``;
* **theirs** -- a ``.npz`` of per-crop ``role`` / ``d1`` / ``d2`` / ``color`` arrays produced OUTSIDE
  this repo by a private instrument (plan v8 claims-hygiene rule: their code and weights never enter
  the repo tree). This module only reads the numbers.

Class layout throughout: ``101`` classes, index 0 = "no number", index ``k + 1`` = jersey number
``k``. Their two 11-way digit heads fold into it through their own ``get_number`` mapping (``100``
is their no-number sentinel); our 100-class rows lift into it with a zero at "number 0".

CLI::

    python -m tools.gsr_w1_seam --reader <out.parquet> --parseq-ckpt <arm4t.ckpt>
    python -m tools.gsr_w1_seam --compare --winner <w.npz> --winner-crops <c.parquet> \
        --reader-parquet <out.parquet> --out results/gsr_benchmark/gsr_v8_w1.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_w1_seam")

#: index 0 = no number, index k+1 = jersey number k (k in 0..99).
N_CLASSES = 101
NO_NUMBER = 0
#: Their no-number sentinel, which propagates through every one of their aggregation functions.
THEIR_SENTINEL = 100


# --------------------------------------------------------------------------------------- sources


def their_probs(npz: Path) -> np.ndarray:
    """Their per-crop digit logits -> ``[N, 101]`` number distributions (pure).

    The two heads are independent softmaxes over ``{0..9, absent}``; the joint over the ``11 x 11``
    cells is mapped through their own ``get_number`` rule (``digit1 == 10`` -> single digit or the
    ``100`` sentinel), so several cells can land on the same number and their masses add. Nothing is
    thresholded here -- that is the point of the session.

    Args:
        npz: Instrument output holding ``d1`` ``[N, 11]`` and ``d2`` ``[N, 11]`` logits.

    Returns:
        ``[N, 101]`` rows summing to 1.
    """
    blob = np.load(npz)
    p1, p2 = _softmax(blob["d1"].astype(np.float64)), _softmax(blob["d2"].astype(np.float64))
    out = np.zeros((len(p1), N_CLASSES))
    for a, b in itertools.product(range(11), range(11)):
        num = (100 if b == 10 else b) if a == 10 else a * 10 + b
        out[:, NO_NUMBER if num == THEIR_SENTINEL else num + 1] += p1[:, a] * p2[:, b]
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row softmax (pure)."""
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def our_probs(df: pd.DataFrame) -> np.ndarray:
    """Our persisted PARSeq positional softmaxes -> ``[N, 101]`` (pure).

    Uses :func:`generator.jersey_id.parseq_positions_to_probs`, the same fold the whole chain uses,
    then lifts the 100-class layout into the shared 101 one (our reader cannot emit "number 0").
    """
    from generator.jersey_id import parseq_positions_to_probs  # noqa: PLC0415

    out = np.zeros((len(df), N_CLASSES))
    out[:, NO_NUMBER] = 1.0
    p0 = np.stack(df["p0"].to_numpy())
    p1 = np.stack(df["p1"].to_numpy())
    ok = np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
    for i in np.flatnonzero(ok):
        folded = parseq_positions_to_probs(p0[i], p1[i])
        out[i, 0] = folded[0]
        out[i, 2:] = folded[1:]  # our class j (1..99) is number j -> index j+1
    return out


# ---------------------------------------------------------------------------- the three families


def majority_reads(probs: np.ndarray, *, sentinel: str = "revote", min_crop_conf: float = 0.0,
                   min_votes: int = 1) -> list[tuple[int, float]]:
    """THEIR aggregation: per-crop argmax, then a tracklet majority vote.

    Reimplemented clean-room from the prose description in ``docs/WINNER_REPO_RECON.md`` §3 (their
    ``majority_vote`` in ``IDATR/refine_tracklets.py`` and ``majority_jersey`` in
    ``write_json_file_team.py``), not copied from their source. The three sentinel rules:

    * ``plain`` -- ``Counter.most_common(1)``, nothing special about the no-number class;
    * ``revote`` -- if the modal value is the no-number sentinel and the votes are not *entirely*
      sentinel, drop the sentinels and re-vote;
    * ``thresh97`` -- the sentinel wins only when it holds ``>= 97%`` of the votes, else it is
      deleted and the runner-up wins.

    ``min_crop_conf`` and ``min_votes`` are knobs their code does NOT have; they are given to this
    family deliberately, so the baseline is stronger than the one they ship.

    Args:
        probs: ``[n_crops, 101]`` per-crop distributions for one tracklet.
        sentinel: ``plain`` / ``revote`` / ``thresh97``.
        min_crop_conf: Floor on the argmax cell's mass for a crop to vote at all.
        min_votes: Minimum votes the winner must hold.

    Returns:
        ``[(number, vote_share)]`` or ``[]`` when the tracklet reads no number.
    """
    if probs.size == 0:
        return []
    idx = probs.argmax(1)
    conf = probs[np.arange(len(probs)), idx]
    idx = idx[conf >= min_crop_conf]
    if not len(idx):
        return []
    votes = Counter(int(i) for i in idx)
    if sentinel == "revote" and votes.most_common(1)[0][0] == NO_NUMBER and len(votes) > 1:
        del votes[NO_NUMBER]
    elif sentinel == "thresh97" and votes[NO_NUMBER] / sum(votes.values()) < 0.97:  # noqa: PLR2004
        votes.pop(NO_NUMBER, None)
    if not votes:
        return []
    best, n = votes.most_common(1)[0]
    if best == NO_NUMBER or n < min_votes:
        return []
    return [(best - 1, n / len(idx))]


def fusion_reads(probs: np.ndarray, *, evidence: str = "soft", scale: float = 100.0,
                 min_conf: float = 0.0, min_crops: int = 1, max_p_none: float = 1.01,
                 max_u: float = 1.01) -> list[tuple[int, float]]:
    """OUR aggregation: uncertainty-filtered additive Dirichlet fusion of the same rows.

    ``fuse_tracklet`` is used unchanged; the only new thing is the adapter that turns a softmax row
    into Dirichlet evidence, ``alpha = 1 + e_i * p_i``:

    * ``soft`` -- ``e_i = 1``: the parameter-free Bayesian soft-count posterior ``Dir(1 + sum_i p_i)``
      (per-crop ``u`` is then a constant, so ``max_u`` is inert by construction);
    * ``conf`` -- ``e_i = scale * max_c p_i(c)``: a diffuse crop contributes less evidence, so
      ``u = K / S`` is a live per-crop channel and the V3 ``max_u`` filter has something to bite on;
    * ``logprob`` -- the classical naive-Bayes product of per-crop distributions (not a Dirichlet;
      reported as the other textbook fusion, never part of the gate).

    Returns:
        ``[(number, fused_confidence)]`` or ``[]``.
    """
    from generator.evidential_jersey import fuse_tracklet  # noqa: PLC0415

    if probs.size == 0:
        return []
    if evidence == "logprob":
        keep = probs[:, NO_NUMBER] < max_p_none
        if int(keep.sum()) < min_crops:
            return []
        lp = np.log(np.clip(probs[keep], 1e-12, None)).sum(0)
        p = _softmax(lp[None, :])[0]
        best = int(p[1:].argmax()) + 1
        if p[NO_NUMBER] >= p[best] or p[best] < min_conf:
            return []
        return [(best - 1, float(p[best]))]
    e = np.ones((len(probs), 1)) if evidence == "soft" else scale * probs.max(1, keepdims=True)
    alpha = 1.0 + e * probs
    out = fuse_tracklet(alpha.astype(np.float64), max_u=max_u, max_p_none=max_p_none,
                        min_conf=min_conf, min_crops=min_crops)
    return [(num - 1, conf) for num, conf in out]


def votes_reads(probs: np.ndarray, *, min_crop_conf: float = 0.50, min_votes: int = 1,
                emit_all: bool = False) -> list[tuple[int, float]]:
    """OUR votes machinery (:func:`generator.jersey_id.percrop_votes`) on the same rows."""
    from generator.jersey_id import percrop_votes  # noqa: PLC0415

    if probs.size == 0:
        return []
    out = percrop_votes(probs, min_crop_conf=min_crop_conf, min_votes=min_votes,
                        emit_all=emit_all)
    return [(num - 1, conf) for num, conf in out]


FAMILIES = {"majority": majority_reads, "fusion": fusion_reads, "votes": votes_reads}


def grid(family: str, *, mode: str = "gate") -> list[dict]:
    """The DEV knob grid for one family (pure).

    ``mode`` selects which grid:

    * ``gate`` -- the pre-registered grid of ``results/GSR_V8_W1.md`` §1.4 (majority 45, fusion 60,
      votes 30). Only these rows may enter the gate statistic;
    * ``ablation`` -- the declared ``max_u`` points for the fusion family, reported outside the gate;
    * ``posthoc`` -- written AFTER the gate was read, to diagnose the fusion family's coverage
      ceiling (the registered ``max_p_none`` grid held only ``{0.3, 1.01}``, an absolute floor that
      cannot express the relative "is the number bigger than the no-number mass" test the other two
      families use). Never gating; labelled ``posthoc`` in the output.
    """
    if family == "majority":
        if mode != "gate":
            return []
        return [{"sentinel": s, "min_crop_conf": c, "min_votes": v}
                for s in ("plain", "revote", "thresh97")
                for c in (0.0, 0.3, 0.5, 0.7, 0.9) for v in (1, 2, 3)]
    if family == "fusion":
        if mode == "posthoc":
            return [{"evidence": e, "min_conf": c, "min_crops": k, "max_p_none": pn, "max_u": u}
                    for e in ("soft", "conf") for c in (0.0, 0.3, 0.5)
                    for k in (1, 2) for pn in (0.4, 0.5, 0.6, 0.7, 0.9)
                    for u in ((1.01, 0.6) if e == "conf" else (1.01,))]
        us = (0.6, 0.8) if mode == "ablation" else (1.01,)
        return [{"evidence": e, "min_conf": c, "min_crops": k, "max_p_none": pn, "max_u": u}
                for e in ("soft", "conf") for c in (0.0, 0.3, 0.5, 0.7, 0.9)
                for k in (1, 2, 3) for pn in (0.3, 1.01)
                for u in (us if e == "conf" else (1.01,))
                if not (mode == "ablation" and e == "soft")]
    if family == "votes":
        if mode != "gate":
            return []
        return [{"min_crop_conf": c, "min_votes": v, "emit_all": a}
                for c in (0.0, 0.3, 0.5, 0.7, 0.9) for v in (1, 2, 3) for a in (False, True)]
    raise ValueError(family)


def disagreement(df: pd.DataFrame, a: np.ndarray, b: np.ndarray) -> dict:
    """Where two per-crop sources agree, differ, and what an oracle union would buy (pure).

    Args:
        df: The shared crop table.
        a: ``[N, 101]`` source A (here: their head).
        b: ``[N, 101]`` source B (here: our reader).

    Returns:
        Per-crop counts on the crops whose GT track carries a number.
    """
    truth = np.array([int(j) if isinstance(j, str) and j else -1 for j in df["jersey"]])
    m = truth > 0
    ra, rb = a.argmax(1) != NO_NUMBER, b.argmax(1) != NO_NUMBER
    na, nb = a[:, 1:].argmax(1), b[:, 1:].argmax(1)
    oka, okb = (na == truth) & ra, (nb == truth) & rb
    both = m & ra & rb
    return {"numbered_crops": int(m.sum()),
            "reads_theirs": int((m & ra).sum()), "reads_ours": int((m & rb).sum()),
            "both_read": int(both.sum()),
            "agree_where_both_read": float((na[both] == nb[both]).mean()),
            "theirs_right": int(oka[m].sum()), "ours_right": int(okb[m].sum()),
            "theirs_only_right": int((oka & ~okb & m).sum()),
            "ours_only_right": int((okb & ~oka & m).sum()),
            "both_right": int((oka & okb & m).sum()),
            "neither_right": int((~oka & ~okb & m).sum()),
            "oracle_union_recall": float(((oka | okb) & m).sum() / max(int(m.sum()), 1))}


# ------------------------------------------------------------------------------------- grading


def grade(df: pd.DataFrame, probs: np.ndarray, family: str, rule: dict,
          roles: tuple[str, ...] = ("player", "goalkeeper")) -> dict:
    """Grade one aggregation rule at TRACKLET level against GSR jersey GT.

    The definitions are :func:`tools.ocr_density.measure`'s verbatim, so every number is comparable
    with ``results/GSR_V7_V3.md`` §2.6: ``d`` = read tracks / tracks, ``read_precision`` = correct
    reads / auditable reads (auditable = the GT track carries a number).

    Args:
        df: Crop table (``seq``, ``track_id``, ``role``, ``jersey``), aligned with ``probs``.
        probs: ``[N, 101]`` per-crop distributions.
        family: ``majority`` / ``fusion`` / ``votes``.
        rule: Keyword arguments for that family's function.
        roles: GT roles forming the graded universe.

    Returns:
        Metrics dict (``d``, ``read_precision``, ``track_precision``, abstention columns, all
        denominators).
    """
    fn = FAMILIES[family]
    keep = df["role"].isin(roles).to_numpy()
    pos = np.flatnonzero(keep)  # groupby(...).indices is positional inside the kept subframe
    jersey = df["jersey"].to_numpy()
    n_tracks = n_read = reads_ok = reads_aud = trk_ok = trk_aud = 0
    unnum_tracks = unnum_emit = 0
    for _key, idx in df[keep].groupby(["seq", "track_id"]).indices.items():
        n_tracks += 1
        out = fn(probs[pos[idx]], **rule)
        true = jersey[pos[idx]][0]
        true = None if not isinstance(true, str) or not true else true
        if true is None:
            unnum_tracks += 1
            unnum_emit += int(bool(out))
        if not out:
            continue
        n_read += 1
        if true is None:
            continue
        trk_aud += 1
        trk_ok += int(str(out[0][0]) == true)
        reads_aud += len(out)
        reads_ok += sum(1 for num, _c in out if str(num) == true)
    return {"family": family, "rule": rule, "n_tracks": n_tracks, "n_tracks_read": n_read,
            "d": n_read / max(n_tracks, 1), "n_reads_auditable": reads_aud,
            "read_precision": reads_ok / max(reads_aud, 1), "n_tracks_auditable": trk_aud,
            "track_precision": trk_ok / max(trk_aud, 1),
            "n_tracks_unnumbered": unnum_tracks,
            "false_emit_rate_unnumbered": unnum_emit / max(unnum_tracks, 1)}


def frontier(rows: list[dict], targets=(0.30, 0.40, 0.50, 0.60)) -> dict[str, dict]:
    """Best read precision reachable at each coverage target (pure) -- the gate's own statistic."""
    out = {}
    for t in targets:
        ok = [r for r in rows if r["d"] >= t]
        out[f"{t:.2f}"] = max(ok, key=lambda r: r["read_precision"]) if ok else None
    return out


def percrop_table(df: pd.DataFrame, probs: np.ndarray, emits=(0.05, 0.10, 0.20, 0.2761)) -> dict:
    """Per-crop precision at matched emit rates (the S2/V3 head-to-head bar).

    Emit is measured on the crops whose GT track carries a number (4,965 on DEV-20), which is the
    denominator ``results/GSR_V7_V3.md`` §2.3's ``0.2761`` was computed against.
    """
    numbered = np.array([isinstance(j, str) and bool(j) for j in df["jersey"]])
    truth = np.array([int(j) if isinstance(j, str) and j else -1 for j in df["jersey"]])
    best = probs[:, 1:].argmax(1)
    conf = probs[np.arange(len(probs)), best + 1]
    reads = (probs.argmax(1) != NO_NUMBER)
    correct = (best == truth) & numbered
    out = {"n_crops": int(len(df)), "n_numbered": int(numbered.sum()),
           "max_emit": float(reads[numbered].mean()),
           "precision_at_max_emit": float(correct[reads & numbered].sum()
                                          / max(int((reads & numbered).sum()), 1)),
           "at_emit": {}}
    order = np.sort(conf[reads & numbered])[::-1]
    for e in emits:
        k = int(round(e * numbered.sum()))
        if k < 1 or k > len(order):
            out["at_emit"][f"{e:.4f}"] = None
            continue
        thr = order[k - 1]
        sel = reads & numbered & (conf >= thr)
        out["at_emit"][f"{e:.4f}"] = {"threshold": float(thr), "n": int(sel.sum()),
                                      "emit": float(sel.sum() / numbered.sum()),
                                      "precision": float(correct[sel].mean())}
    return out


# --------------------------------------------------------------------------------- reader pass


def run_reader(out_parquet: Path, parseq_ckpt: str | None, *, chunk: int = 1200,
               leg_thresh: float = 0.0) -> Path:
    """GPU: run OUR v6 reader over the DEV-20 GT crops and persist the per-crop evidence.

    ``leg_thresh = 0.0`` deliberately disables the external legibility gate so the persisted rows
    cover every crop their head also saw; the shipped ``0.5`` gate is re-applied offline by
    filtering the persisted ``leg`` column, which makes the head-to-head both gated and ungated.
    """
    from eval.gsr_jersey import _build_recognizer  # noqa: PLC0415

    df = dev20_crops()
    root = Path("outputs/gsr/gt_crops/validation")
    paths = [str(root / n) for n in df["name"]]
    recog = _build_recognizer(parseq_ckpt=parseq_ckpt, leg_thresh=leg_thresh)
    leg, torso, p0, p1 = [], [], [], []
    for s in range(0, len(paths), chunk):
        _probs, detail = recog.crop_reads(paths[s : s + chunk])
        leg.append(detail["leg"])
        torso.append(detail["torso"])
        p0.append(detail["p0"])
        p1.append(detail["p1"])
        logger.info("reader %d / %d", min(s + chunk, len(paths)), len(paths))
    df["leg"] = np.concatenate(leg)
    df["torso"] = np.concatenate(torso)
    df["p0"] = list(np.concatenate(p0))
    df["p1"] = list(np.concatenate(p1))
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    return out_parquet


def dev20_crops() -> pd.DataFrame:
    """The DEV-20 GT crop table (the 7,055 crops every reader trial used)."""
    from tools.gsr_crops import plan_split  # noqa: PLC0415
    from tools.jersey_head_trial import dev20_sequences  # noqa: PLC0415

    keep = set(dev20_sequences())
    crops = [c for c in plan_split("validation") if c.seq in keep]
    df = pd.DataFrame({"name": [c.name for c in crops], "seq": [c.seq for c in crops],
                       "track_id": [c.track_id for c in crops], "frame": [c.frame for c in crops],
                       "role": [c.role for c in crops], "team": [c.team for c in crops],
                       "jersey": [c.jersey for c in crops]})
    root = Path("outputs/gsr/gt_crops/validation")
    return df[[(root / n).exists() for n in df["name"]]].reset_index(drop=True)


def _demo() -> None:
    """Self-check on the three aggregators and the grader (the non-trivial logic)."""
    p = np.zeros((5, N_CLASSES))
    p[:3, 8] = 0.9  # three crops read number 7 confidently
    p[:3, NO_NUMBER] = 0.1
    p[3:, NO_NUMBER] = 1.0  # two crops abstain
    assert majority_reads(p, sentinel="plain") == [(7, 0.6)], majority_reads(p, sentinel="plain")
    assert majority_reads(p, sentinel="revote")[0][0] == 7
    assert fusion_reads(p)[0][0] == 7
    assert votes_reads(p)[0][0] == 7
    # sentinel majority: plain abstains, revote rescues the minority read, thresh97 too
    q = np.zeros((5, N_CLASSES))
    q[:2, 8] = 1.0
    q[2:, NO_NUMBER] = 1.0
    assert majority_reads(q, sentinel="plain") == []
    assert majority_reads(q, sentinel="revote") == [(7, 0.4)]
    assert majority_reads(q, sentinel="thresh97") == [(7, 0.4)]
    # an all-sentinel tracklet abstains under every rule
    z = np.zeros((3, N_CLASSES))
    z[:, NO_NUMBER] = 1.0
    assert all(majority_reads(z, sentinel=s) == [] for s in ("plain", "revote", "thresh97"))
    assert fusion_reads(z) == [] and votes_reads(z) == []
    # their digit fold: (d1=10, d2=7) -> number 7, (10, 10) -> no number, (1, 2) -> 12
    import tempfile  # noqa: PLC0415

    lg = np.full((3, 11), -20.0, np.float32)
    lg2 = np.full((3, 11), -20.0, np.float32)
    lg[0, 10], lg2[0, 7] = 20.0, 20.0
    lg[1, 10], lg2[1, 10] = 20.0, 20.0
    lg[2, 1], lg2[2, 2] = 20.0, 20.0
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.npz"
        np.savez(f, d1=lg, d2=lg2)
        got = their_probs(f).argmax(1)
    assert list(got) == [8, NO_NUMBER, 13], list(got)
    assert len(grid("majority")) == 45 and len(grid("fusion")) == 60 and len(grid("votes")) == 30
    assert len(grid("fusion", mode="ablation")) == 60 and len(grid("fusion", mode="posthoc")) == 90
    print("gsr_w1_seam demo OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", default=None, help="run OUR reader, write this parquet")
    ap.add_argument("--parseq-ckpt", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--winner", default=None, help="instrument .npz (outside the repo)")
    ap.add_argument("--winner-crops", default=None)
    ap.add_argument("--reader-parquet", default=None)
    ap.add_argument("--leg-gate", type=float, default=0.5, help="our reader's shipped legibility gate")
    ap.add_argument("--seqs", nargs="*", default=None, help="restrict every source to these sequences")
    ap.add_argument("--out", default="results/gsr_benchmark/gsr_v8_w1.json")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        _demo()
        return
    if a.reader:
        run_reader(Path(a.reader), a.parseq_ckpt)
        return
    if not a.compare:
        ap.error("nothing to do: --reader, --compare or --demo")

    sources: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    if a.winner:
        wdf = pd.read_parquet(a.winner_crops)
        sources["theirs"] = (wdf, their_probs(Path(a.winner)))
    if a.reader_parquet:
        rdf = pd.read_parquet(a.reader_parquet)
        pr = our_probs(rdf)
        gated = pr.copy()
        blocked = rdf["leg"].to_numpy() <= a.leg_gate
        gated[blocked] = 0.0
        gated[blocked, NO_NUMBER] = 1.0
        sources["ours"] = (rdf, gated)
        sources["ours_ungated"] = (rdf, pr)
    if a.seqs:
        sources = {n: (d[d["seq"].isin(a.seqs)].reset_index(drop=True),
                       p[d["seq"].isin(a.seqs).to_numpy()]) for n, (d, p) in sources.items()}
    out: dict = {"sources": {}, "percrop": {}, "frontier": {}, "sweeps": {}, "seqs": a.seqs}
    for name, (df, probs) in sources.items():
        out["sources"][name] = {"n_crops": int(len(df)),
                                "n_tracks": int(df.groupby(["seq", "track_id"]).ngroups)}
        out["percrop"][name] = percrop_table(df, probs)
        rows = []
        for fam in FAMILIES:
            for mode in ("gate", "ablation", "posthoc"):
                for rule in grid(fam, mode=mode):
                    r = grade(df, probs, fam, rule)
                    if mode != "gate":
                        r[mode] = True
                    rows.append(r)
        out["sweeps"][name] = rows
        gated = [r for r in rows if not r.get("posthoc")]
        out["frontier"][name] = {
            fam: frontier([r for r in gated if r["family"] == fam and not r.get("ablation")])
            for fam in FAMILIES}
        out["frontier"][name]["fusion_with_u"] = frontier(
            [r for r in gated if r["family"] == "fusion"])
        out["frontier"][name]["fusion_posthoc"] = frontier(
            [r for r in rows if r["family"] == "fusion"])
    if "theirs" in sources and "ours" in sources:
        out["disagreement"] = disagreement(sources["ours"][0], sources["theirs"][1],
                                           sources["ours"][1])
        out["disagreement_ungated"] = disagreement(sources["ours"][0], sources["theirs"][1],
                                                   sources["ours_ungated"][1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
