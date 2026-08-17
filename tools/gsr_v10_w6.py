"""v10 W6 -- freeze, gate and package the arm-D bundle.

The session driver for the v10 freeze. Every arm here is built by :mod:`tools.gsr_v10_w5` (arm D
is that module's registered composition, unchanged); this file only routes it at a chosen candidate
cache / sequence split, pairs two chains against each other, and writes the freeze record.

Stages (each is its own CLI flag, run one at a time -- one process per work directory)::

    --positions   CPU. Arm homographies from a candidate cache -> positions parquets.
    --chain       CPU. tools.gsr_eiou.run_point on those positions (EIoU -> connector -> solver).
    --pair        CPU. tools.gsr_v9_w7.score_pair of two chain outputs, flags ON and OFF.
    --freeze      Writes results/gsr_v10_frozen.json. MUST run before any TEST-38 read.
    --package     test-49 tail: legitimacy audit, zip, zip self-score.

The control an arm pairs against is named explicitly (``--ctrl-tag``), because this session needs
two different controls: the on-record W5 lineage for the DEV s4 sub-gate, and the WITHIN-CACHE
control (the freeze cache's own gate pick) for the TEST-38 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: Splits, read from the v9-W9 records so the freeze cannot drift from the on-record lineage.
_W9 = Path("results/gsr_benchmark")

#: On-record v9 numbers the within-cache control is reported against (informational drift).
ON_RECORD = {
    "t38": {"off": 49.4970, "full": 53.1978},
    "test49": {"off": 53.0846, "full": 55.4062},
}


def seqs_for(split: str) -> list[str]:
    """Sequence names of a split (``dev20`` / ``t38`` / ``test49``)."""
    from tools.gsr_v10_w5 import DEV20  # noqa: PLC0415

    if split == "dev20":
        return list(DEV20)
    stem = {"t38": "gsr_v9_w9_t38", "test49": "gsr_v9_w9_test49"}[split]
    return json.loads((_W9 / f"{stem}.json").read_text(encoding="utf-8"))["seqs"]


def _md5(path: Path) -> str:
    """Hex md5 of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def _sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_positions(out_dir: Path, cache_dir: Path, base_positions: str, seqs: list[str],
                    arms: list[str], tag: str) -> dict:
    """Arm homographies from ``cache_dir`` -> ``positions_v10w5<arm><tag>_v6det`` parquets.

    Unlike :func:`tools.gsr_v10_w5.score_arms`, the ``ctrl`` arm here is ALSO written as its own
    positions set: the freeze gate pairs against the cache's own gate pick (the within-cache
    control), not against the on-record shipped positions.

    Args:
        out_dir: Artifact root (``outputs/gsr`` for DEV, ``outputs/gsr_srv`` for TEST-38).
        cache_dir: Candidate cache holding one ``<SEQ>.npz`` per sequence.
        base_positions: Positions subdir supplying detections / foot points.
        seqs: Sequences to build.
        arms: Arm names accepted by :func:`tools.gsr_v10_w5.arm_homographies`.
        tag: Artifact suffix, e.g. ``"_t38"``.

    Returns:
        ``{arm: {seq: stats}}``.
    """
    import tools.gsr_v10_w5 as w5  # noqa: PLC0415

    w5.POSITIONS_SUBDIR = base_positions
    return w5.build_positions(out_dir, cache_dir, seqs, arms, tag)


def chain(data_dir: Path, out_dir: Path, seqs: list[str], arm: str, tag: str) -> dict:
    """The frozen v6 chain on one arm's positions set; returns its own GS-HOTA read.

    Writes ``deleak_v10w5_<arm><tag>/predictions/data`` -- the submission the pairing scores.
    """
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v6det import TAU  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache_v6det"
    sub = f"positions_v10w5{arm}{tag}_v6det"
    res = eiou.run_point(data_dir, out_dir, seqs,
                         eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                         embedder="clip_v6det", tau=TAU, tag=f"v10w5_{arm}{tag}",
                         percrop_variant="_v6_v6det", positions_subdir=sub)
    h = res["gs_hota"]
    print(f"{arm}{tag:<8} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
          f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f}", flush=True)
    return {"positions_subdir": sub, "gs_hota": h, "gs_hota_per_seq": res["gs_hota_per_seq"]}


def pair(data_dir: Path, out_dir: Path, seqs: list[str], ctrl_tag: str, arm_tag: str,
         work: Path) -> dict:
    """``score_pair`` of two chain outputs named by their full ``<arm><tag>`` suffixes."""
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    ctrl = out_dir / f"deleak_v10w5_{ctrl_tag}" / "predictions" / "data"
    arm = out_dir / f"deleak_v10w5_{arm_tag}" / "predictions" / "data"
    for p in (ctrl, arm):
        missing = [s for s in seqs if not (p / f"{s}.json").exists()]
        if missing:
            raise FileNotFoundError(f"{p}: missing {len(missing)} sequences, first {missing[:3]}")
    return score_pair(ctrl, arm, data_dir, work, seqs)


def gate_verdict(paired: dict, *, min_mean: float, min_helped: int, max_p: float,
                 n: int) -> dict:
    """Apply the registered TEST-38 bars to a ``score_pair`` result (flags ON)."""
    on = paired["on"]["paired"]
    h, d = on["GS-HOTA"], on["GS-DetA"]
    checks = {
        "mean_gs_hota": {"value": h["mean"], "bar": min_mean, "pass": h["mean"] >= min_mean},
        "helped": {"value": h["helped"], "bar": min_helped, "n": n,
                   "pass": h["helped"] >= min_helped},
        "wilcoxon_p": {"value": h["wilcoxon_p"], "bar": max_p, "pass": h["wilcoxon_p"] < max_p},
        "det_a_not_worse": {"value": d["mean"], "bar": 0.0, "pass": d["mean"] >= 0.0},
    }
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def cache_lineage(cache_dirs: dict[str, Path], seqs: list[str]) -> dict:
    """Per-sequence cache provenance: size, mtime and md5, so ctrl and arm can be proved co-sourced.

    Args:
        cache_dirs: ``{label: directory}`` of candidate caches.
        seqs: Sequences the freeze covers.

    Returns:
        ``{label: {"n": .., "missing": [..], "per_seq": {seq: {...}}}}``.
    """
    out: dict[str, dict] = {}
    for label, d in cache_dirs.items():
        rec, missing = {}, []
        for s in seqs:
            p = d / f"{s}.npz"
            if not p.exists():
                missing.append(s)
                continue
            st = p.stat()
            rec[s] = {"bytes": st.st_size, "md5": _md5(p),
                      "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                      .isoformat(timespec="seconds")}
        out[label] = {"dir": str(d), "n": len(rec), "missing": missing, "per_seq": rec}
    return out


def freeze(bundle_extra: dict, lineage: dict, dev: dict, gate: dict,
           dest: Path = Path("results/gsr_v10_frozen.json")) -> dict:
    """Write the v10 freeze record. Called BEFORE any TEST-38 number of this session exists.

    Args:
        bundle_extra: What v10 adds on top of the frozen v9 bundle (arm D, s4 verdict).
        lineage: Output of :func:`cache_lineage` plus any narrative provenance.
        dev: The DEV evidence the bundle was chosen on.
        gate: The pre-declared TEST-38 gate.
        dest: Where the record is written.

    Returns:
        The written record.
    """
    import numpy  # noqa: PLC0415
    import pandas  # noqa: PLC0415
    import scipy  # noqa: PLC0415

    v9 = json.loads(Path("results/gsr_v9_frozen.json").read_text(encoding="utf-8"))
    code = [Path(p) for p in ("eval/gsr_score.py", "tools/gsr_deleak.py", "tools/gsr_eiou.py",
                              "tools/gsr_v6det.py", "tools/gsr_v10_w2_calib.py",
                              "tools/gsr_v10_w5.py", "tools/gsr_v10_w6.py",
                              "generator/camera_track.py", "generator/calibrate.py")]
    weights = {k: v for k, v in (("detector", v9["bundle"]["detector"]),
                                 ("jersey_reader", v9["bundle"]["jersey_reader"]))}
    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "v10-1.0",
        "session": "v10 W6 (freeze / gate / package)",
        "parent": {"file": "results/gsr_v9_frozen.json",
                   "declared_at": v9["declared_at"],
                   "sha256": _sha256(Path("results/gsr_v9_frozen.json"))},
        "bundle": {"v9_chain": v9["bundle"]["chain"],
                   "v9_flags": {"VOTE_TRACK_ATTRS": v9["bundle"]["new_in_v9"]["VOTE_TRACK_ATTRS"],
                                "GK_SIDE_REPAIR": v9["bundle"]["new_in_v9"]["GK_SIDE_REPAIR"],
                                "call_site": v9["bundle"]["new_in_v9"]["call_site"]},
                   "weights": weights, **bundle_extra},
        "code_sha256": {str(p): _sha256(p) for p in code if p.exists()},
        "environment": {
            "scoring_host": __import__("socket").gethostname(),
            "platform": __import__("platform").platform(),
            "python": __import__("platform").python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "trackeval": v9["environment"]["trackeval"]},
        "lineage": lineage,
        "dev_evidence": dev,
        "pre_declared_gate": gate,
        "note": "declared BEFORE any TEST-38 GS-HOTA of the v10 bundle existed; test-49 is "
                "untouched at declaration time.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"froze v10 bundle -> {dest} ({record['declared_at']})")
    return record


def package(data_dir: Path, out_dir: Path, seqs: list[str], arm_tag: str, stem: str) -> dict:
    """test-49 tail: gt-free audit of the packaged chain, zip, zip self-score.

    Args:
        data_dir: GSR root.
        out_dir: Artifact root the chain was run in.
        seqs: The 49 test sequences.
        arm_tag: ``<arm><tag>`` of the chain to package (e.g. ``"D_test49"``).
        stem: Package stem, e.g. ``"gtfree_v10"``.

    Returns:
        ``{"legitimacy": ..., "manifest": ..., "zip_selfscore": ...}``.
    """
    import eval.gsr_score as gsc  # noqa: PLC0415

    from tools.gsr_calibfill import verify_gtfree  # noqa: PLC0415
    from tools.gsr_calibgate import zip_selfscore  # noqa: PLC0415
    from tools.gsr_deleak import package_free  # noqa: PLC0415
    from tools.gsr_v9_w7 import FLAGS_ON, _prepare  # noqa: PLC0415

    src = out_dir / f"deleak_v10w5_{arm_tag}"
    flagged = out_dir / f"v10_w6_pkg_{arm_tag}"
    changed = _prepare(src / "predictions" / "data", flagged, seqs, FLAGS_ON)
    gta = sorted(out_dir.glob("eval_v4_gta_*_v6_v6det_eiou_*"))
    if len(gta) != 1:
        raise FileNotFoundError(f"expected one GTA base arm in {out_dir}, found {gta}")
    frozen = json.loads(Path("results/gsr_v10_frozen.json").read_text(encoding="utf-8"))
    # The audit must reproduce EXACTLY what the zip holds, so the two v9 flags are switched on for
    # it: the packaged rows are the base arm put through vote -> keeper repair, nothing else.
    gsc.VOTE_TRACK_ATTRS = tuple(FLAGS_ON["vote"])
    gsc.GK_SIDE_REPAIR = tuple(FLAGS_ON["gk"])
    out: dict = {"flag_rows_changed": dict(changed),
                 "audited_dir": str(flagged),
                 "legitimacy": verify_gtfree(flagged, gta[0], data_dir,
                                             out_dir / "positions_gate_v6det", seqs)}
    out["manifest"] = package_free(flagged, seqs, frozen, stem=stem)
    zip_path = Path(out["manifest"]["zip"])
    out["zip_selfscore"] = zip_selfscore(zip_path, data_dir, seqs,
                                         zip_path.parent / f"zip_selfscore_{stem}.json")[
                                             "combined"]
    print("legitimacy:", out["legitimacy"])
    print("zip self-score:", out["zip_selfscore"])
    return out


def _demo() -> None:
    """Assert the two pieces of logic this file actually owns."""
    ok = {"on": {"paired": {"GS-HOTA": {"mean": 1.4, "helped": 26, "wilcoxon_p": 0.002},
                            "GS-DetA": {"mean": 0.6}}}}
    v = gate_verdict(ok, min_mean=1.0, min_helped=24, max_p=0.01, n=38)
    assert v["pass"], v
    bad = {"on": {"paired": {"GS-HOTA": {"mean": 1.4, "helped": 23, "wilcoxon_p": 0.002},
                             "GS-DetA": {"mean": -0.1}}}}
    v = gate_verdict(bad, min_mean=1.0, min_helped=24, max_p=0.01, n=38)
    assert not v["pass"] and not v["checks"]["helped"]["pass"] \
        and not v["checks"]["det_a_not_worse"]["pass"], v
    assert len(seqs_for("t38")) == 38 and len(seqs_for("test49")) == 49
    assert len(seqs_for("dev20")) == 20
    print("gsr_v10_w6 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--base-positions", default="positions_v10w2ctrl_v6det")
    ap.add_argument("--split", choices=("dev20", "t38", "test49"), default="dev20")
    ap.add_argument("--seqs", default=None, help="override the split's sequence list")
    ap.add_argument("--arms", default="D")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ctrl-tag", default=None, help="<arm><tag> of the control chain to pair with")
    ap.add_argument("--results", type=Path, default=Path("results/gsr_benchmark/gsr_v10_w6.json"))
    ap.add_argument("--key", default=None, help="key this stage writes under in --results")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--lineage", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else seqs_for(args.split)
    arms = args.arms.split(",")
    payload: dict = {}
    t0 = time.time()
    if args.positions:
        payload["positions"] = build_positions(args.out_dir, args.cache_dir, args.base_positions,
                                               seqs, arms, args.tag)
    if args.chain:
        payload["chain"] = {a: chain(args.data_dir, args.out_dir, seqs, a, args.tag) for a in arms}
    if args.pair:
        payload["pair"] = {
            f"{a}{args.tag}": pair(args.data_dir, args.out_dir, seqs, args.ctrl_tag,
                                   f"{a}{args.tag}",
                                   args.out_dir / "v10_w6" / "pair" / f"{a}{args.tag}")
            for a in arms}
        for k, v in payload["pair"].items():
            for state in ("off", "on"):
                p = v[state]
                print(f"{k} [{state}] ctrl {p['control']['combined']['GS-HOTA']:.4f} -> "
                      f"arm {p['arm']['combined']['GS-HOTA']:.4f}  "
                      f"mean {p['paired']['GS-HOTA']['mean']:+.4f}  "
                      f"{p['paired']['GS-HOTA']['helped']}/{p['paired']['GS-HOTA']['hurt']}  "
                      f"p={p['paired']['GS-HOTA']['wilcoxon_p']:.3g}", flush=True)
    if args.lineage:
        payload["lineage"] = cache_lineage({args.tag or "cache": args.cache_dir}, seqs)
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        key = args.key or (args.tag or "run")
        prev.setdefault(key, {}).update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results}[{key}] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
