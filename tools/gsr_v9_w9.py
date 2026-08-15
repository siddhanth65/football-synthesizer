"""Campaign v9 session W9: freeze the v9 stack, gate it on TEST-38, spend test-49 once.

The v9 bundle is the frozen v6 chain (``results/gsr_v6_frozen.json``) plus exactly two
post-processing flags -- ``eval.gsr_score.VOTE_TRACK_ATTRS = ("role", "team", "jersey")`` (v9 W3)
and ``GK_SIDE_REPAIR = ("team",)`` (v9 W4). Both are *pure functions of a written submission*: they
read our own predictions and rewrite attribute values in place. Nothing upstream of the submission
JSON changes, so an arm and its control are the SAME extraction put through two different
post-processing states -- the extraction-drift hazard of kb v9-w7-002 (+1.02 GS-DetA between
identical runs eight days apart) cannot enter a comparison made this way.

That is the whole instrument here: given one submission directory, score it under several declared
flag states and pair the states per sequence. The same directories that get scored are the ones that
get packaged, so what is measured is what is shipped.

CLI::

    python -m tools.gsr_v9_w9 --demo
    python -m tools.gsr_v9_w9 --src outputs/gsr/deleak_v6detdev_clip_e0.3r1w0.5a0.3 \
        --states off,vote,full --out results/gsr_benchmark/gsr_v9_w9_dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("gsr_v9_w9")

#: The declared flag states. ``full`` is the v9 shipping bundle; ``off`` is the v6 shipped chain.
STATES: dict[str, dict[str, tuple[str, ...]]] = {
    "off": {"vote": (), "gk": ()},
    "vote": {"vote": ("role", "team", "jersey"), "gk": ()},
    "full": {"vote": ("role", "team", "jersey"), "gk": ("team",)},
}


def dir_digest(src: Path, seqs: list[str]) -> str:
    """Sha256 over the per-sequence prediction files, pinning one extraction lineage (pure IO)."""
    h = hashlib.sha256()
    for name in seqs:
        h.update(name.encode())
        h.update(hashlib.sha256((src / f"{name}.json").read_bytes()).digest())
    return h.hexdigest()


def score_states(src: Path, data_dir: Path, work: Path, seqs: list[str],
                 states: list[str]) -> dict:
    """Score one submission lineage under each declared flag state.

    Args:
        src: ``predictions/data`` directory of the submission to post-process.
        data_dir: GSR ground-truth root.
        work: Scratch root; each state lands in ``work/<state>`` as a scorable arm directory.
        seqs: Sequences to score.
        states: Keys of :data:`STATES`, in report order.

    Returns:
        ``{state: {"combined": ..., "per_seq": ..., "changed": ..., "dir": ...}}``.
    """
    import shutil  # noqa: PLC0415

    from eval.gsr_score import EVAL_CONFIGS, gs_hota  # noqa: PLC0415
    from tools.gsr_v9_w7 import _prepare  # noqa: PLC0415, SLF001 - the shipping-order transform

    out: dict[str, dict] = {}
    for state in states:
        dest = work / state
        shutil.rmtree(dest, ignore_errors=True)
        changed = _prepare(src, dest, seqs, STATES[state])
        res = gs_hota(dest, data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        out[state] = {"combined": res["combined"], "per_seq": res["per_seq"],
                      "changed": dict(changed), "dir": str(dest)}
        c = res["combined"]
        logger.info("%-5s GS-HOTA %.4f DetA %.4f AssA %.4f LocA %.4f IDF1 %.4f", state,
                    c["GS-HOTA"], c["GS-DetA"], c["GS-AssA"], c["GS-LocA"], c["IDF1"])
    return out


def paired(res: dict, base: str, arm: str, seqs: list[str]) -> dict:
    """Per-sequence paired statistics of ``arm`` minus ``base`` (both keys of ``res``)."""
    import numpy as np  # noqa: PLC0415
    from scipy.stats import wilcoxon  # noqa: PLC0415

    stats = {}
    for key in ("GS-HOTA", "GS-DetA", "GS-AssA"):
        d = np.array([res[arm]["per_seq"][s][key] - res[base]["per_seq"][s][key] for s in seqs])
        stats[key] = {"mean": float(d.mean()), "median": float(np.median(d)),
                      "helped": int((d > 0).sum()), "hurt": int((d < 0).sum()),
                      "n": len(d), "worst": float(d.min()), "best": float(d.max()),
                      "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
    stats["pooled"] = {k: res[arm]["combined"][k] - res[base]["combined"][k]
                       for k in ("GS-HOTA", "GS-DetA", "GS-AssA", "GS-LocA", "IDF1")}
    return stats


def _md5(path: Path) -> str:
    """Hex md5 of a file (provenance check, not crypto)."""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def _sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(lineages: dict, dev: dict, gate: dict, *,
           dest: Path = Path("results/gsr_v9_frozen.json"), note: str = "") -> dict:
    """Write the v9 freeze record: bundle, hashes, environment, lineages and the TEST-38 gate.

    Args:
        lineages: ``{split: {"dir": ..., "digest": ..., "on_record": ..., "extracted": ...}}``.
        dev: The DEV-20 leave-one-out evidence the bundle was chosen on.
        gate: The registered TEST-38 gate (thresholds + the pre-declared fail response).
        dest: Where the freeze is written.
        note: Free-text provenance note.

    Returns:
        The written record.
    """
    import platform  # noqa: PLC0415
    import sys  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    import numpy  # noqa: PLC0415
    import scipy  # noqa: PLC0415
    import trackeval  # noqa: PLC0415

    from eval.gsr_score import GK_SIDE_REPAIR, VOTE_TRACK_ATTRS  # noqa: PLC0415

    v6 = json.loads(Path("results/gsr_v6_frozen.json").read_text(encoding="utf-8"))
    parseq = Path.home() / "jersey-number-pipeline" / "models" / "parseq_v6_arm4t.ckpt"
    det = Path("models/gsr_det/gsr_v3_ft_b_last.pt")
    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "v9-1.0",
        "session": "v9 W9 (freeze / gate / package)",
        "parent": {"file": "results/gsr_v6_frozen.json",
                   "declared_at": v6["declared_at"],
                   "sha256": _sha256(Path("results/gsr_v6_frozen.json"))},
        "bundle": {
            "chain": "the frozen v6 chain, unchanged: S4b detector -> ByteTrack -> PnLCalib -> "
                     "calibration gate + fill -> EIoU re-association -> PARSeq arm-4t reader "
                     "(0.80 floor) -> GTA connector tau 0.450 + jersey-compatible merge -> frozen "
                     "MILP solver, free team map, self roster",
            "detector": {"weights": str(det), "md5": _md5(det),
                         "parked_md5": "2074d8741f97a6892a1322d0f808244c"},
            "jersey_reader": {"ckpt": str(parseq), "md5": _md5(parseq),
                              "parked_md5": "39c0c15defafda072ca0cefd0a4b8e18"},
            "association": {"module": "tools/gsr_eiou.py",
                            "sha256": _sha256(Path("tools/gsr_eiou.py")),
                            "params": "e=0.3 rounds=1 w_app=0.5 app_max=0.30, embedder clip"},
            "new_in_v9": {
                "VOTE_TRACK_ATTRS": list(VOTE_TRACK_ATTRS),
                "GK_SIDE_REPAIR": list(GK_SIDE_REPAIR),
                "call_site": "tools.gsr_deleak.write_arm (vote first, keeper repair second)",
                "provenance": ["results/GSR_V9_W3.md section 4 (+1.7999 GS-DetA DEV-20, 20/0)",
                               "results/gsr_v9_w4_registered.json (+2.2676 GS-DetA DEV-20, 18/0)"],
                "gt_free": "both read only our own predictions; no Labels-GameState.json",
            },
            "everything_else_off": "every other v9 candidate failed its registered gate and stays "
                                   "OFF: gk_side role/role2/role2dom (W4/W5), second-keeper arms "
                                   "(W5), name-borrowing (W6), KEEP_UNTRACKED and MIN_HITS=1 (W7), "
                                   "attach-inherit (W8, STOP before build)",
        },
        "code_sha256": {str(p): _sha256(p) for p in (
            Path("eval/gsr_score.py"), Path("tools/gsr_deleak.py"), Path("tools/gsr_eiou.py"),
            Path("tools/gsr_v6det.py"), Path("tools/gsr_v9_w9.py"))},
        "environment": {
            "scoring_host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "trackeval": f"{trackeval.__file__} (sn-trackeval 0.4.0 @ 9c25232, byte-audited)",
        },
        "lineages": lineages,
        "dev20": dev,
        "pre_declared_gate": gate,
        "note": note,
    }
    dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("froze v9 bundle -> %s", dest)
    return record


def demo() -> None:
    """Assert the pure seams: state table, digest stability, paired arithmetic."""
    import tempfile  # noqa: PLC0415

    assert STATES["off"] == {"vote": (), "gk": ()}
    assert STATES["full"]["gk"] == ("team",)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "A.json").write_text('{"predictions": []}', encoding="utf-8")
        first = dir_digest(d, ["A"])
        assert first == dir_digest(d, ["A"])
        (d / "A.json").write_text('{"predictions": [1]}', encoding="utf-8")
        assert dir_digest(d, ["A"]) != first
    res = {"a": {"per_seq": {"s1": {"GS-HOTA": 1.0, "GS-DetA": 1.0, "GS-AssA": 1.0},
                             "s2": {"GS-HOTA": 2.0, "GS-DetA": 2.0, "GS-AssA": 2.0}},
                 "combined": {"GS-HOTA": 1.5, "GS-DetA": 1.5, "GS-AssA": 1.5, "GS-LocA": 9.0,
                              "IDF1": 3.0}},
           "b": {"per_seq": {"s1": {"GS-HOTA": 3.0, "GS-DetA": 3.0, "GS-AssA": 3.0},
                             "s2": {"GS-HOTA": 3.0, "GS-DetA": 3.0, "GS-AssA": 3.0}},
                 "combined": {"GS-HOTA": 3.0, "GS-DetA": 3.0, "GS-AssA": 3.0, "GS-LocA": 9.0,
                              "IDF1": 4.0}}}
    st = paired(res, "a", "b", ["s1", "s2"])
    assert st["GS-HOTA"]["mean"] == 1.5
    assert (st["GS-HOTA"]["helped"], st["GS-HOTA"]["hurt"]) == (2, 0)
    assert abs(st["pooled"]["GS-HOTA"] - 1.5) < 1e-12
    assert abs(st["pooled"]["GS-LocA"]) < 1e-12
    print("gsr_v9_w9 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=None, help="arm directory (holds predictions/data)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--work", type=Path, default=Path("outputs/gsr/v9_w9"))
    ap.add_argument("--states", default="off,full", help=f"subset of {tuple(STATES)}")
    ap.add_argument("--base", default="off", help="paired-comparison baseline state")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if args.src is None or args.out is None:
        raise SystemExit("--src and --out are required")
    src = args.src / "predictions" / "data"
    seqs = sorted(p.stem for p in src.glob("*.json"))
    states = args.states.split(",")
    logger.info("src %s: %d sequences, states %s", src, len(seqs), states)
    res = score_states(src, args.data_dir, args.work / args.src.name, seqs, states)
    payload = {"src": str(args.src), "digest": dir_digest(src, seqs), "seqs": seqs,
               "states": {k: {"combined": v["combined"], "changed": v["changed"],
                              "dir": v["dir"], "per_seq": v["per_seq"]}
                          for k, v in res.items()},
               "paired": {f"{args.base}->{s}": paired(res, args.base, s, seqs)
                          for s in states if s != args.base}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
