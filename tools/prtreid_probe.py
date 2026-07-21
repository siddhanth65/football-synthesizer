"""Probe: does PRTreID (part-based, SoccerNet-trained) break the same-kit ReID wall?

Runs the *identical* protocol as the 2026-07-16 four-way OSNet ReID ladder
(``results/gsr_benchmark/GSR_BENCHMARK.md``, "Stage 2c"): on the three frozen GSR pilot sequences
(``eval.gsr_score.PILOT_SEQS``, the only data with GT track ids, hence the only data where merge
precision is auditable), embed every ByteTrack fragment, keep the constraint-valid pairs
(:func:`generator.track_relink.pair_similarities` -- temporally disjoint, same team, same role,
motion-feasible), and report

* same-kit separation: median cosine of same-player pairs minus median cosine of different-player
  pairs (all pairs are same-team by construction, i.e. same kit);
* merge precision at the frozen threshold 0.80 (the pre-committed >= 80% production bar);
* a threshold sweep, so a model that separates at a *different* operating point is not missed.

PRTreID lives outside this repo (clone + weights under :data:`PRTREID_DIR`); it is imported by path
with stubs for its training-only third-party deps. Only the ``globl`` test embedding is used, which
is pose-mask-free (``BPBreID.forward`` computes it by plain average pooling), matching the
sn-gamestate inference config ``test_embeddings: ["globl"]``.

Embeddings are cached per (sequence, arm) under ``<out-dir>/relink_cache_<arm>/``, so a long GPU
pass is resumable by disk state: re-running skips whatever is already on disk.

Example:
    python -m tools.prtreid_probe --arm prtreid
    python -m tools.prtreid_probe --arm osnet          # baseline, reuses the shipped cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("prtreid_probe")

#: External PRTreID checkout (clone of github.com/VlSomers/prtreid); override with ``$PRTREID_DIR``.
PRTREID_DIR = Path(os.environ.get("PRTREID_DIR", str(Path.home() / "prtreid")))
#: SoccerNet-trained PRTreID checkpoint (zenodo record 10653453).
PRTREID_WEIGHTS = PRTREID_DIR / "weights" / "prtreid-soccernet-baseline.pth.tar"
#: Training-only deps of the ``prtreid`` package that inference never touches.
_STUB_MODULES = ("wandb", "albumentations", "monai", "torchmetrics", "tabulate", "deepdiff")
#: PRTreID input size (h, w) and ImageNet normalisation -- same as the OSNet arm.
_PRTREID_HW = (256, 128)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
#: Frozen production merge threshold (tuned on the pilot in Stage 2a, then frozen).
FROZEN_THRESHOLD = 0.80
#: Pre-committed merge-precision bar for graduating relink to per-player facts.
PRECISION_BAR = 0.80


def _stub_module(name: str) -> None:
    """Register a dummy module under ``name`` so an unused training-time import succeeds."""
    mod = types.ModuleType(name)
    mod.__file__ = f"<stub:{name}>"

    def _getattr(attr: str) -> type:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return type(attr, (object,), {})

    mod.__getattr__ = _getattr  # type: ignore[method-assign]
    sys.modules[name] = mod


def _import_prtreid(max_stubs: int = 15):
    """Import the external ``prtreid`` model factory, stubbing training-only deps (IO).

    Args:
        max_stubs: Safety bound on how many missing modules will be stubbed.

    Returns:
        The ``prtreid.models.bpbreid.bpbreid`` constructor.

    Raises:
        FileNotFoundError: If :data:`PRTREID_DIR` does not contain the package.
        ImportError: If the package still fails to import after stubbing.
    """
    if not (PRTREID_DIR / "prtreid").is_dir():
        raise FileNotFoundError(f"prtreid checkout not found at {PRTREID_DIR}")
    if str(PRTREID_DIR) not in sys.path:
        sys.path.insert(0, str(PRTREID_DIR))
    for name in _STUB_MODULES:
        try:
            __import__(name)
        except ImportError:
            _stub_module(name)
    for _ in range(max_stubs):
        try:
            from prtreid.models.bpbreid import bpbreid  # noqa: PLC0415

            return bpbreid
        except ModuleNotFoundError as exc:
            if not exc.name:
                raise
            _stub_module(exc.name)
    raise ImportError(f"prtreid still unimportable after {max_stubs} stubs")


class PrtreidEmbedder:
    """PRTreID (BPBreID/HRNet-32, SoccerNet-trained) appearance embedder.

    Duck-type-compatible with :class:`generator.track_relink.OsnetEmbedder` (same ``embed`` signature
    and the same crop preprocessing), so the relink code path is byte-identical across arms. Emits
    the 256-d ``globl`` embedding, which needs no pose masks.
    """

    def __init__(self, weights: Path = PRTREID_WEIGHTS, device: str | None = None,
                 batch_size: int = 32) -> None:
        import torch  # noqa: PLC0415

        if not weights.exists():
            raise FileNotFoundError(f"PRTreID weights not found at {weights}")
        bpbreid = _import_prtreid()
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        n_classes = int(ckpt["state_dict"]["module.global_identity_classifier.classifier.weight"]
                        .shape[0])
        # pretrained=False: the SoccerNet checkpoint supplies the backbone; no ImageNet HRNet needed.
        model = bpbreid(n_classes, loss="part_based", pretrained=False, config=cfg)
        state = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"PRTreID checkpoint missing {len(missing)} keys, e.g. {missing[:3]}")
        logger.info("PRTreID loaded: %d classes, %d unexpected keys, device=%s",
                    n_classes, len(unexpected), self.device)
        self.model = model.to(self.device).eval()

    def embed(self, crops: list[np.ndarray], batch_size: int | None = None) -> np.ndarray:
        """Embed RGB uint8 crops -> ``(N, 256)`` L2-normalised float32 ``globl`` features."""
        import cv2  # noqa: PLC0415

        torch = self._torch
        if not crops:
            return np.zeros((0, 256), np.float32)
        bs = batch_size or self.batch_size
        out = []
        for start in range(0, len(crops), bs):
            chunk = crops[start:start + bs]
            batch = np.empty((len(chunk), _PRTREID_HW[0], _PRTREID_HW[1], 3), np.float32)
            for i, c in enumerate(chunk):
                r = cv2.resize(c, (_PRTREID_HW[1], _PRTREID_HW[0]), interpolation=cv2.INTER_LINEAR)
                batch[i] = (r.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode():
                embeddings = self.model(t)[0]
                feats = embeddings["globl"].float().cpu().numpy()
            del t
            if self.device == "cuda":
                torch.cuda.empty_cache()
            out.append(feats)
        feats = np.concatenate(out, axis=0)
        return feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)


@dataclass(frozen=True)
class SeparationStats:
    """Same-kit cosine diagnostic over constraint-valid fragment pairs.

    Attributes:
        n_same: Number of auditable pairs whose two fragments share a GT track id.
        n_diff: Number of auditable pairs with different GT track ids (same kit by construction).
        same_median: Median cosine of the same-player pairs.
        diff_median: Median cosine of the different-player pairs.
    """

    n_same: int
    n_diff: int
    same_median: float
    diff_median: float

    @property
    def separation(self) -> float:
        """Median same-player cosine minus median different-player cosine."""
        return self.same_median - self.diff_median


def separation_stats(fragments, sims: dict[tuple[int, int], float],
                     gt_ids: dict[int, int]) -> SeparationStats:
    """Split constraint-valid pair cosines by GT identity and summarise (pure).

    Args:
        fragments: The sequence's fragments (index space of ``sims``).
        sims: ``{(i, j): cosine}`` from :func:`generator.track_relink.pair_similarities`.
        gt_ids: ``track_id -> dominant GT track id``; pairs missing either id are skipped.

    Returns:
        A :class:`SeparationStats` over the auditable pairs.
    """
    same: list[float] = []
    diff: list[float] = []
    for (i, j), cos in sims.items():
        ga = gt_ids.get(fragments[i].track_id)
        gb = gt_ids.get(fragments[j].track_id)
        if ga is None or gb is None:
            continue
        (same if ga == gb else diff).append(cos)
    return SeparationStats(
        n_same=len(same), n_diff=len(diff),
        same_median=float(np.median(same)) if same else float("nan"),
        diff_median=float(np.median(diff)) if diff else float("nan"),
    )


def _thr_key(threshold: float) -> str:
    """Stable dict key for a threshold (3 decimals, so 0.98 and 0.985 do not collide)."""
    return f"{threshold:.3f}"


def _build_embedder(arm: str, weights: str | None, device: str):
    """Build the arm's embedder (``prtreid`` or ``osnet``)."""
    if arm == "prtreid":
        return PrtreidEmbedder(device=device)
    from generator.track_relink import OsnetEmbedder  # noqa: PLC0415

    model_name = "osnet_x0_25" if weights is None else (
        "osnet_ain_x1_0" if "ain" in weights else "osnet_x1_0")
    return OsnetEmbedder(model_name, device=device, weights=weights)


def probe_sequences(
    seqs: list[str], data_dir: Path, out_dir: Path, *, arm: str, weights: str | None,
    thresholds: list[float],
) -> dict:
    """Embed, diagnose separation, and audit merge precision for one arm (GPU/IO).

    Args:
        seqs: GSR sequence names (must have GT ``Labels-GameState.json`` and a positions parquet).
        data_dir: GSR dataset root.
        out_dir: Pipeline output root (holds ``positions/`` and the embedding caches).
        arm: ``"prtreid"`` or ``"osnet"``.
        weights: OSNet weights spec (model-zoo key or path); ignored for the PRTreID arm.
        thresholds: Merge thresholds to audit; :data:`FROZEN_THRESHOLD` is always included.

    Returns:
        A payload with per-sequence and pooled separation + merge-precision numbers.
    """
    from generator.track_relink import (  # noqa: PLC0415
        RelinkParams, _load_or_build_embeddings, fragment_gt_ids, greedy_merge,
        load_gt_ids_by_frame, merge_precision, pair_similarities, summarize_fragments,
    )

    # The shipped ImageNet-OSNet cache keeps its original path so the baseline arm needs no GPU;
    # every other (arm, weights) combination gets its own directory.
    tag = arm if weights is None else f"{arm}_{Path(weights).stem}"
    cache_dir = out_dir / ("relink_cache" if arm == "osnet" and weights is None
                           else f"relink_cache_{tag}")
    params = RelinkParams(weights=weights)
    cold = [s for s in seqs if not (cache_dir / f"{s}.npz").exists()]
    logger.info("arm=%s cache=%s cold=%s", arm, cache_dir, cold or "none (resuming from disk)")
    embedder = detector = None
    if cold:
        import torch  # noqa: PLC0415

        from generator.extract import _build_detector  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = _build_embedder(arm, weights, device)
        detector = _build_detector(device, "football")

    per_seq: dict[str, dict] = {}
    all_same: list[float] = []
    all_diff: list[float] = []
    prec: dict[float, list[int]] = {t: [0, 0] for t in thresholds}
    for name in seqs:
        seq_dir = data_dir / name
        df = pd.read_parquet(out_dir / "positions" / f"{name}.parquet")
        fragments = summarize_fragments(df)
        embeddings, counts = _load_or_build_embeddings(
            seq_dir, df, params=params, embedder=embedder, detector=detector,
            cache_path=cache_dir / f"{name}.npz")
        sims = pair_similarities(fragments, embeddings)
        gt_ids = fragment_gt_ids(df, load_gt_ids_by_frame(seq_dir))
        sep = separation_stats(fragments, sims, gt_ids)
        for (i, j), cos in sims.items():
            ga, gb = gt_ids.get(fragments[i].track_id), gt_ids.get(fragments[j].track_id)
            if ga is not None and gb is not None:
                (all_same if ga == gb else all_diff).append(cos)
        seq_prec: dict[str, list[int]] = {}
        for t in thresholds:
            remap = greedy_merge(fragments, sims, t)
            c, n = merge_precision(remap, gt_ids)
            prec[t][0] += c
            prec[t][1] += n
            seq_prec[_thr_key(t)] = [c, n]
        per_seq[name] = {
            "n_fragments": len(fragments), "n_embedded": len(embeddings),
            "mean_crops": float(np.mean(list(counts.values()))) if counts else 0.0,
            "n_pairs": len(sims), "n_same": sep.n_same, "n_diff": sep.n_diff,
            "same_median": sep.same_median, "diff_median": sep.diff_median,
            "merge_precision": seq_prec,
        }
        logger.info("%s: frags %d, pairs %d (same %d / diff %d), sep %+.4f",
                    name, len(fragments), len(sims), sep.n_same, sep.n_diff, sep.separation)

    pooled = SeparationStats(
        n_same=len(all_same), n_diff=len(all_diff),
        same_median=float(np.median(all_same)) if all_same else float("nan"),
        diff_median=float(np.median(all_diff)) if all_diff else float("nan"),
    )
    return {
        "arm": arm, "sequences": seqs, "cache_dir": str(cache_dir),
        "weights": str(PRTREID_WEIGHTS) if arm == "prtreid" else weights,
        "per_seq": per_seq,
        "pooled": {
            "n_same": pooled.n_same, "n_diff": pooled.n_diff,
            "same_median": pooled.same_median, "diff_median": pooled.diff_median,
            "separation": pooled.separation,
        },
        "merge_precision": {
            _thr_key(t): {"correct": c, "total": n,
                          "precision": (c / n if n else float("nan"))}
            for t, (c, n) in prec.items()
        },
    }


def _print_report(payload: dict) -> None:
    """Print the arm's numbers ASCII-only (cp1252 console)."""
    p = payload["pooled"]
    print(f"\n=== arm: {payload['arm']} (weights={payload['weights']}) ===")
    print(f"pairs: same-player {p['n_same']}, different-player same-kit {p['n_diff']}")
    print(f"same-player median cos {p['same_median']:.3f} | diff-player median cos "
          f"{p['diff_median']:.3f} | separation {p['separation']:+.4f}")
    print(f"{'thresh':>7} {'correct':>8} {'total':>7} {'precision':>10}")
    for th, r in sorted(payload["merge_precision"].items()):
        pct = "n/a" if not r["total"] else f"{100 * r['precision']:.1f}%"
        print(f"{th:>7} {r['correct']:>8} {r['total']:>7} {pct:>10}")
    frozen = payload["merge_precision"].get(_thr_key(FROZEN_THRESHOLD), {})
    if frozen.get("total"):
        verdict = "CLEARS" if frozen["precision"] >= PRECISION_BAR else "FAILS"
        print(f"verdict at frozen threshold {FROZEN_THRESHOLD:.2f}: {verdict} the "
              f"{100 * PRECISION_BAR:.0f}% bar ({100 * frozen['precision']:.1f}%)")


def _demo() -> None:
    """Self-check of the separation diagnostic on synthetic pairs (asserts; runnable)."""
    from generator.track_relink import Fragment  # noqa: PLC0415

    frags = [Fragment(i, 0, 1, (0.0, 0.0), (0.0, 0.0), "player", 0, 1) for i in (10, 11, 12)]
    sims = {(0, 1): 0.9, (0, 2): 0.5, (1, 2): 0.4}
    gt = {10: 1, 11: 1, 12: 2}  # 10 & 11 are the same player; 12 is a same-kit teammate
    s = separation_stats(frags, sims, gt)
    assert (s.n_same, s.n_diff) == (1, 2), s
    assert s.same_median == 0.9 and s.diff_median == 0.45  # noqa: PLR2004
    assert abs(s.separation - 0.45) < 1e-9
    s2 = separation_stats(frags, sims, {10: 1, 11: 1})  # fragment 12 unauditable -> its pairs drop
    assert (s2.n_same, s2.n_diff) == (1, 0), s2
    print("prtreid_probe demo OK: separation diagnostic splits and skips unauditable pairs")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, PILOT_SEQS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="prtreid", choices=["prtreid", "osnet"])
    ap.add_argument("--weights", default=None, help="OSNet model-zoo key or checkpoint path")
    ap.add_argument("--seqs", default=",".join(PILOT_SEQS))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--thresholds", default="0.60,0.70,0.80,0.85,0.90,0.95")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--demo", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    thresholds = sorted({float(t) for t in args.thresholds.split(",")} | {FROZEN_THRESHOLD})
    payload = probe_sequences(
        args.seqs.split(","), args.data_dir, args.out_dir,
        arm=args.arm, weights=args.weights, thresholds=thresholds)
    _print_report(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
