"""Sweep driver for the evidence-density law: conditions -> precision/coverage frontiers.

Runs :mod:`tools.evidence_sim` over a bounded, pre-declared grid and writes
``results/evidence_density/sweep.json`` plus the figures used by
``results/EVIDENCE_DENSITY_LAW.md``. Every condition is graded on the FOOTPASS action events with
the metrics Stage 2's post-mortem demanded: **coverage at a precision floor**, never bare accuracy.

Scale note. The primary grid runs on a deterministic 20-half sample of the 96 TRAIN halves
(``sorted(paths)[::5]``) and on all 6 VAL halves; VAL is never used to choose anything. Running all
96 TRAIN halves at every condition would cost ~4x the wall clock for a standard error already below
0.01 at 18k events per condition.

CLI::

    python -m tools.evidence_sweep --run          # the grid (CPU, multiprocess)
    python -m tools.evidence_sweep --report       # tables + PNGs from the cached JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tools.evidence_sim import (
    DIGEST_DIR,
    FRAG_ARMS,
    SimConfig,
    calibrate_fragmentation,
    coverage_at,
    frontier,
    score_half,
    solver_config,
)

logger = logging.getLogger("evidence_sweep")

OUT_DIR = Path("results/evidence_density")
SWEEP_JSON = OUT_DIR / "sweep.json"

#: Pre-declared precision floors (Stage 2: "maximise coverage subject to a precision floor").
FLOORS = (0.85, 0.60)
#: The commentary go/no-go operating point declared in ``docs/ATTRIBUTION_RESEARCH_PLAN.md``.
GO_NO_GO = {"mention_rate": 1.5, "mention_precision": 0.60, "mention_lag_iqr": 4.0}


def train_halves() -> list[Path]:
    """Deterministic 20-half TRAIN sample used to fit the law."""
    return sorted(DIGEST_DIR.glob("train_*.npz"))[::5]


def val_halves() -> list[Path]:
    """The 3 held-out VAL games (6 halves)."""
    return sorted(DIGEST_DIR.glob("val_*.npz"))


def conditions() -> list[tuple[str, SimConfig]]:
    """The bounded grid, in the order it is reported (pure)."""
    out: list[tuple[str, SimConfig]] = []
    # --- calibration gate: GSR geometry (750-frame solve unit) at the measured operating point ---
    out.append(("gate/gsr750", SimConfig(chunk_frames=750, ocr_density=0.087, ocr_precision=0.86)))
    # 8.7% is per *original* track; per merged tracklet (the solve unit) the measured rate is 10.3%.
    out.append(("gate/gsr750_d0.103",
                SimConfig(chunk_frames=750, ocr_density=0.103, ocr_precision=0.86)))
    out.append(("gate/gsr750_noevidence", SimConfig(chunk_frames=750, ocr_density=0.0)))
    out.append(("ref/noevidence", SimConfig(ocr_density=0.0)))
    # --- A: OCR-like density x precision, measured fragmentation, match-scale solve unit ---
    for prec in (0.86, 0.95):
        for d in (0.087, 0.15, 0.25, 0.40, 0.60, 1.0):
            out.append((f"ocr/d{d:g}_p{prec:g}", SimConfig(ocr_density=d, ocr_precision=prec)))
    # --- B: commentary-like rate x precision x lag, OCR off (channel measured on its own) ---
    for prec in (0.60, 0.80, 0.95):
        for rate in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            out.append((f"cmt/r{rate:g}_p{prec:g}_l4",
                        SimConfig(ocr_density=0.0, mention_rate=rate, mention_precision=prec,
                                  mention_lag_iqr=4.0)))
        for rate in (1.0, 1.5, 3.0):
            out.append((f"cmt/r{rate:g}_p{prec:g}_l8",
                        SimConfig(ocr_density=0.0, mention_rate=rate, mention_precision=prec,
                                  mention_lag_iqr=8.0)))
    # --- C: the go/no-go point and its neighbourhood, stacked on the measured OCR channel ---
    for rate in (1.0, 1.5, 2.0):
        for prec in (0.60, 0.80):
            out.append((f"both/r{rate:g}_p{prec:g}",
                        SimConfig(ocr_density=0.087, ocr_precision=0.86, mention_rate=rate,
                                  mention_precision=prec, mention_lag_iqr=4.0)))
    # --- C2: is it the *alignment* or the channel? lag-compensated and oracle-bound commentary ---
    for prec in (0.60, 0.95):
        for rate in (1.5, 3.0, 5.0):
            out.append((f"cmtc/r{rate:g}_p{prec:g}_l4",
                        SimConfig(ocr_density=0.0, mention_rate=rate, mention_precision=prec,
                                  mention_lag_iqr=4.0, mention_lag_compensate=True)))
        for rate in (1.5, 3.0):
            out.append((f"cmto/r{rate:g}_p{prec:g}",
                        SimConfig(ocr_density=0.0, mention_rate=rate, mention_precision=prec,
                                  mention_bind_oracle=True)))
    for iqr in (0.5, 1.0, 2.0):
        out.append((f"cmtc/r3_p0.95_l{iqr:g}",
                    SimConfig(ocr_density=0.0, mention_rate=3.0, mention_precision=0.95,
                              mention_lag_iqr=iqr, mention_lag_compensate=True)))
    # --- D: fragmentation axis at three evidence points ---
    for arm, rate in FRAG_ARMS.items():
        for tag, kw in (("ocr0.087", {"ocr_density": 0.087}),
                        ("ocr0.40", {"ocr_density": 0.40}),
                        ("cmt1.5", {"ocr_density": 0.0, "mention_rate": 1.5,
                                    "mention_precision": 0.60})):
            out.append((f"frag/{arm}_{tag}", SimConfig(frag_rate=rate, ocr_precision=0.86, **kw)))
    # Volume-controlled fragmentation: ``ocr_density`` is per *tracklet*, so a more fragmented arm
    # silently gets more reads. These arms hold reads per 30 s of on-screen time fixed at the
    # connector arm's measured 0.370 (= 0.0867 reads/tracklet x 4.27 tracklets per 30 s), which is
    # the physically invariant quantity: fragmenting a track does not create legible frames.
    for arm, achieved in (("none", 0.63), ("gsr_baseline", 5.31), ("fulham_baseline", 6.19)):
        out.append((f"fragc/{arm}_matched",
                    SimConfig(frag_rate=FRAG_ARMS[arm], ocr_precision=0.86,
                              ocr_density=min(1.0, 0.370 / achieved))))
    return out


def _run_one(args: tuple[str, dict, float]) -> dict:
    """Worker: score one half under one condition (loads its own digest)."""
    path, cfg_d, scale = args
    cfg = SimConfig(**cfg_d)
    d = np.load(path)
    half = {k: d[k] for k in d.files}
    r = score_half(half, cfg, scale, solver_config(cfg))
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def _aggregate(parts: list[dict], raw_path: Path | None = None) -> dict:
    """Merge per-half records, cache the raw per-event outcomes, derive the pre-declared metrics."""
    cat = {k: np.concatenate([np.asarray(p[k]) for p in parts])
           for k in ("correct", "answered", "conf", "base_correct", "base_answered",
                     "app_correct", "app_answered", "anchored", "attributable")}
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **{k: (v.astype(np.float32) if k == "conf" else v)
                                         for k, v in cat.items()})
    n = cat["correct"].size
    front = frontier(cat["correct"], cat["answered"], cat["conf"], n)
    ans, cor = cat["answered"], cat["correct"]
    n_frag = sum(p["n_fragments"] for p in parts)
    vis = sum(p["visible_seconds"] for p in parts)
    reads = sum(p["n_reads"] for p in parts)
    return {
        "n_events": int(n),
        "attributable": float(cat["attributable"].mean()),
        "anchored": float(cat["anchored"].mean()),
        "accuracy_all_events": float(cor.mean()),
        "coverage_full": float(ans.mean()),
        "precision_full": float(cor[ans].mean()) if ans.any() else float("nan"),
        "cov@0.85": coverage_at(front, 0.85),
        "cov@0.60": coverage_at(front, 0.60),
        "base_coverage": float(cat["base_answered"].mean()),
        "base_precision": (float(cat["base_correct"][cat["base_answered"]].mean())
                           if cat["base_answered"].any() else float("nan")),
        "app_coverage": float(cat["app_answered"].mean()),
        "app_precision": (float(cat["app_correct"][cat["app_answered"]].mean())
                          if cat["app_answered"].any() else float("nan")),
        "frag_per_30s_visible": n_frag / max(vis / 30.0, 1e-9),
        "n_fragments": int(n_frag),
        "reads_per_tracklet": reads / max(n_frag, 1),
        "read_precision": (sum(p["n_reads_correct"] for p in parts) / reads) if reads else None,
        "frontier": front,
    }


def run(workers: int = 5) -> None:
    """Execute the grid on TRAIN and VAL and cache every condition's metrics."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    splits = {"train": train_halves(), "val": val_halves()}
    logger.info("%d conditions x (%d train + %d val) halves", len(conds),
                len(splits["train"]), len(splits["val"]))
    from tools.evidence_sim import load_halves  # noqa: PLC0415
    cal_halves = load_halves("train")[::9]
    scales: dict[tuple[float, int], float] = {}
    blob = json.loads(SWEEP_JSON.read_text(encoding="utf-8")) if SWEEP_JSON.exists() else {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, cfg) in enumerate(conds):
            if name in blob:
                continue
            key = (cfg.frag_rate, cfg.chunk_frames)
            if key not in scales:
                scales[key] = calibrate_fragmentation(cal_halves, *key)
            t0 = time.time()
            rec = {"config": asdict(cfg), "scale": scales[key]}
            for split, paths in splits.items():
                jobs = [(str(p), asdict(cfg), scales[key]) for p in paths]
                raw = OUT_DIR / "raw" / f"{name.replace('/', '_')}_{split}.npz"
                rec[split] = _aggregate(list(pool.map(_run_one, jobs)), raw)
            blob[name] = rec
            SWEEP_JSON.write_text(json.dumps(blob, indent=1), encoding="utf-8")
            logger.info("[%d/%d] %-24s train cov@.85 %.3f cov@.60 %.3f prec_full %.3f (%.0fs)",
                        i + 1, len(conds), name, rec["train"]["cov@0.85"],
                        rec["train"]["cov@0.60"], rec["train"]["precision_full"],
                        time.time() - t0)


# === Reporting ===================================================================================
def _table(blob: dict, names: list[str], split: str = "train") -> str:
    """Markdown table of the pre-declared metrics for a set of conditions (pure)."""
    head = ("| condition | events | attributable | anchored | coverage | precision | cov@0.85 | "
            "cov@0.60 | acc/event | direct-read cov/prec | appearance prec |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for n in names:
        r = blob[n][split]
        rows.append(f"| {n} | {r['n_events']:,} | {r['attributable']:.3f} | {r['anchored']:.3f} | "
                    f"{r['coverage_full']:.3f} | {r['precision_full']:.3f} | {r['cov@0.85']:.3f} | "
                    f"{r['cov@0.60']:.3f} | {r['accuracy_all_events']:.3f} | "
                    f"{r['base_coverage']:.3f} / {r['base_precision']:.3f} | "
                    f"{r['app_precision']:.3f} |")
    return head + "\n".join(rows)


def _plot(blob: dict) -> None:
    """Write the four figures under ``results/evidence_density``."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    def front(name: str, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        f = np.array([[c, p] for _t, c, p in blob[name][split]["frontier"] if np.isfinite(p)])
        return f[:, 0], f[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, prec in zip(axes, (0.86, 0.95)):
        for d in (0.087, 0.15, 0.25, 0.40, 0.60, 1.0):
            n = f"ocr/d{d:g}_p{prec:g}"
            if n in blob:
                ax.plot(*front(n), marker="", label=f"d={d:g}")
        ax.axhline(0.85, ls=":", c="k", lw=0.8)
        ax.axvline(0.50, ls=":", c="k", lw=0.8)
        ax.set(xlabel="coverage (all events)", title=f"OCR-like channel, read precision {prec:g}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("precision (answered events)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ocr_frontier.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, prec in zip(axes, (0.60, 0.80, 0.95)):
        for rate in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            n = f"cmt/r{rate:g}_p{prec:g}_l4"
            if n in blob:
                ax.plot(*front(n), label=f"{rate:g}/min")
        ax.axhline(0.85, ls=":", c="k", lw=0.8)
        ax.axvline(0.50, ls=":", c="k", lw=0.8)
        ax.set(xlabel="coverage (all events)", ylim=(0.0, 1.02),
               title=f"name precision {prec:g}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("precision (answered events)")
    axes[0].legend(fontsize=8)
    fig.suptitle("commentary channel, lag IQR 4 s (uncompensated binding)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "commentary_frontier.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, lab in (("cmtc/r3_p0.95_l0.5", "0.5"), ("cmtc/r3_p0.95_l1", "1"),
                      ("cmtc/r3_p0.95_l2", "2"), ("cmtc/r3_p0.95_l4", "4"),
                      ("cmto/r3_p0.95", "0 (oracle bind)")):
        if name in blob:
            ax.plot(*front(name), label=f"lag IQR {lab} s")
    if "ocr/d0.087_p0.86" in blob:
        ax.plot(*front("ocr/d0.087_p0.86"), c="k", ls="--",
                label="today's OCR (d=0.087, p=0.86)")
    ax.axhline(0.85, ls=":", c="k", lw=0.8)
    ax.axvline(0.50, ls=":", c="k", lw=0.8)
    ax.set(xlabel="coverage (all events)", ylabel="precision (answered events)", ylim=(0.0, 1.02),
           title="commentary at 3 names/min, precision 0.95, median-lag compensated:\n"
                 "everything depends on timestamp alignment")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "commentary_alignment.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for prec in (0.86, 0.95):
        ds = [d for d in (0.087, 0.15, 0.25, 0.40, 0.60, 1.0) if f"ocr/d{d:g}_p{prec:g}" in blob]
        for ax, floor in zip(axes, FLOORS):
            ax.plot(ds, [blob[f"ocr/d{d:g}_p{prec:g}"]["train"][f"cov@{floor:.2f}"] for d in ds],
                    marker="o", label=f"OCR p={prec:g}")
    for prec in (0.60, 0.80, 0.95):
        rs = [r for r in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0) if f"cmt/r{r:g}_p{prec:g}_l4" in blob]
        # Both channels share one x axis: reads that actually land on a tracklet, per tracklet.
        xs = [blob[f"cmt/r{r:g}_p{prec:g}_l4"]["train"]["reads_per_tracklet"] for r in rs]
        for ax, floor in zip(axes, FLOORS):
            ax.plot(xs, [blob[f"cmt/r{r:g}_p{prec:g}_l4"]["train"][f"cov@{floor:.2f}"] for r in rs],
                    marker="s", ls="--", label=f"commentary p={prec:g}")
    for ax, floor in zip(axes, FLOORS):
        ax.axhline(0.50, ls=":", c="k", lw=0.8)
        ax.set(xlabel="evidence reads landed per tracklet",
               ylabel=f"coverage at precision >= {floor:g}", xscale="log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_at_floor.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for tag in ("ocr0.087", "ocr0.40", "cmt1.5"):
        arms = [a for a in FRAG_ARMS if f"frag/{a}_{tag}" in blob]
        ax.plot([blob[f"frag/{a}_{tag}"]["train"]["frag_per_30s_visible"] for a in arms],
                [blob[f"frag/{a}_{tag}"]["train"]["precision_full"] for a in arms],
                marker="o", label=f"{tag} (density fixed per tracklet)")
    matched = [(f"fragc/{a}_matched" if a != "connector" else "frag/connector_ocr0.087")
               for a in FRAG_ARMS]
    matched = [n for n in matched if n in blob]
    ax.plot([blob[n]["train"]["frag_per_30s_visible"] for n in matched],
            [blob[n]["train"]["precision_full"] for n in matched],
            marker="D", ls="--", c="k", label="OCR at matched read VOLUME (the controlled arm)")
    ax.set(xlabel="fragments per 30 s of on-screen time",
           ylabel="precision at full coverage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fragmentation_interaction.png", dpi=130)
    plt.close(fig)


def headlines(blob: dict) -> str:
    """The two pre-declared answers, read straight off the grid (pure)."""
    lines = ["| channel | setting | cov@0.85 | cov@0.60 | precision at full coverage | clears "
             "0.85/0.50? |", "|---|---|---|---|---|---|"]
    rows: list[tuple[str, str, str]] = []
    for prec in (0.86, 0.95):
        rows += [("OCR-like", f"read precision {prec:g}", f"ocr/d{d:g}_p{prec:g}")
                 for d in (0.087, 0.15, 0.25, 0.40, 0.60, 1.0)]
    for prec in (0.60, 0.80, 0.95):
        rows += [("commentary", f"name precision {prec:g}, lag IQR 4 s",
                  f"cmt/r{r:g}_p{prec:g}_l4") for r in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)]
    for chan, setting, name in rows:
        if name not in blob:
            continue
        r = blob[name]["train"]
        ok = "YES" if r["cov@0.85"] >= 0.50 else "no"
        lines.append(f"| {chan} | {setting} / {name.split('/')[1]} | {r['cov@0.85']:.3f} | "
                     f"{r['cov@0.60']:.3f} | {r['precision_full']:.3f} | {ok} |")
    return "\n".join(lines)


def report() -> None:
    """Print the tables the law document is built from."""
    blob = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    groups = {
        "calibration gate + references": [n for n in blob if n.startswith(("gate/", "ref/"))],
        "OCR-like channel": [n for n in blob if n.startswith("ocr/")],
        "commentary channel": [n for n in blob if n.startswith("cmt/")],
        "commentary: lag-compensated (cmtc) and oracle-bound (cmto)":
            [n for n in blob if n.startswith(("cmtc/", "cmto/"))],
        "commentary stacked on measured OCR": [n for n in blob if n.startswith("both/")],
        "fragmentation axis": [n for n in blob if n.startswith("frag/")],
        "fragmentation at matched read volume": [n for n in blob if n.startswith("fragc/")],
    }
    for title, names in groups.items():
        if not names:
            continue
        print(f"\n### {title} (TRAIN)\n")
        print(_table(blob, names))
    print("\n### headline: does any grid point reach precision 0.85 at coverage >= 0.50?\n")
    print(headlines(blob))
    print("\n### VAL transfer (held out)\n")
    keep = [n for n in blob if n.startswith(("gate/", "ocr/", "both/"))
            or n.startswith("cmt/r1.5_")]
    print(_table(blob, keep, split="val"))
    _plot(blob)
    print(f"\nfigures -> {OUT_DIR}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.run:
        run(args.workers)
    if args.report:
        report()


if __name__ == "__main__":
    main()
