"""Score the closed-set carrier identification against Sid's hand labels.

Input is ``labels_filled.csv`` as exported by ``results/carrier_attr/labelpack/label.html`` (same
column order as ``labels.csv``). The model side is rebuilt exactly as
:mod:`tools.carrier_attribution_probe` scores it: cached embeddings, leave-one-track-out, and the
frozen operating point from ``results/carrier_attr/frozen_threshold.json`` -- nothing is re-tuned
here.

Reported:

* **identification precision** -- of moments the model ASSIGNED a name and the human named a real
  player on a correctly boxed carrier, the share that agree.
* **end-to-end precision** -- same, but ``WRONG_PLAYER_BOXED`` moments count as wrong: if the box is
  on the wrong player, a confident name is wrong no matter how good the re-id is.
* **carrier-selection error rate** -- ``WRONG_PLAYER_BOXED`` share, i.e. how often the pipeline
  picked the wrong player *before* identification even starts.
* **abstention breakdown** -- assigned / below-similarity / below-margin / no prediction row.
* precision split by **carrier distance band** and by whether the true player was
  **gallery-covered** (a player absent from the gallery cannot be identified correctly -- any
  assignment there is guaranteed wrong).

``UNKNOWN`` and ``NOT_A_PASS`` rows (and blank rows) are excluded from every rate and counted in the
exclusion table.

Run (CPU)::

    python -m tools.score_carrier_labels                       # default labels_filled.csv
    python -m tools.score_carrier_labels --labels path/to.csv
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from tools.carrier_attribution_probe import OUT_DIR, PROBE_MATCHES, assign, load_scored

#: Default export location of the labelling tool.
DEFAULT_LABELS = OUT_DIR / "labelpack" / "labels_filled.csv"
#: Answers that are not a player name.
UNKNOWN = "UNKNOWN"
WRONG_BOX = "WRONG_PLAYER_BOXED"
NOT_A_PASS = "NOT_A_PASS"
SPECIALS = (UNKNOWN, WRONG_BOX, NOT_A_PASS)
#: Carrier ball-distance bands (metres, left-closed).
DIST_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf))


def norm_name(name: object) -> str:
    """Canonical form of a player name for comparison (NFKC, casefolded, single-spaced)."""
    s = unicodedata.normalize("NFKC", str(name or "")).strip()
    return " ".join(s.split()).casefold()


def ascii_safe(text: object) -> str:
    """ASCII transliteration of a name, so the cp1252 Windows console can print it."""
    s = unicodedata.normalize("NFKD", str(text))
    return s.encode("ascii", "ignore").decode("ascii")


def load_labels(path: Path) -> pd.DataFrame:
    """Read a filled label CSV (BOM tolerated) with the label columns as strings."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"player_name": str, "notes": str,
                                                        "confidence_1to3": str})
    for col in ("player_name", "notes", "confidence_1to3"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["carrier_dist_m"] = pd.to_numeric(df["carrier_dist_m"], errors="coerce")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    return df


def load_predictions(matches: tuple[str, ...] = PROBE_MATCHES) -> pd.DataFrame:
    """Model answer per carrier moment at the frozen operating point.

    Args:
        matches: match ids with cached embeddings.

    Returns:
        ``match, chunk, frame, pred_player`` (``None`` = abstained) plus ``best_sim`` and ``margin``
        so the abstention reason can be split.
    """
    frozen = json.loads((OUT_DIR / "frozen_threshold.json").read_text(encoding="utf-8"))
    out: list[pd.DataFrame] = []
    for mid in matches:
        if not (OUT_DIR / f"{mid}_query_emb.npy").exists():
            continue
        scored = load_scored(mid)
        pred = assign(scored, frozen["min_sim"], frozen["min_margin"])
        out.append(pd.DataFrame({"match": mid, "chunk": scored["chunk"],
                                 "frame": scored["frame"].astype("Int64"),
                                 "pred_player": pred.to_numpy(),
                                 "best_sim": scored["best_sim"], "margin": scored["margin"]})
                   .assign(min_sim=frozen["min_sim"], min_margin=frozen["min_margin"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["match", "chunk", "frame", "pred_player", "best_sim", "margin"])


def gallery_players(matches: tuple[str, ...] = PROBE_MATCHES) -> dict[str, set[str]]:
    """Normalised set of players the gallery can even produce, per match."""
    out: dict[str, set[str]] = {}
    for mid in matches:
        path = OUT_DIR / f"{mid}_gallery.parquet"
        if path.exists():
            out[mid] = {norm_name(p) for p in pd.read_parquet(path)["player"].unique()}
    return out


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _band(dist: float) -> str:
    if pd.isna(dist):
        return "unknown"
    for lo, hi in DIST_BANDS:
        if lo <= dist < hi:
            return f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"
    return "unknown"


def score_labels(labels: pd.DataFrame, preds: pd.DataFrame,
                 gallery: dict[str, set[str]]) -> dict:
    """All reported rates from a filled label table and the model's answers.

    Args:
        labels: rows of the filled label CSV.
        preds: :func:`load_predictions` output (joined on ``match, chunk, frame``).
        gallery: per-match set of normalised gallery player names.

    Returns:
        Nested dict with ``exclusions``, ``headline``, ``abstention``, ``by_dist``,
        ``by_gallery``, ``by_confidence`` and the per-moment ``rows`` frame.
    """
    df = labels.copy()
    df["ans"] = df["player_name"].str.strip()
    excl = {
        "total": int(len(df)),
        "blank": int((df["ans"] == "").sum()),
        UNKNOWN: int((df["ans"] == UNKNOWN).sum()),
        NOT_A_PASS: int((df["ans"] == NOT_A_PASS).sum()),
    }
    judged = df[~df["ans"].isin(["", UNKNOWN, NOT_A_PASS])].copy()
    excl["judged"] = int(len(judged))

    judged = judged.merge(preds, on=["match", "chunk", "frame"], how="left", indicator=True)
    judged["joined"] = judged["_merge"] == "both"
    judged["assigned"] = judged["pred_player"].notna()
    judged["wrong_box"] = judged["ans"] == WRONG_BOX
    judged["truth"] = judged["ans"].map(norm_name)
    judged["pred"] = judged["pred_player"].map(norm_name)
    judged["correct"] = (~judged["wrong_box"]) & judged["assigned"] & (
        judged["truth"] == judged["pred"])
    judged["gallery_covered"] = [
        norm_name(a) in gallery.get(m, set()) for a, m in zip(judged["ans"], judged["match"])]
    judged["dist_band"] = judged["carrier_dist_m"].map(_band)

    good_box = judged[~judged["wrong_box"]]
    ided = good_box[good_box["assigned"]]
    e2e = judged[judged["assigned"]]
    headline = {
        "n_judged": int(len(judged)),
        "n_wrong_box": int(judged["wrong_box"].sum()),
        "carrier_selection_error_rate": _rate(int(judged["wrong_box"].sum()), len(judged)),
        "n_assigned_good_box": int(len(ided)),
        "identification_precision": _rate(int(ided["correct"].sum()), len(ided)),
        "n_assigned_all": int(len(e2e)),
        "end_to_end_precision": _rate(int(e2e["correct"].sum()), len(e2e)),
    }

    abst = judged[judged["joined"] & ~judged["assigned"]]
    no_emb = abst["best_sim"].isna()
    lo_sim = (~no_emb) & (abst["best_sim"] < abst["min_sim"])
    abstention = {
        "assigned": int(judged["assigned"].sum()),
        "abstain_below_similarity": int(lo_sim.sum()),
        "abstain_below_margin": int(len(abst) - int(lo_sim.sum()) - int(no_emb.sum())),
        "abstain_no_embedding": int(no_emb.sum()),
        "no_prediction_row": int((~judged["joined"]).sum()),
        "abstain_rate": _rate(int((~judged["assigned"]).sum()), len(judged)),
    }

    def split(frame: pd.DataFrame, key: str) -> list[dict]:
        rows = []
        for val, g in frame.groupby(key, dropna=False):
            a = g[g["assigned"]]
            rows.append({key: val, "n_judged": int(len(g)), "n_assigned": int(len(a)),
                         "precision": _rate(int(a["correct"].sum()), len(a)),
                         "assign_rate": _rate(len(a), len(g))})
        return sorted(rows, key=lambda r: str(r[key]))

    return {
        "exclusions": excl,
        "headline": headline,
        "abstention": abstention,
        "by_dist": split(good_box, "dist_band"),
        "by_gallery": split(good_box, "gallery_covered"),
        "by_confidence": split(good_box, "confidence_1to3"),
        "rows": judged,
    }


def print_report(res: dict) -> None:
    """Print the scoring report (ASCII only)."""
    ex, hl, ab = res["exclusions"], res["headline"], res["abstention"]
    print("=== carrier label scoring ===")
    print(f"labelled rows            : {ex['total']}")
    print(f"  excluded blank         : {ex['blank']}")
    print(f"  excluded UNKNOWN       : {ex[UNKNOWN]}")
    print(f"  excluded NOT_A_PASS    : {ex[NOT_A_PASS]}")
    print(f"  judged                 : {ex['judged']}")
    print()
    print(f"carrier-selection error  : {hl['carrier_selection_error_rate']} "
          f"({hl['n_wrong_box']}/{hl['n_judged']} WRONG_PLAYER_BOXED)")
    print(f"identification precision : {hl['identification_precision']} "
          f"(n={hl['n_assigned_good_box']} assigned on a correct box)")
    print(f"end-to-end precision     : {hl['end_to_end_precision']} "
          f"(n={hl['n_assigned_all']}, wrong box counted wrong)")
    print()
    print("abstention breakdown (judged moments):")
    for k in ("assigned", "abstain_below_similarity", "abstain_below_margin",
              "abstain_no_embedding", "no_prediction_row"):
        print(f"  {k:<26}: {ab[k]}")
    print(f"  {'abstain_rate':<26}: {ab['abstain_rate']}")
    for title, key in (("by carrier distance", "by_dist"), ("by gallery coverage", "by_gallery"),
                       ("by human confidence", "by_confidence")):
        print()
        print(f"{title}:")
        rows = res[key]
        if not rows:
            print("  (none)")
            continue
        head = list(rows[0])
        print("  " + "  ".join(f"{c:<14}" for c in head))
        for r in rows:
            print("  " + "  ".join(f"{str(r[c]):<14}" for c in head))
    bad = res["rows"]
    bad = bad[bad["assigned"] & ~bad["correct"] & ~bad["wrong_box"]]
    if len(bad):
        print()
        print(f"sample identification errors (truth -> predicted), {len(bad)} total:")
        for r in bad.head(8).itertuples(index=False):
            print(f"  {ascii_safe(r.player_name):<24} -> {ascii_safe(r.pred_player):<24} "
                  f"sim={r.best_sim:.3f} d={r.carrier_dist_m}m")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS), help="filled label csv")
    ap.add_argument("--out", default=str(OUT_DIR / "label_scoring.csv"),
                    help="per-moment joined output csv")
    a = ap.parse_args()
    labels = load_labels(Path(a.labels))
    res = score_labels(labels, load_predictions(), gallery_players())
    print_report(res)
    keep = ["id", "match", "chunk", "frame", "carrier_dist_m", "player_name", "confidence_1to3",
            "pred_player", "best_sim", "margin", "assigned", "wrong_box", "correct",
            "gallery_covered", "dist_band"]
    rows = res["rows"]
    rows[[c for c in keep if c in rows]].to_csv(a.out, index=False, encoding="utf-8")
    print()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
