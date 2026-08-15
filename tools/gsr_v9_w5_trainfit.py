"""W5 train fit: false-positive rate of the concurrency-aware second-keeper rule on GSR-TRAIN GT.

GT carries exactly one keeper identity per side, so ANY non-keeper the rule promotes on train is a
false positive. No new threshold is introduced: the |x| / penalty-share gates are W4's frozen
GK_MIN_ABSX_M / GK_MIN_PEN_FRAC, and the new test ("deeper than every keeper on the pitch at the
same time") has no parameter at all.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.gsr_score import DEFAULT_DATA_DIR, GK_MIN_ABSX_M, GK_MIN_PEN_FRAC, penalty_frac
from tools.gsr_deleak import split_names


def identities(seq_dir: Path) -> dict[int, dict]:
    """Per GT identity: role, median pitch x, penalty share, frame set."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    rows: dict[int, list] = defaultdict(list)
    for a in gt["annotations"]:
        bp, at = a.get("bbox_pitch"), a.get("attributes") or {}
        if not bp or at.get("role") not in {"player", "goalkeeper", "referee"}:
            continue
        rows[int(a["track_id"])].append((bp["x_bottom_middle"], bp["y_bottom_middle"],
                                         at["role"], a["image_id"]))
    out = {}
    for tid, rs in rows.items():
        x = np.array([r[0] for r in rs], float)
        y = np.array([r[1] for r in rs], float)
        out[tid] = {"role": rs[0][2], "med_x": float(np.median(x)),
                    "pen": float(penalty_frac(x, y)), "frames": {r[3] for r in rs}, "n": len(rs)}
    return out


def fire(info: dict[int, dict], *, dominance: bool) -> list[int]:
    """Track ids the second-keeper rule promotes (see module docstring for the rule)."""
    out = []
    for half in (-1.0, 1.0):
        here = {t: i for t, i in info.items() if np.sign(i["med_x"]) == half}
        keepers = [t for t, i in here.items() if i["role"] == "goalkeeper"]
        for t, i in here.items():
            if i["role"] != "player":
                continue
            if abs(i["med_x"]) < GK_MIN_ABSX_M or i["pen"] < GK_MIN_PEN_FRAC:
                continue
            conc = [k for k in keepers if here[k]["frames"] & i["frames"]]
            if not conc:
                out.append(t)
            elif dominance and all(abs(i["med_x"]) > abs(here[k]["med_x"]) for k in conc):
                out.append(t)
    return out


def main() -> None:
    """Report train-GT firing counts (every firing is a false positive by construction)."""
    seqs = split_names(DEFAULT_DATA_DIR, "train")
    tot = {"seqs": len(seqs), "cand": 0, "fire_strict": 0, "fire_dom": 0, "keepers": 0,
           "players": 0, "rows_fire_dom": 0, "per_seq": {}}
    for s in seqs:
        info = identities(DEFAULT_DATA_DIR / s)
        tot["keepers"] += sum(i["role"] == "goalkeeper" for i in info.values())
        tot["players"] += sum(i["role"] == "player" for i in info.values())
        cand = [t for t, i in info.items()
                if i["role"] == "player" and abs(i["med_x"]) >= GK_MIN_ABSX_M
                and i["pen"] >= GK_MIN_PEN_FRAC]
        fs, fd = fire(info, dominance=False), fire(info, dominance=True)
        tot["cand"] += len(cand)
        tot["fire_strict"] += len(fs)
        tot["fire_dom"] += len(fd)
        tot["rows_fire_dom"] += sum(info[t]["n"] for t in fd)
        if fs or fd:
            tot["per_seq"][s] = {"cand": len(cand), "strict": fs, "dom": fd,
                                 "rows_dom": sum(info[t]["n"] for t in fd)}
    print(json.dumps(tot, indent=1))
    Path("results/gsr_benchmark/gsr_v9_w5_trainfit.json").write_text(
        json.dumps(tot, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
