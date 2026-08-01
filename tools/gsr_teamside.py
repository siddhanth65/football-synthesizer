"""N1b -- the second family of team-side resolvers, and the ceiling that closes the question.

``results/GSR_DELEAK.md`` section 3 left the cluster -> ``left``/``right`` permutation as the
binding constraint on a legitimate SoccerNet-GSR number: the positional resolver
(:func:`eval.gsr_score.resolve_team_map_free`, "the cluster with the smaller mean pitch-x is left")
is 45/49 on the official test split and each miss annihilates its clip (-32.94 GS-HOTA). Three
families were named as unexplored. This module grades all three on a development pool that never
includes the test split, and measures the one number that decides the whole question: **what each
rule scores when it is handed the ground-truth positions instead of ours.**

CLI::

    python -m tools.gsr_teamside --invariance   # family 1: is the solve side-permutation sensitive?
    python -m tools.gsr_teamside --ceiling      # oracle ceiling of every rule, on GT, dev only
    python -m tools.gsr_teamside --dev          # every rule on our positions, valid + train probe
    python -m tools.gsr_teamside --windows      # the same on sub-clips (a stress pool with failures)
    python -m tools.gsr_teamside --gklink       # can a detected keeper be linked to a kit cluster?
    python -m tools.gsr_teamside --freeze       # pre-declare the choice, before any test read
    python -m tools.gsr_teamside --verify       # the single pre-declared test-split read
    python -m tools.gsr_teamside --demo         # runnable self-check, no data needed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import (
    CENTRE_SHIFT_X,
    CENTRE_SHIFT_Y,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    GSR_DIST_TOL_M,
    load_gt_people_by_frame,
    resolve_team_map,
    to_centred,
)
from tools.gsr_deleak import TEST_OUT_DIR, TRAIN_PROBE_DIR, split_names

logger = logging.getLogger("gsr_teamside")

#: Where this study's artifacts land.
RESULTS_DIR = Path("results/gsr_benchmark/teamside")
#: The pre-declared choice, written before the test split is read.
FROZEN_PATH = Path("results/gsr_teamside_frozen.json")
#: Train sequences whose kit clustering is degenerate (GSR_DELEAK section 3) -- not gradable.
DEGENERATE_TRAIN = frozenset(f"SNGS-{i:03d}" for i in range(60, 78))
#: Ball-to-player distance counted as possession, metres.
POSS_M = 6.0
#: Frames over which ball displacement is measured (25 fps -> 0.4 s).
POSS_LAG = 10
#: Half the pitch length, metres -- the ``pitch_x`` value that separates the two goals.
HALF_X = 52.5


# === The rules ===================================================================================
def side_scores(df: pd.DataFrame) -> dict[str, float]:
    """Every candidate side rule on one positions table, as a signed score.

    The convention is uniform: **a negative score means cluster 0 defends the left goal**. Missing
    evidence yields ``nan`` rather than a guess, so each rule is graded only where it applies.

    Args:
        df: Positions table for one sequence or sub-clip (``role``, ``team``, ``pitch_x``).

    Returns:
        Rule name -> signed score, plus the ``n_*`` evidence counters used for margin analysis.
    """
    pl = df[(df["role"] == "player") & df["team"].isin([0, 1]) & np.isfinite(df["pitch_x"])]
    out: dict[str, float] = {"n_rows": float(len(pl)), "n_frames_cal": float(pl["frame"].nunique()),
                             "cal_cov": float(pl["frame"].nunique())
                             / max(df["frame"].nunique(), 1)}
    if pl.empty or pl["team"].nunique() < 2:
        return out
    g = pl.groupby("team")["pitch_x"]
    out["meanx"] = float(g.mean()[0] - g.mean()[1])          # the incumbent
    out["medx"] = float(g.median()[0] - g.median()[1])
    out["x_span"] = float(pl["pitch_x"].quantile(0.95) - pl["pitch_x"].quantile(0.05))

    cent = pl.groupby(["frame", "team"])["pitch_x"].mean().unstack()
    both = cent.dropna()
    if len(both) > 5:
        d = both[0] - both[1]
        out["frame_mean"] = float(d.mean())
        out["frame_vote"] = float(np.sign(d).mean())
        sd = float(d.std(ddof=1))
        out["frame_t"] = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else out["frame_mean"]
    n0, n1 = cent[0].dropna() if 0 in cent else pd.Series(dtype=float), \
        cent[1].dropna() if 1 in cent else pd.Series(dtype=float)
    if len(n0) > 5 and len(n1) > 5:
        se = float(np.sqrt(n0.var(ddof=1) / len(n0) + n1.var(ddof=1) / len(n1)))
        gap = float(n0.mean() - n1.mean())
        out["cent_t"] = gap / se if se > 0 else gap

    for k in (1, 3):  # the defensive line / the front line
        lo = pl.groupby(["frame", "team"])["pitch_x"].apply(lambda s, k=k: s.nsmallest(k).mean())
        hi = pl.groupby(["frame", "team"])["pitch_x"].apply(lambda s, k=k: s.nlargest(k).mean())
        lo, hi = lo.unstack(), hi.unstack()
        if {0, 1} <= set(lo.columns):
            out[f"deep{k}"] = float((lo[0] - lo[1]).mean())
            out[f"high{k}"] = float((hi[0] - hi[1]).mean())
    lowest = pl.loc[pl.groupby("frame")["pitch_x"].idxmin(), "team"]
    highest = pl.loc[pl.groupby("frame")["pitch_x"].idxmax(), "team"]
    out["extreme_owner"] = float(((highest == 0).mean() - (highest == 1).mean()
                                  - (lowest == 0).mean() + (lowest == 1).mean()) / 2)

    ball = df[(df["role"] == "ball") & np.isfinite(df["pitch_x"])].groupby("frame")["pitch_x"].mean()
    out["n_ball_frames"] = float(len(ball))
    if len(ball) >= 10:
        j = pl.join(ball.rename("bx"), on="frame").dropna(subset=["bx"])
        if len(j) and j["team"].nunique() == 2:
            rel = j.assign(r=j["pitch_x"] - j["bx"]).groupby("team")["r"].mean()
            out["ball_rel"] = float(rel[0] - rel[1])
    out.update(_drift_scores(pl, ball))

    gkrows = df[(df["role"] == "goalkeeper") & np.isfinite(df["pitch_x"])]
    gk = gkrows.groupby("frame")["pitch_x"].mean()
    out["n_gk_frames"] = float(len(gk))
    if len(gk) >= 10:
        j = pl.join(gk.rename("gx"), on="frame").dropna(subset=["gx"])
        if len(j) and j["team"].nunique() == 2:
            # a cluster close to a keeper standing at the left goal is defending left
            signed = (j["pitch_x"] - j["gx"]).abs() * np.where(j["gx"] < HALF_X, 1.0, -1.0)
            m = signed.groupby(j["team"]).mean()
            out["gk_anchor"] = float(m[0] - m[1])
    out["gk_self"] = _gk_self(gkrows)
    return out


def _gk_self(gkrows: pd.DataFrame) -> float:
    """The definitional rule: a keeper's own team defends the goal he stands in.

    This is what the GSR ``left``/``right`` attribute *means*, so on ground truth it is exact. On our
    positions it needs the keeper's team, and the two-way kit clustering does not supply it -- see
    :func:`gk_link`.
    """
    g = gkrows[gkrows["team"].isin([0, 1])]
    if g.empty:
        return float("nan")
    m = g.groupby("team")["pitch_x"].mean()
    if len(m) == 2:
        return float(m[0] - m[1])
    c = int(m.index[0])                                  # only one keeper seen: use his own half
    return float((m.iloc[0] - HALF_X) * (1.0 if c == 0 else -1.0))


def _drift_scores(pl: pd.DataFrame, ball: pd.Series) -> dict[str, float]:
    """Attack direction: which way does the ball travel while each cluster holds it?

    A team attacking rightwards defends the left goal, so the cluster with the more positive mean
    ball displacement is ``left``. This is the only candidate that is not a depth statistic and so
    the only one that could survive a clip whose depth ordering inverts.
    """
    out: dict[str, float] = {"n_poss": 0.0}
    if len(ball) < 20 or pl.empty:
        return out
    bx = ball.sort_index()
    dx = bx.shift(-POSS_LAG) - bx
    j = pl.join(bx.rename("bx"), on="frame").dropna(subset=["bx"])
    if j.empty:
        return out
    near = j.assign(d=(j["pitch_x"] - j["bx"]).abs()).query("d <= @POSS_M")
    if near.empty:
        return out
    own = near.loc[near.groupby("frame")["d"].idxmin(), ["frame", "team"]] \
        .set_index("frame")["team"]
    move = dx.reindex(own.index).dropna()
    own = own.reindex(move.index)
    out["n_poss"] = float(len(move))
    if len(move) < 10 or own.nunique() < 2:
        return out
    t = pd.DataFrame({"team": own.to_numpy(), "dx": move.to_numpy()})
    m = t.groupby("team")["dx"].mean()
    out["poss_drift"] = float(m[1] - m[0])
    out["poss_share"] = float(np.sign(t["dx"]).groupby(t["team"]).mean().pipe(lambda s: s[1] - s[0]))
    owner = own.to_dict()
    d2 = j.assign(owner=j["frame"].map(owner)).dropna(subset=["owner"])
    d2 = d2[d2["team"] != d2["owner"]]  # the defending block, relative to the ball
    if len(d2) and d2["team"].nunique() == 2:
        r = d2.assign(r=d2["pitch_x"] - d2["bx"]).groupby("team")["r"].mean()
        out["poss_rel"] = float(r[0] - r[1])
    return out


def gt_positions(seq_dir: Path) -> pd.DataFrame:
    """Ground-truth objects of one sequence as a positions-shaped table.

    The GT sides ``left``/``right`` are re-coded as clusters 0/1 so the very same rules in
    :func:`side_scores` can be run on them; a rule that is wrong here cannot be rescued by any
    estimator of ours.
    """
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    idx = {i["image_id"]: int(Path(i["file_name"]).stem) - 1 for i in gt["images"]}
    side = {"left": 0, "right": 1}
    rows = []
    for a in gt["annotations"]:
        bp, at = a.get("bbox_pitch"), a.get("attributes") or {}
        if not bp or a["image_id"] not in idx:
            continue
        rows.append((idx[a["image_id"]], at.get("role") or "ball",
                     side.get(at.get("team"), -1),
                     # GT bbox_pitch is the CENTRED frame; our positions are [0,105]x[0,68]. The
                     # depth rules are shift-invariant but gk_anchor/gk_self are not, so undo it.
                     bp["x_bottom_middle"] + CENTRE_SHIFT_X, bp["y_bottom_middle"] + CENTRE_SHIFT_Y))
    return pd.DataFrame(rows, columns=["frame", "role", "team", "pitch_x", "pitch_y"])


# === Grading =====================================================================================
def grade(rows: list[dict]) -> dict:
    """Accuracy of every rule over graded rows carrying ``gt0`` (``'left'`` or ``'right'``)."""
    d = pd.DataFrame(rows)
    y = d["gt0"] == "left"
    out = {}
    for c in sorted(set(d.columns) - {"seq", "gt0", "k", "w"}):
        if c.startswith(("n_", "cal_", "x_span")):
            continue
        v = pd.to_numeric(d[c], errors="coerce")
        m = v.notna()
        ok = (v < 0) == y
        out[c] = {"n": int(m.sum()), "correct": int((ok & m).sum()),
                  "accuracy": round(float((ok & m).sum() / max(int(m.sum()), 1)), 4),
                  "wrong": d.loc[m & ~ok, "seq"].tolist()[:12]}
    return out


def auc(score: np.ndarray, correct: np.ndarray) -> float:
    """Probability a correct decision carries a higher confidence than a wrong one (ties 0.5)."""
    m = np.isfinite(score)
    pos, neg = score[m & correct], score[m & ~correct]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = np.subtract.outer(pos, neg)
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def dev_names(data_dir: Path) -> dict[str, Path]:
    """The development pool: valid-58 plus every non-degenerate train sequence already extracted."""
    pool = {n: DEFAULT_OUT_DIR / "positions" for n in split_names(data_dir, "valid")}
    for n in split_names(data_dir, "train"):
        if n in DEGENERATE_TRAIN:
            continue
        if (TRAIN_PROBE_DIR / "positions" / f"{n}.parquet").exists():
            pool[n] = TRAIN_PROBE_DIR / "positions"
    return pool


def score_pool(pool: dict[str, Path], data_dir: Path, *, windows: int = 1) -> list[dict]:
    """Run every rule over a pool of sequences (optionally split into ``windows`` sub-clips)."""
    rows = []
    for seq, pos_dir in sorted(pool.items()):
        df = pd.read_parquet(pos_dir / f"{seq}.parquet")
        gt0 = resolve_team_map(df, load_gt_people_by_frame(data_dir / seq))[0]
        f0, f1 = int(df["frame"].min()), int(df["frame"].max())
        for w in range(windows):
            lo = f0 + (f1 - f0 + 1) * w / windows
            hi = f0 + (f1 - f0 + 1) * (w + 1) / windows
            sub = df if windows == 1 else df[(df["frame"] >= lo) & (df["frame"] < hi)]
            rows.append({"seq": seq, "gt0": gt0, "k": windows, "w": w, **side_scores(sub)})
        logger.info("scored %s (%s)", seq, pos_dir.parent.name)
    return rows


# === Family 1: is the identity solve side-sensitive at all? ======================================
def solver_invariance(names: list[str], data_dir: Path, out_dir: Path) -> dict:
    """Solve each bundle under BOTH cluster -> side permutations and compare the evidence fit.

    The N1b brief's primary candidate is "solve both ways, keep the permutation with the better
    internal fit". The solver consumes :attr:`Identity.team` (a cluster index) and never the side
    string, so this is expected to return a bit-identical objective; measured rather than argued.
    """
    from eval.gsr_identity import load_bundles, solve_bundle_scored  # noqa: PLC0415
    from generator.identity_solve import SolverConfig  # noqa: PLC0415
    from tools.gsr_deleak import SOLVER_CONFIG, retarget  # noqa: PLC0415

    cfg = SolverConfig.load(SOLVER_CONFIG)
    bundles = load_bundles(data_dir, out_dir, names, votes_subdir="koshkina_percrop_votes",
                           cache_subdir="identity_bundles_percrop")
    per_seq = {}
    for name in names:
        got = {}
        for tag, sides in (("A", {0: "left", 1: "right"}), ("B", {0: "right", 1: "left"})):
            b = retarget(bundles[name], roster="self", sides=sides)
            assign, conf = solve_bundle_scored(b, cfg)
            got[tag] = (assign, np.asarray(conf), [(i.team, i.number) for i in b["identities"]])
        per_seq[name] = {
            "same_assignment": got["A"][0] == got["B"][0],
            "max_abs_posterior_delta": float(np.abs(got["A"][1] - got["B"][1]).max()),
            "sum_posterior_A": float(got["A"][1].sum()), "sum_posterior_B": float(got["B"][1].sum()),
            "same_roster_slots": got["A"][2] == got["B"][2],
            "n_named": int(sum(a is not None for a in got["A"][0])),
        }
        logger.info("%s: %s", name, per_seq[name])
    return {"per_seq": per_seq, "verdict": (
        "the solver objective is invariant under the cluster->side permutation: Identity.key (the "
        "side string) is never read, only Identity.team (the cluster index). Every criterion the "
        "brief lists -- total objective, summed posteriors, self-roster/OCR consistency, "
        "coverage-at-abstention -- is a function of that objective, so all of them are invariant "
        "too. Solve-both-and-vote carries exactly zero bits.")}


# === Goalkeeper linkage ==========================================================================
def gk_link(pool: dict[str, Path], data_dir: Path) -> dict:
    """How often does a detected keeper's kit cluster equal his own team's cluster?

    The GSR ``left``/``right`` attribute is anchored on the keeper (his side is the goal he stands
    in), so linking a keeper to a kit cluster would give a rule immune to depth inversions. Measured
    by matching our keeper detections to GT keepers and comparing the kit cluster's resolved side
    with the GT keeper's own side.
    """
    rows = []
    for seq, pos_dir in sorted(pool.items()):
        df = pd.read_parquet(pos_dir / f"{seq}.parquet")
        gt_people = load_gt_people_by_frame(data_dir / seq)
        tmap = resolve_team_map(df, gt_people)
        gk = df[(df["role"] == "goalkeeper") & df["team"].isin([0, 1])
                & np.isfinite(df["pitch_x"])]
        n = ok = 0
        for r in gk.itertuples():
            cands = [g for g in gt_people.get(int(r.frame), [])
                     if g[2] == "goalkeeper" and g[3] in ("left", "right")]
            if not cands:
                continue
            cx, cy = to_centred(float(r.pitch_x), float(r.pitch_y))
            dist = [float(np.hypot(g[0] - cx, g[1] - cy)) for g in cands]
            j = int(np.argmin(dist))
            if dist[j] > GSR_DIST_TOL_M:
                continue
            n += 1
            ok += tmap[int(r.team)] == cands[j][3]
        if n:
            rows.append({"seq": seq, "n": n, "ok": ok, "acc": round(ok / n, 4)})
    d = pd.DataFrame(rows)
    return {"n_sequences": len(d), "n_matched_gk_rows": int(d["n"].sum()),
            "row_accuracy": round(float(d["ok"].sum() / max(d["n"].sum(), 1)), 4),
            "sequence_majority_correct": int((d["acc"] > 0.5).sum()),
            "per_seq": d.to_dict("records")}


# === Self-check ==================================================================================
def _demo() -> None:
    """Self-check: the sign convention, the ball rule, and the GT re-coding (asserts; runnable)."""
    frames = np.repeat(np.arange(40), 6)
    df = pd.DataFrame({
        "frame": frames,
        "role": ["player"] * 240,
        # cluster 0 sits deep left, cluster 1 high right -> cluster 0 defends left -> score < 0
        "team": [0, 0, 0, 1, 1, 1] * 40,
        "pitch_x": [20.0, 25.0, 30.0, 70.0, 75.0, 80.0] * 40,
        "pitch_y": [34.0] * 240,
    })
    s = side_scores(df)
    for rule in ("meanx", "medx", "frame_mean", "frame_vote", "frame_t", "cent_t", "deep1",
                 "high1", "extreme_owner"):
        assert s[rule] < 0, (rule, s[rule])
    flipped = side_scores(df.assign(team=1 - df["team"]))
    assert flipped["meanx"] > 0 and flipped["extreme_owner"] > 0, flipped

    # a keeper parked on the RIGHT goal line must not drag the players-only rules
    gkrow = pd.DataFrame({"frame": [0], "role": ["goalkeeper"], "team": [0], "pitch_x": [104.0],
                          "pitch_y": [34.0]})
    assert side_scores(pd.concat([df, gkrow]))["meanx"] == s["meanx"]

    # ball rule: cluster 0 sits left of the ball -> negative
    ball = pd.DataFrame({"frame": np.arange(40), "role": "ball", "team": -1,
                         "pitch_x": 50.0, "pitch_y": 34.0})
    assert side_scores(pd.concat([df, ball]))["ball_rel"] < 0

    got = grade([{"seq": "A", "gt0": "left", "meanx": -3.0},
                 {"seq": "B", "gt0": "right", "meanx": -1.0}])["meanx"]
    assert got == {"n": 2, "correct": 1, "accuracy": 0.5, "wrong": ["B"]}, got
    assert auc(np.array([3.0, 2.0, 1.0]), np.array([True, True, False])) == 1.0
    print("gsr_teamside demo OK: sign convention holds, keepers are excluded from the depth rules, "
          "the ball rule reads the ball, grading and AUC agree with hand counts")


# === Pre-declaration =============================================================================
def _declaration(data_dir: Path) -> dict:
    """The choice, its evidence and its known ceiling -- written before the test split is read."""
    def load(name: str) -> dict:
        path = RESULTS_DIR / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    ceiling = load("teamside_ceiling.json").get("rules", {})
    dev = load("teamside_dev_k1.json").get("rules", {})
    return {
        "declared_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "rule": "meanx",
        "implementation": "eval.gsr_score.resolve_team_map_free (UNCHANGED from N1)",
        "decision": (
            "keep the incumbent. Family 1 (solve both permutations and vote) carries provably zero "
            "bits; families 2 and 3 (flow, keeper, ensembles) were graded on the dev pool and none "
            "beat it on whole clips. Among the rules our pipeline can actually compute, meanx also "
            "has the highest accuracy on ground-truth positions (112/115) -- i.e. the highest score "
            "reachable if our detections, clustering and calibration were exact."),
        "known_higher_ceiling": (
            "gk_self -- 'a keeper's own team defends the goal he stands in' -- is 113/113 on "
            "ground-truth positions, including all three clips where the depth ordering inverts. It "
            "is not shippable: it needs the keeper's TEAM, and the two-way kit clustering puts a "
            "detected keeper in his own team's cluster in only 24.4% of matched rows (13 of 57 "
            "sequences by majority). Linking a keeper to a team is the named upgrade path and it is "
            "an appearance problem, not a geometry one."),
        "margin": (
            "NONE SHIPPED. No confidence separates right from wrong on whole clips, because the "
            "residual failures are not estimation failures: they are clips whose ground-truth "
            "geometry itself inverts, where the evidence is strong and points the wrong way."),
        "dev_pool": sorted(dev_names(data_dir)),
        "dev_accuracy_meanx": dev.get("meanx", {}),
        "gt_ceiling_dev_meanx": ceiling.get("meanx", {}),
        "verification": ("one read of the official test split, side-correctness of `meanx` only; "
                         "no other rule is graded there and no package is rebuilt unless the "
                         "accuracy improves on the incumbent's 45/49"),
    }


# === CLI =========================================================================================
def _write(path: Path, payload: dict) -> None:
    """Write an indented UTF-8 JSON payload and log where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--invariance", action="store_true", help="family 1: solve under both maps")
    ap.add_argument("--ceiling", action="store_true", help="oracle ceiling of every rule on GT")
    ap.add_argument("--dev", action="store_true", help="every rule on our positions (dev pool)")
    ap.add_argument("--windows", type=int, default=0, help="stress pool: N sub-clips per sequence")
    ap.add_argument("--gklink", action="store_true", help="keeper kit-cluster linkage rate")
    ap.add_argument("--freeze", action="store_true", help="pre-declare the choice before the test")
    ap.add_argument("--verify", action="store_true", help="the single pre-declared test read")
    ap.add_argument("--demo", action="store_true", help="runnable self-check")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if args.invariance:
        names = [n for n in split_names(args.data_dir, "valid")][:6]
        _write(RESULTS_DIR / "teamside_invariance.json",
               solver_invariance(names, args.data_dir, args.out_dir))
        return
    if args.ceiling:
        rows = []
        for split in ("train", "valid"):  # dev only -- the test split is not read here
            for seq in split_names(args.data_dir, split):
                rows.append({"seq": seq, "split": split, "gt0": "left",
                             **side_scores(gt_positions(args.data_dir / seq))})
                logger.info("ceiling %s", seq)
        _write(RESULTS_DIR / "teamside_ceiling.json",
               {"note": "GT sides re-coded as clusters 0/1, so 'left' is always cluster 0 and a "
                        "correct rule scores negative", "rules": grade(rows), "per_seq": rows})
        return
    if args.gklink:
        _write(RESULTS_DIR / "teamside_gklink.json", gk_link(dev_names(args.data_dir),
                                                             args.data_dir))
        return
    if args.dev or args.windows:
        k = args.windows or 1
        pool = dev_names(args.data_dir)
        rows = score_pool(pool, args.data_dir, windows=k)
        payload: dict = {"n_sequences": len(pool), "windows": k, "rules": grade(rows),
                         "per_row": rows}
        if k > 1:  # margin analysis needs failures, and only the stress pool has enough
            d = pd.DataFrame(rows)
            ok = np.asarray((pd.to_numeric(d["meanx"], errors="coerce") < 0) == (d["gt0"] == "left"))
            cand = {"abs_meanx": d["meanx"].abs(), "abs_frame_t": d.get("frame_t", pd.Series()).abs(),
                    "abs_cent_t": d.get("cent_t", pd.Series()).abs(), "cal_cov": d["cal_cov"],
                    "n_frames_cal": d["n_frames_cal"]}
            payload["margin_auc"] = {k2: round(auc(np.asarray(v, float), ok), 4)
                                     for k2, v in cand.items() if len(v)}
        _write(RESULTS_DIR / f"teamside_dev_k{k}.json", payload)
        return
    if args.freeze:
        _write(FROZEN_PATH, _declaration(args.data_dir))
        return
    if args.verify:
        frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        names = split_names(args.data_dir, "test")
        pool = {n: TEST_OUT_DIR / "positions" for n in names}  # the test split has its own out dir
        rows = score_pool(pool, args.data_dir, windows=1)
        res = grade(rows)[frozen["rule"]]  # the frozen rule only -- no fishing over the others
        _write(RESULTS_DIR / "teamside_verify.json", {"frozen": frozen, "test": res})
        logger.info("VERIFY %s on test-%d: %d/%d = %.4f  wrong %s", frozen["rule"], len(names),
                    res["correct"], res["n"], res["accuracy"], res["wrong"])
        return
    ap.error("choose one of --invariance / --ceiling / --dev / --windows N / --gklink / --freeze"
             " / --verify")


if __name__ == "__main__":
    main()
