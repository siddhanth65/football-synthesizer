"""Scrape the FifaPhy app (https://fifaphy.vercel.app) — a static-JS app whose data lives in JS files.

The app embeds everything client-side as ``const NAME = {...};`` blocks (no API/auth). This downloads the
data bundles and parses each top-level object out with ``json.JSONDecoder.raw_decode`` (robust to trailing
code), so we can inspect the schema and export the physical-performance tables to parquet/CSV for use as
ground-truth against our CV-derived metrics (2026 World Cup = national-team scope).

Usage:
    python tools/scrape_fifaphy.py --explore                 # map the structure
    python tools/scrape_fifaphy.py --export outputs/fifaphy  # dump tables
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://fifaphy.vercel.app"
DATA_FILES = ["data.js", "catalog.js", "posreal.js", "ratings.js"]


def fetch(name: str, cache_dir: Path) -> str:
    """Download ``name`` from the app (cached locally) and return its text."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / name
    if local.exists():
        return local.read_text(encoding="utf-8")
    req = Request(f"{BASE}/{name}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r:  # noqa: S310 - fixed trusted host
        txt = r.read().decode("utf-8", "replace")
    local.write_text(txt, encoding="utf-8")
    return txt


def parse_consts(txt: str) -> dict:
    """Extract every ``const/var/let/window.NAME = <json>`` top-level object/array from a JS file."""
    out: dict[str, object] = {}
    dec = json.JSONDecoder()
    for m in re.finditer(r'(?m)^(?:const|let|var|window\.)\s*([\w.]+)\s*=\s*(?=[\[{])', txt):
        name = m.group(1)
        try:
            val, _ = dec.raw_decode(txt, m.end())
            out[name] = val
        except json.JSONDecodeError:
            continue
    return out


def _summarise(name: str, val, depth: int = 0, max_depth: int = 3) -> list[str]:
    pad = "  " * depth
    lines = []
    if isinstance(val, dict):
        keys = list(val.keys())
        lines.append(f"{pad}{name}: dict n={len(keys)} keys[:6]={keys[:6]}")
        if depth < max_depth and keys:
            lines += _summarise(f"[{keys[0]!r}]", val[keys[0]], depth + 1, max_depth)
    elif isinstance(val, list):
        lines.append(f"{pad}{name}: list len={len(val)}")
        if depth < max_depth and val:
            lines += _summarise("[0]", val[0], depth + 1, max_depth)
    else:
        s = str(val)
        lines.append(f"{pad}{name}: {type(val).__name__} = {s[:80]}")
    return lines


def export_tables(tournaments: dict, out_dir: Path) -> list[Path]:
    """Flatten each tournament's player-match physical records (+ matches) to parquet + CSV."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tk, t in tournaments.items():
        data = t.get("data", {})
        matches = data.get("matches", {})
        mdf = pd.DataFrame([{"matchId": mid, **m} for mid, m in matches.items()])
        rows = []
        for r in data.get("records", []):
            if not isinstance(r, dict):
                continue
            base = {k: r.get(k) for k in
                    ("playerId", "name", "teamId", "teamName", "teamCode", "matchId", "position")}
            rows.append({**base, **(r.get("metrics") or {})})
        pdf = pd.DataFrame(rows)
        if not pdf.empty and not mdf.empty:  # attach match context for easy filtering
            pdf = pdf.merge(mdf[["matchId", "home", "away", "date", "stage", "group"]],
                            on="matchId", how="left")
        for name, df in (("player_match", pdf), ("matches", mdf)):
            if df.empty:
                continue
            for ext, writer in ((".parquet", df.to_parquet), (".csv", df.to_csv)):
                p = out_dir / f"{tk}_{name}{ext}"
                writer(p, index=False)
                written.append(p)
        print(f"{tk}: {len(pdf)} player-match rows x {pdf.shape[1] if not pdf.empty else 0} cols, "
              f"{len(mdf)} matches")
    return written


def export_posreal_ratings(bundles: dict, out_dir: Path) -> list[Path]:
    """Flatten POSREAL (per-match position labels) and RATINGS (per-match ratings) to long tables."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    pos = bundles.get("posreal.js", {}).get("POSREAL", {}).get("players", {})
    if pos:
        rows = [{"playerId": pid, "matchId": mid, "position": p,
                 "mainPosition": rec.get("main"), "line": rec.get("line")}
                for pid, rec in pos.items() for mid, p in (rec.get("m") or {}).items()]
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "posreal_player_match.parquet", index=False)
        df.to_csv(out_dir / "posreal_player_match.csv", index=False)
        written += [out_dir / "posreal_player_match.parquet", out_dir / "posreal_player_match.csv"]
        print(f"posreal: {len(df)} player-match position labels")
    rat = bundles.get("ratings.js", {}).get("RATINGS", {}).get("byPlayer", {})
    if rat:
        rows = [{"playerId": pid, "matchId": mid, "rating": r}
                for pid, rec in rat.items() for mid, r in (rec.get("m") or {}).items()]
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "ratings_player_match.parquet", index=False)
        df.to_csv(out_dir / "ratings_player_match.csv", index=False)
        written += [out_dir / "ratings_player_match.parquet", out_dir / "ratings_player_match.csv"]
        print(f"ratings: {len(df)} player-match ratings")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="outputs/fifaphy/_cache", help="where to cache the .js files")
    ap.add_argument("--explore", action="store_true", help="print the parsed structure and exit")
    ap.add_argument("--export", default=None, help="dir to write extracted tables (parquet + csv)")
    args = ap.parse_args()

    cache = Path(args.cache)
    bundles = {f: parse_consts(fetch(f, cache)) for f in DATA_FILES}
    if args.explore:
        for fname, consts in bundles.items():
            print(f"\n===== {fname}: {list(consts)} =====")
            for name, val in consts.items():
                print("\n".join(_summarise(name, val)))
    if args.export:
        out = Path(args.export)
        written = export_tables(bundles["data.js"].get("TOURNAMENTS", {}), out)
        written += export_posreal_ratings(bundles, out)
        print(f"\nwrote {len(written)} files to {args.export}/")
        for p in written:
            print("  ", p)


if __name__ == "__main__":
    main()
