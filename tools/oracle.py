"""PL match-oracle: per-fixture TEAM aggregates from Sofascore (primary) + FBref (fallback).

Provides ground-truth team aggregates (possession, passes, shots, xG) for validating CV-derived
numbers, replacing the fragile/403-blocked direct-FBref path. Sofascore is the primary source via
ScraperFC's headless-browser API; the already-cached ``soccerdata`` FBref pages serve as an offline
cross-check (no live FBref fetch).

Design:
  * **Cache-first, always.** Every raw response is written under ``outputs/oracle/`` and re-read on
    the next call; the network fetcher (and its 5 s rate limit) only runs on a cache miss.
  * **Injectable fetcher.** :func:`get_match_aggregates` takes an optional ``fetcher`` so the pure
    parsing / cache logic is testable without the network.
  * **Provenance on every result** (source, fetch date, raw cache path, match id).

This lives in ``tools/`` rather than ``core/`` on purpose: it pulls in ScraperFC + botasaurus (a
headless browser) and network I/O, which ``core/`` (registry + pitch constants, imported everywhere)
must stay free of. It is a data-acquisition/validation adapter, not a pipeline primitive.

Lives on CPU + network only. Verified empirically against Brighton 2-1 Manchester Utd (2024-08-24,
Sofascore match id 12436888).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

ORACLE_DIR = Path("outputs/oracle")
SOFA_DIR = ORACLE_DIR / "sofascore"
# read-only soccerdata FBref HTML cache (cross-check only; never a live fetch)
FBREF_CACHE = Path.home() / "soccerdata" / "data" / "FBref"
RATE_LIMIT_S = 5.0

# Sofascore statisticsItems `key` -> our logical metric. Counts are cast to int, rates/xG to float.
_SOFA_KEYS: dict[str, str] = {
    "possession_pct": "ballPossession",
    "passes_att": "passes",
    "passes_cmp": "accuratePasses",
    "shots": "totalShotsOnGoal",
    "shots_on_target": "shotsOnGoal",
    "xg": "expectedGoals",
}
_FLOAT_METRICS = {"possession_pct", "xg"}
_METRIC_ORDER = ["possession_pct", "passes_cmp", "passes_att", "shots", "shots_on_target", "xg"]


# --------------------------------------------------------------------------------------------------
# Fetcher abstraction (real one wraps ScraperFC; tests inject a stub)
# --------------------------------------------------------------------------------------------------


class SofascoreFetcher(Protocol):
    """Minimal Sofascore data source contract used by the oracle."""

    def match_dicts(self, year: str, league: str) -> list[dict[str, Any]]:
        """Return the raw list of match dicts for a league season."""
        ...

    def team_match_stats(self, match_id: int | str) -> pd.DataFrame:
        """Return ScraperFC's team-stats DataFrame for a match id."""
        ...


class ScraperFCFetcher:
    """Default fetcher: ScraperFC's ``Sofascore`` module with a fixed rate limit after each call."""

    def __init__(self, rate_limit_s: float = RATE_LIMIT_S) -> None:
        """Store the rate limit; the ScraperFC client is created lazily on first use."""
        self.rate_limit_s = rate_limit_s
        self._client: Any = None

    def _sofa(self) -> Any:
        if self._client is None:
            from ScraperFC import Sofascore  # noqa: PLC0415

            self._client = Sofascore()
        return self._client

    def match_dicts(self, year: str, league: str) -> list[dict[str, Any]]:
        """Fetch season match dicts, then sleep ``rate_limit_s``."""
        out = self._sofa().get_match_dicts(year, league)
        time.sleep(self.rate_limit_s)
        return list(out)

    def team_match_stats(self, match_id: int | str) -> pd.DataFrame:
        """Fetch team match stats, then sleep ``rate_limit_s``."""
        out = self._sofa().scrape_team_match_stats(match_id)
        time.sleep(self.rate_limit_s)
        return out


# --------------------------------------------------------------------------------------------------
# Name / slug helpers
# --------------------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    """ASCII-safe filename slug."""
    keep = "".join(c if c.isalnum() else "_" for c in text)
    return "_".join(p for p in keep.split("_") if p)


def _name_match(query: str, full: str) -> bool:
    """Loose team-name match (token containment, case-insensitive).

    Handles ``"Man Utd"`` vs ``"Manchester United"`` and ``"Brighton"`` vs
    ``"Brighton & Hove Albion"`` without a hardcoded alias table.
    """
    q = query.lower()
    f = full.lower()
    if q in f or f in q:
        return True
    q_tokens = [t for t in q.replace("&", " ").split() if len(t) > 2]
    f_tokens = [t for t in f.replace("&", " ").split() if len(t) > 2]
    for qt in q_tokens:
        for ft in f_tokens:
            # token equality or one a >=3-char prefix of the other ("man" ~ "manchester")
            if qt == ft or (min(len(qt), len(ft)) >= 3 and (qt.startswith(ft) or ft.startswith(qt))):
                return True
    return False


def _ts_to_date(ts: int | None) -> str | None:
    """UTC ``YYYY-MM-DD`` for a Sofascore startTimestamp."""
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------------------------------
# Sofascore: cache-first fetch + pure parse
# --------------------------------------------------------------------------------------------------


def season_match_dicts(
    year: str, league: str, fetcher: SofascoreFetcher, cache_dir: Path = SOFA_DIR
) -> list[dict[str, Any]]:
    """Season match dicts, cache-first (JSON under ``cache_dir``)."""
    path = cache_dir / f"match_dicts_{_slug(f'{league}_{year}')}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    dicts = fetcher.match_dicts(year, league)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dicts), encoding="utf-8")
    return dicts


def resolve_fixture(
    query: dict[str, str], fetcher: SofascoreFetcher, cache_dir: Path = SOFA_DIR
) -> dict[str, Any]:
    """Find the match dict for a ``{home, away, date, league, year}`` query.

    Args:
        query: Must contain ``home``, ``away``, ``date`` (``YYYY-MM-DD``), ``league``, ``year``.
        fetcher: Sofascore data source.
        cache_dir: Where season dicts are cached.

    Returns:
        The matching Sofascore match dict.

    Raises:
        LookupError: If no fixture matches home/away/date.
    """
    dicts = season_match_dicts(query["year"], query["league"], fetcher, cache_dir)
    for d in dicts:
        home = d.get("homeTeam", {}).get("name", "")
        away = d.get("awayTeam", {}).get("name", "")
        if (
            _ts_to_date(d.get("startTimestamp")) == query["date"]
            and _name_match(query["home"], home)
            and _name_match(query["away"], away)
        ):
            return d
    raise LookupError(
        f"no Sofascore fixture for {query['home']} vs {query['away']} on {query['date']}"
    )


def team_stats_df(
    match_id: int | str, fetcher: SofascoreFetcher, cache_dir: Path = SOFA_DIR
) -> pd.DataFrame:
    """Raw ScraperFC team-stats DataFrame for a match, cache-first (parquet)."""
    path = cache_dir / f"team_stats_{match_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetcher.team_match_stats(match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def parse_sofascore_team_stats(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Extract per-side aggregates from a ScraperFC team-stats DataFrame.

    Args:
        df: Output of ``scrape_team_match_stats`` (rows per stat, ``period``/``group`` columns).

    Returns:
        ``{"home": {metric: value}, "away": {metric: value}}`` for the whole match (``period ==
        "ALL"``). Missing stats map to ``None``.
    """
    all_rows = df[df["period"] == "ALL"] if "period" in df.columns else df
    out: dict[str, dict[str, float | int | None]] = {"home": {}, "away": {}}
    for metric, key in _SOFA_KEYS.items():
        rows = all_rows[all_rows["key"] == key]
        for side, col in (("home", "homeValue"), ("away", "awayValue")):
            if rows.empty or pd.isna(rows.iloc[0][col]):
                out[side][metric] = None
            elif metric in _FLOAT_METRICS:
                out[side][metric] = round(float(rows.iloc[0][col]), 2)
            else:
                out[side][metric] = int(round(float(rows.iloc[0][col])))
    return out


# --------------------------------------------------------------------------------------------------
# FBref fallback: parse the already-cached soccerdata HTML (no live fetch)
# --------------------------------------------------------------------------------------------------


def _fbref_matchlog(team: str, stat: str, season_code: str, cache_dir: Path) -> pd.DataFrame:
    """Load and flatten one cached FBref matchlog table (largest table on the page)."""
    path = cache_dir / f"matchlogs_{team}_{season_code}_{stat}.html"
    if not path.exists():
        raise FileNotFoundError(f"FBref cache miss: {path}")
    tables = pd.read_html(path)
    df = max(tables, key=lambda t: t.shape[0])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            b if (b and not str(b).startswith("Unnamed")) else a for a, b in df.columns
        ]
    return df


def fbref_team_aggregates(
    team: str, date: str, season_code: str = "2425", cache_dir: Path = FBREF_CACHE
) -> dict[str, float | int | None]:
    """Per-team aggregates for a fixture from the cached FBref matchlogs.

    Only possession and shots are cached (schedule + shooting); the passing matchlog was never
    cached, so ``passes_cmp``/``passes_att``/``xg`` come back ``None`` (documented limitation of the
    offline cross-check).

    Args:
        team: FBref team spelling (e.g. ``"Brighton"``, ``"Manchester Utd"``).
        date: ``YYYY-MM-DD``.
        season_code: FBref cache season suffix (``"2425"`` for 24/25).
        cache_dir: FBref HTML cache dir.

    Returns:
        Metric dict with the standard keys; unavailable metrics are ``None``.
    """
    sch = _fbref_matchlog(team, "schedule", season_code, cache_dir)
    srow = sch[sch["Date"].astype(str).str.startswith(date)]
    if srow.empty:
        raise LookupError(f"no FBref schedule row for {team} on {date}")
    srow = srow.iloc[0]
    shots = sot = None
    try:
        sh = _fbref_matchlog(team, "shooting", season_code, cache_dir)
        shrow = sh[sh["Date"].astype(str).str.startswith(date)]
        if not shrow.empty:
            shots = int(round(float(shrow.iloc[0]["Sh"])))
            sot = int(round(float(shrow.iloc[0]["SoT"])))
    except (FileNotFoundError, KeyError):
        pass
    xg = float(srow["xG"]) if "xG" in sch.columns and pd.notna(srow.get("xG")) else None
    return {
        "possession_pct": round(float(srow["Poss"]), 2) if pd.notna(srow["Poss"]) else None,
        "passes_cmp": None,
        "passes_att": None,
        "shots": shots,
        "shots_on_target": sot,
        "xg": xg,
    }


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------


def get_match_aggregates(
    match: int | str | dict[str, str],
    source: str = "sofascore",
    *,
    fetcher: SofascoreFetcher | None = None,
    cache_dir: Path | None = None,
    season_code: str = "2425",
) -> dict[str, Any]:
    """Per-team aggregates for one fixture, with provenance.

    Args:
        match: Sofascore match id (``int``/digit-``str``), or a query dict. For Sofascore the query
            needs ``{home, away, date, league, year}``; for FBref it needs ``{home, away, date}``
            using FBref team spellings.
        source: ``"sofascore"`` (primary) or ``"fbref"`` (offline cross-check).
        fetcher: Sofascore data source; defaults to :class:`ScraperFCFetcher`. Ignored for FBref.
        cache_dir: Override the raw-cache dir (Sofascore) or the FBref HTML cache dir.
        season_code: FBref cache season suffix (fallback only).

    Returns:
        Dict with ``source``, ``match_id``, ``fixture`` (home/away/date), ``teams`` (list of
        per-team aggregate dicts, each with ``team``/``side`` + the six metrics), and
        ``provenance`` (source, fetch_date, raw_cache_path, match_id).

    Raises:
        ValueError: If ``source`` is unknown.
        LookupError: If the fixture cannot be resolved.
    """
    fetch_date = dt.date.today().isoformat()
    if source == "sofascore":
        cdir = cache_dir or SOFA_DIR
        fx = fetcher or ScraperFCFetcher()
        if isinstance(match, dict):
            d = resolve_fixture(match, fx, cdir)
            match_id = d["id"]
            fixture = {
                "home": d["homeTeam"]["name"],
                "away": d["awayTeam"]["name"],
                "date": _ts_to_date(d.get("startTimestamp")),
            }
        else:
            match_id = int(match)
            fixture = {"home": "home", "away": "away", "date": None}
        df = team_stats_df(match_id, fx, cdir)
        parsed = parse_sofascore_team_stats(df)
        raw_path = cdir / f"team_stats_{match_id}.parquet"
        teams = [
            {"team": fixture["home"], "side": "home", **parsed["home"]},
            {"team": fixture["away"], "side": "away", **parsed["away"]},
        ]
        return {
            "source": "sofascore",
            "match_id": match_id,
            "fixture": fixture,
            "teams": teams,
            "provenance": {
                "source": "sofascore",
                "fetch_date": fetch_date,
                "raw_cache_path": str(raw_path),
                "match_id": match_id,
            },
        }
    if source == "fbref":
        cdir = cache_dir or FBREF_CACHE
        if not isinstance(match, dict):
            raise ValueError("FBref source needs a {home, away, date} query dict")
        home, away, date = match["home"], match["away"], match["date"]
        teams = []
        for team, side in ((home, "home"), (away, "away")):
            agg = fbref_team_aggregates(team, date, season_code, cdir)
            teams.append({"team": team, "side": side, **agg})
        return {
            "source": "fbref",
            "match_id": None,
            "fixture": {"home": home, "away": away, "date": date},
            "teams": teams,
            "provenance": {
                "source": "fbref",
                "fetch_date": fetch_date,
                "raw_cache_path": str(cdir),
                "match_id": None,
            },
        }
    raise ValueError(f"unknown source: {source!r}")


def cross_validate(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    poss_tol_pp: float = 2.0,
    pass_tol_pct: float = 5.0,
) -> dict[str, Any]:
    """Compare two aggregate dicts side-by-side and flag material disagreements.

    Teams are aligned by ``side`` (home/away) so differing name spellings across sources do not
    matter. A flag is raised when possession differs by more than ``poss_tol_pp`` percentage points
    or completed passes differ by more than ``pass_tol_pct`` percent. Metrics missing in either
    source are reported but never flagged.

    Args:
        primary: Aggregate dict (e.g. Sofascore).
        secondary: Aggregate dict to compare against (e.g. FBref).
        poss_tol_pp: Possession disagreement threshold (percentage points).
        pass_tol_pct: Completed-pass disagreement threshold (percent of primary).

    Returns:
        ``{"rows": [...per team/metric...], "flags": [...str...]}``.
    """
    prim = {t["side"]: t for t in primary["teams"]}
    sec = {t["side"]: t for t in secondary["teams"]}
    rows: list[dict[str, Any]] = []
    flags: list[str] = []
    for side in ("home", "away"):
        if side not in prim or side not in sec:
            continue
        pteam = prim[side]
        steam = sec[side]
        for metric in _METRIC_ORDER:
            pv = pteam.get(metric)
            sv = steam.get(metric)
            diff = None if (pv is None or sv is None) else round(pv - sv, 2)
            rows.append(
                {
                    "side": side,
                    "team_primary": pteam["team"],
                    "team_secondary": steam["team"],
                    "metric": metric,
                    "primary": pv,
                    "secondary": sv,
                    "diff": diff,
                }
            )
            if diff is None:
                continue
            if metric == "possession_pct" and abs(diff) > poss_tol_pp:
                flags.append(
                    f"{side} possession {pv} vs {sv} (delta {diff:+.1f} pp > {poss_tol_pp})"
                )
            if metric == "passes_cmp" and sv and abs(diff) / sv * 100.0 > pass_tol_pct:
                flags.append(
                    f"{side} passes_cmp {pv} vs {sv} (delta {abs(diff) / sv * 100:.1f}% > "
                    f"{pass_tol_pct}%)"
                )
    return {"rows": rows, "flags": flags}
