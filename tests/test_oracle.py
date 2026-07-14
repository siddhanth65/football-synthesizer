"""Tests for tools.oracle: parsing/normalization, cache-first behaviour, cross-validation.

All tests use a stubbed fetcher (no network) or synthetic on-disk fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools import oracle

# --------------------------------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------------------------------

# Sofascore startTimestamp for 2024-08-24 (UTC), verified against the real API.
_TS_20240824 = 1724499000


def _synthetic_team_stats() -> pd.DataFrame:
    """Minimal ScraperFC-style team-stats frame (home=Brighton, away=Man Utd)."""
    rows = [
        ("Ball possession", "ballPossession", 48.0, 52.0),
        ("Passes", "passes", 477.0, 511.0),
        ("Accurate passes", "accuratePasses", 407.0, 446.0),
        ("Total shots", "totalShotsOnGoal", 14.0, 11.0),
        ("Shots on target", "shotsOnGoal", 5.0, 4.0),
        ("Expected goals", "expectedGoals", 2.09, 1.43),
        # noise rows in a non-ALL period must be ignored
        ("Ball possession", "ballPossession", 50.0, 50.0),
    ]
    data = []
    for i, (name, key, hv, av) in enumerate(rows):
        data.append(
            {
                "name": name,
                "key": key,
                "homeValue": hv,
                "awayValue": av,
                "period": "1ST" if i == len(rows) - 1 else "ALL",
                "group": "Match overview",
            }
        )
    return pd.DataFrame(data)


class _StubFetcher:
    """Records call counts; returns canned data without touching the network."""

    def __init__(self) -> None:
        self.n_dicts = 0
        self.n_stats = 0

    def match_dicts(self, year: str, league: str) -> list[dict]:
        self.n_dicts += 1
        return [
            {
                "id": 12436888,
                "homeTeam": {"name": "Brighton & Hove Albion"},
                "awayTeam": {"name": "Manchester United"},
                "startTimestamp": _TS_20240824,
            },
            {
                "id": 999,
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Wolves"},
                "startTimestamp": _TS_20240824,
            },
        ]

    def team_match_stats(self, match_id: int | str) -> pd.DataFrame:
        self.n_stats += 1
        return _synthetic_team_stats()


# --------------------------------------------------------------------------------------------------
# Name matching
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "full", "expected"),
    [
        ("Man Utd", "Manchester United", True),
        ("Brighton", "Brighton & Hove Albion", True),
        ("Manchester Utd", "Manchester United", True),
        ("Man Utd", "Manchester City", True),  # collision (disambiguated by date+both sides)
        ("Brighton", "Manchester United", False),
        ("Arsenal", "Aston Villa", False),
    ],
)
def test_name_match(query: str, full: str, expected: bool) -> None:
    assert oracle._name_match(query, full) is expected


def test_ts_to_date() -> None:
    assert oracle._ts_to_date(_TS_20240824) == "2024-08-24"
    assert oracle._ts_to_date(None) is None


# --------------------------------------------------------------------------------------------------
# Parsing / normalization
# --------------------------------------------------------------------------------------------------


def test_parse_sofascore_team_stats() -> None:
    parsed = oracle.parse_sofascore_team_stats(_synthetic_team_stats())
    # possession/xg are floats; counts are ints
    assert parsed["home"]["possession_pct"] == 48.0
    assert parsed["away"]["possession_pct"] == 52.0
    assert parsed["home"]["passes_cmp"] == 407
    assert parsed["away"]["passes_att"] == 511
    assert parsed["home"]["shots"] == 14
    assert parsed["away"]["shots_on_target"] == 4
    assert parsed["home"]["xg"] == 2.09
    assert isinstance(parsed["home"]["passes_cmp"], int)
    assert isinstance(parsed["home"]["xg"], float)


def test_parse_missing_metric_is_none() -> None:
    df = _synthetic_team_stats()
    df = df[df["key"] != "expectedGoals"]
    parsed = oracle.parse_sofascore_team_stats(df)
    assert parsed["home"]["xg"] is None
    assert parsed["home"]["shots"] == 14


# --------------------------------------------------------------------------------------------------
# Cache-first behaviour
# --------------------------------------------------------------------------------------------------


def test_team_stats_df_cache_first(tmp_path: Path) -> None:
    stub = _StubFetcher()
    df1 = oracle.team_stats_df(12436888, stub, cache_dir=tmp_path)
    assert stub.n_stats == 1
    assert (tmp_path / "team_stats_12436888.parquet").exists()
    df2 = oracle.team_stats_df(12436888, stub, cache_dir=tmp_path)
    assert stub.n_stats == 1  # served from cache, fetcher NOT called again
    pd.testing.assert_frame_equal(df1, df2)


def test_season_match_dicts_cache_first(tmp_path: Path) -> None:
    stub = _StubFetcher()
    oracle.season_match_dicts("24/25", "England Premier League", stub, cache_dir=tmp_path)
    oracle.season_match_dicts("24/25", "England Premier League", stub, cache_dir=tmp_path)
    assert stub.n_dicts == 1  # second call is a cache hit


def test_resolve_fixture(tmp_path: Path) -> None:
    stub = _StubFetcher()
    query = {
        "home": "Brighton",
        "away": "Man Utd",
        "date": "2024-08-24",
        "league": "England Premier League",
        "year": "24/25",
    }
    d = oracle.resolve_fixture(query, stub, cache_dir=tmp_path)
    assert d["id"] == 12436888


def test_resolve_fixture_not_found(tmp_path: Path) -> None:
    stub = _StubFetcher()
    query = {
        "home": "Liverpool",
        "away": "Chelsea",
        "date": "2024-08-24",
        "league": "England Premier League",
        "year": "24/25",
    }
    with pytest.raises(LookupError):
        oracle.resolve_fixture(query, stub, cache_dir=tmp_path)


def test_get_match_aggregates_full(tmp_path: Path) -> None:
    stub = _StubFetcher()
    query = {
        "home": "Brighton",
        "away": "Man Utd",
        "date": "2024-08-24",
        "league": "England Premier League",
        "year": "24/25",
    }
    agg = oracle.get_match_aggregates(query, source="sofascore", fetcher=stub, cache_dir=tmp_path)
    assert agg["match_id"] == 12436888
    assert agg["source"] == "sofascore"
    home = next(t for t in agg["teams"] if t["side"] == "home")
    away = next(t for t in agg["teams"] if t["side"] == "away")
    assert home["team"] == "Brighton & Hove Albion"
    assert home["passes_cmp"] == 407
    assert away["shots"] == 11
    assert agg["provenance"]["source"] == "sofascore"
    assert "team_stats_12436888.parquet" in agg["provenance"]["raw_cache_path"]


def test_get_match_aggregates_bad_source() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        oracle.get_match_aggregates(1, source="opta")


# --------------------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------------------


def _agg(source: str, poss_home: float, shots_home: int, passes_cmp_home: int | None) -> dict:
    return {
        "source": source,
        "teams": [
            {
                "team": f"{source}-home",
                "side": "home",
                "possession_pct": poss_home,
                "passes_cmp": passes_cmp_home,
                "passes_att": None,
                "shots": shots_home,
                "shots_on_target": None,
                "xg": None,
            },
            {
                "team": f"{source}-away",
                "side": "away",
                "possession_pct": 100 - poss_home,
                "passes_cmp": None,
                "passes_att": None,
                "shots": 10,
                "shots_on_target": None,
                "xg": None,
            },
        ],
    }


def test_cross_validate_agreement() -> None:
    a = _agg("sofa", 48.0, 14, 407)
    b = _agg("fbref", 48.0, 14, None)
    cv = oracle.cross_validate(a, b)
    assert cv["flags"] == []
    poss = next(r for r in cv["rows"] if r["side"] == "home" and r["metric"] == "possession_pct")
    assert poss["diff"] == 0.0
    # a metric missing in secondary reports diff=None, never flagged
    pc = next(r for r in cv["rows"] if r["side"] == "home" and r["metric"] == "passes_cmp")
    assert pc["diff"] is None


def test_cross_validate_possession_flag() -> None:
    a = _agg("sofa", 55.0, 14, 407)
    b = _agg("fbref", 48.0, 14, None)  # 7 pp gap > 2 pp tol
    cv = oracle.cross_validate(a, b)
    assert any("possession" in f for f in cv["flags"])


def test_cross_validate_passes_flag() -> None:
    a = _agg("sofa", 48.0, 14, 500)
    b = _agg("fbref", 48.0, 14, 407)  # ~23% gap > 5% tol
    cv = oracle.cross_validate(a, b)
    assert any("passes_cmp" in f for f in cv["flags"])


# --------------------------------------------------------------------------------------------------
# FBref fallback parse (synthetic on-disk HTML)
# --------------------------------------------------------------------------------------------------


def _write_fbref_cache(cache_dir: Path) -> None:
    sched = pd.DataFrame(
        {
            "Date": ["2024-08-24", "2024-09-01"],
            "Opponent": ["Manchester Utd", "Everton"],
            "Poss": [48.0, 60.0],
            "GF": [2, 1],
            "GA": [1, 0],
        }
    )
    shoot = pd.DataFrame(
        {
            "Date": ["2024-08-24", "2024-09-01"],
            "Opponent": ["Manchester Utd", "Everton"],
            "Sh": [14, 9],
            "SoT": [5, 3],
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "matchlogs_Brighton_2425_schedule.html").write_text(
        sched.to_html(index=False), encoding="utf-8"
    )
    (cache_dir / "matchlogs_Brighton_2425_shooting.html").write_text(
        shoot.to_html(index=False), encoding="utf-8"
    )


def test_fbref_team_aggregates(tmp_path: Path) -> None:
    _write_fbref_cache(tmp_path)
    agg = oracle.fbref_team_aggregates("Brighton", "2024-08-24", cache_dir=tmp_path)
    assert agg["possession_pct"] == 48.0
    assert agg["shots"] == 14
    assert agg["shots_on_target"] == 5
    # passing/xg were never cached -> None
    assert agg["passes_cmp"] is None
    assert agg["xg"] is None


def test_get_match_aggregates_fbref(tmp_path: Path) -> None:
    _write_fbref_cache(tmp_path)
    query = {"home": "Brighton", "away": "Brighton", "date": "2024-08-24"}
    agg = oracle.get_match_aggregates(query, source="fbref", cache_dir=tmp_path)
    assert agg["source"] == "fbref"
    assert agg["teams"][0]["shots"] == 14
