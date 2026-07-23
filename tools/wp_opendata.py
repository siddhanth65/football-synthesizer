"""Open-data acquisition for the win-probability base subset: StatsBomb timelines + clubelo Elo.

Two free sources, both cache-first (like ``tools/oracle.py``):

* **StatsBomb open-data** (github.com/statsbomb/open-data): the goal / card *timing* the WP model
  needs, which league match tables never carry. We stream one season's event JSONs, extract only the
  goal and card minutes (per team) plus the final score + date, validate the extracted goals against
  the published final score, and cache a tiny compact timeline JSON per match. Raw events are never
  persisted (keeps the slice < a few MB, well under the 1 GB budget).
* **clubelo** (api.clubelo.com): the team-strength prior. We fetch the all-clubs Elo snapshot valid on
  a given date and fuzzy-match team names against it (clubelo's ``From..To`` ranges mean any date
  resolves to the Elo in force then), caching one CSV per date. Fuzzy matching sidesteps a
  StatsBomb <-> clubelo name table ("Manchester United" vs "Man United").

Network + CPU only. Lives in ``tools/`` (data-acquisition adapter, not a pipeline primitive).
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

from tools.oracle import _slug

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ELO_BASE = "http://api.clubelo.com"
SB_CACHE = Path("data/wp_opendata")            # gitignored (see .gitignore)
ELO_CACHE = Path("outputs/oracle/elo")         # gitignored (outputs/)
_RED_CARDS = {"Red Card", "Second Yellow"}
_POLITE_S = 0.3


_TOKEN_ALIAS = {"utd": "united"}   # normalise registry "Man Utd" -> "Man United"


def _tokens(name: str) -> list[str]:
    """Significant (>2-char) lowercased tokens of a club name, with abbreviations normalised."""
    raw = name.lower().replace("&", " ").replace(".", " ").split()
    return [_TOKEN_ALIAS.get(t, t) for t in raw if len(t) > 2]


def _prefix_match(a: str, b: str) -> bool:
    """True if two tokens are equal or one is a >=3-char prefix of the other ("man" ~ "manchester")."""
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def _match_club(team: str, clubs: list[str]) -> str | None:
    """Best clubelo club name for a query team, or ``None``.

    Football names collide on generic tokens ("City", "United", "Man"), so a plain token-overlap
    matcher hands Leicester City / Man City / Man United the same club. We instead prefer the clubelo
    club whose OWN tokens are all covered by the query (full coverage), breaking ties by most query
    tokens matched then most specific (most tokens) then shortest name. Falls back to best raw overlap.
    """
    qt = _tokens(team)
    scored = []
    for club in clubs:
        ct = _tokens(club)
        if not ct:
            continue
        covered = all(any(_prefix_match(c, q) for q in qt) for c in ct)
        overlap = sum(any(_prefix_match(q, c) for c in ct) for q in qt)
        scored.append((covered, overlap, len(ct), -len(club), club))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    return best[4] if (best[0] or best[1] > 0) else None


def _get_json(url: str, session: requests.Session) -> list | dict:
    r = session.get(url, timeout=90)
    r.raise_for_status()
    return r.json()


def _extract_timeline(events: list[dict], home: str, away: str) -> dict:
    """Pull goal / card minutes (side = +1 home / -1 away) from a StatsBomb events list."""
    goals: list[tuple[int, int]] = []
    reds: list[tuple[int, int]] = []
    yellows: list[tuple[int, int]] = []
    for e in events:
        team = e.get("team", {}).get("name")
        side = 1 if team == home else (-1 if team == away else 0)
        if side == 0:
            continue
        mn = int(e.get("minute", 0))
        typ = e["type"]["name"]
        if typ == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            goals.append((mn, side))
        elif typ == "Own Goal For":              # credited to the team that benefits
            goals.append((mn, side))
        card = (e.get("bad_behaviour", {}).get("card", {}).get("name")
                or e.get("foul_committed", {}).get("card", {}).get("name"))
        if card in _RED_CARDS:
            reds.append((mn, side))
        elif card == "Yellow Card":
            yellows.append((mn, side))
    return {"goals": goals, "reds": reds, "yellows": yellows}


def statsbomb_timelines(comp_id: int, season_id: int, *, cache_dir: Path = SB_CACHE,
                        session: requests.Session | None = None, limit: int | None = None
                        ) -> list[dict]:
    """Compact goal/card timelines for every match in a StatsBomb competition-season (cache-first).

    Each cached timeline JSON is ``{match_id, date, home, away, home_score, away_score, goals, reds,
    yellows, goals_ok}`` where the event lists are ``(minute, side)`` and ``goals_ok`` flags whether
    the extracted goal tally matched the published final score (a data-quality gate; mismatches keep
    the published score and set the flag false so the caller can drop them).

    Args:
        comp_id: StatsBomb competition id (e.g. 2 = Premier League).
        season_id: StatsBomb season id (e.g. 27 = 2015/2016).
        cache_dir: root cache dir; one JSON per match under ``<cache_dir>/<comp>_<season>/``.
        session: optional shared requests session.
        limit: cap the number of matches (debug / smoke).

    Returns:
        List of timeline dicts (one per match).
    """
    sess = session or requests.Session()
    out_dir = cache_dir / f"{comp_id}_{season_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = _get_json(f"{SB_BASE}/matches/{comp_id}/{season_id}.json", sess)
    if limit is not None:
        matches = matches[:limit]
    timelines: list[dict] = []
    for i, m in enumerate(matches):
        mid = m["match_id"]
        path = out_dir / f"{mid}.json"
        if path.exists():
            timelines.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        events = _get_json(f"{SB_BASE}/events/{mid}.json", sess)
        tl = _extract_timeline(events, home, away)
        gh = sum(1 for _, s in tl["goals"] if s == 1)
        ga = sum(1 for _, s in tl["goals"] if s == -1)
        rec = {"match_id": mid, "date": m["match_date"], "home": home, "away": away,
               "home_score": hs, "away_score": as_, "goals": tl["goals"], "reds": tl["reds"],
               "yellows": tl["yellows"], "goals_ok": bool(gh == hs and ga == as_)}
        path.write_text(json.dumps(rec), encoding="utf-8")
        timelines.append(rec)
        time.sleep(_POLITE_S)
        if (i + 1) % 25 == 0:
            print(f"  statsbomb {comp_id}/{season_id}: {i + 1}/{len(matches)} cached")
    return timelines


def elo_snapshot(date: str, *, cache_dir: Path = ELO_CACHE,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """All-clubs clubelo Elo valid on ``date`` (``YYYY-MM-DD``), cache-first (one CSV per date)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"snapshot_{date}.csv"
    if path.exists():
        return pd.read_csv(path)
    sess = session or requests.Session()
    r = sess.get(f"{ELO_BASE}/{date}", timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(path, index=False)
    time.sleep(_POLITE_S)
    return df


def elo_at(team: str, date: str, *, cache_dir: Path = ELO_CACHE,
           session: requests.Session | None = None) -> float | None:
    """clubelo Elo of a team on a date, fuzzy-matched by name (``None`` if unmatched).

    Args:
        team: team name in any common spelling (StatsBomb / registry / Sofascore).
        date: ``YYYY-MM-DD``.
        cache_dir: Elo snapshot cache dir.
        session: optional shared session.

    Returns:
        Elo rating (float) or ``None`` when no clubelo club name matches.
    """
    snap = elo_snapshot(date, cache_dir=cache_dir, session=session)
    club = _match_club(team, [str(c) for c in snap["Club"]])
    if club is None:
        return None
    return float(snap.loc[snap["Club"] == club, "Elo"].iloc[0])


def _demo() -> None:
    """Smoke test (network): one PL 15/16 match timeline + one Elo lookup. Skips offline."""
    try:
        tl = statsbomb_timelines(2, 27, limit=1)[0]
        assert tl["goals_ok"], tl
        e = elo_at("Manchester United", "2015-08-08")
        assert e and 1500 < e < 2200, e
        print(f"wp_opendata self-check OK (sample {tl['home']} {tl['home_score']}-"
              f"{tl['away_score']} {tl['away']}, ManU Elo {e:.0f}) slug={_slug(tl['home'])}")
    except (requests.RequestException, OSError) as exc:  # offline / rate-limited: don't fail hard
        print(f"wp_opendata self-check SKIPPED (network: {exc})")


if __name__ == "__main__":
    _demo()
