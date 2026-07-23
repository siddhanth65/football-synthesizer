"""Season-scale playing-style factor profile from public FBref event counts (Fernandez-Navarro 2016).

Phase-0 of the style-analysis v2 architecture (results/STYLE_RESEARCH_METHODS.md Axis 2 #4): the OT
territorial embedding gives no *named* style axes, and factor loadings are unstable at our n=6 match
scale. Fernandez-Navarro et al. (*Attacking and defensive styles of play*, J Sports Sci 2016) show that
season-scale event counts factor into interpretable style axes -- direct-vs-possession, pressing height,
width. We derive those axes at season scale from **public FBref 2024-25 squad aggregates** (all 20 PL
teams, so the factor structure is estimated on 20 samples, not 6), then read off coordinates for the seven
teams in our corpus (Manchester Utd + the six opponents).

Data: FBref squad tables cached under ``data/fbref_style_cache`` (gitignored, 3.2 MB), fetched via
soccerdata's rate-limited session. We use the fully-populated **/stats/ page** squad tables -- standard,
shooting, misc -- in BOTH ``_for`` (a team's own play) and ``_against`` (what it concedes) form.

**Data limitation (honest).** The richer FBref squad *passing*, *possession* and *defense* pages -- which
carry long-ball share (directness) and tackle-by-third (the clean season-scale *pressing-height* signal)
-- were **blocked by FBref anti-scraping** (connection refused after the /stats/ page fetched; the same
403/block ``tools/oracle.py`` documents). We did not hammer them. So this season profile spans
possession, attacking directness, width and defensive engagement, but **not pressing height at season
scale**: pressing/block height comes instead from our own tracking (``fingerprint.block_height``, Task 1).
Interception/tackle/foul rates here are labelled *defensive engagement*, not *height*.

Method: build interpretable per-90 / share features (below), z-score across the 20 teams, PCA (sklearn).
Each principal component is *named* by its loadings -- e.g. a component that loads possession% high and
shots-conceded low is a possession-control axis. We do NOT force the FN factor labels onto components; we
report the loadings and name only what the data supports.

Honesty: FBref counts are league-season totals, a different population than our six broadcast matches --
so this is a *season prior*, cross-checked against our per-match validated pass metrics where they exist,
not a restatement of them.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment

FBREF_CACHE = Path("data/fbref_style_cache")
_PAGE = "teams_ENG-Premier League_2425_{page}.html"

# FBref Squad spelling -> our registry team name (only the corpus seven need mapping).
CORPUS = {
    "Manchester Utd": "Man Utd", "Brighton": "Brighton", "Liverpool": "Liverpool",
    "Fulham": "Fulham", "Crystal Palace": "Crystal Palace", "Tottenham": "Tottenham",
    "Southampton": "Southampton",
}


def _read(page: str, table_id: str) -> pd.DataFrame:
    """Parse one FBref table (by id, incl. comment-embedded); flat disambiguated columns.

    The 2-level FBref header is flattened to ``"group/stat"`` when the group is a real span (e.g.
    ``"Long/Att"`` vs ``"Total/Att"``) and to ``"stat"`` when the top level is an ``Unnamed`` placeholder.
    """
    html = (FBREF_CACHE / _PAGE.format(page=page)).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("table", id=table_id)
    if node is None:
        for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
            if table_id in c:
                node = BeautifulSoup(c, "lxml").find("table", id=table_id)
                break
    if node is None:
        raise LookupError(f"table {table_id!r} not found in {page}")
    df = pd.read_html(StringIO(str(node)), header=[0, 1])[0]
    df.columns = [stat if str(grp).startswith("Unnamed") else f"{grp}/{stat}" for grp, stat in df.columns]
    return df


def _squad_table(page: str, table_id: str) -> pd.DataFrame:
    """A populated FBref *squad* table (standard/shooting/misc/defense), index=Squad."""
    df = _read(page, table_id)
    df = df[df["Squad"].astype(str).str.strip().ne("") & df["Squad"].notna()]
    # FBref "_against" tables index each row as "vs Manchester Utd" -- strip so it joins the "_for" tables.
    df["Squad"] = df["Squad"].astype(str).str.replace(r"^vs\s+", "", regex=True)
    return df.set_index("Squad")


def _num(s: pd.Series) -> pd.Series:
    """Comma-safe numeric coercion (FBref renders 20,123 for large counts)."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_features() -> pd.DataFrame:
    """Per-team season style features for all 20 PL teams (index = FBref Squad name).

    Features (interpretable, per-90 or share, from the populated /stats/ for+against squad tables):
        Possession/control: ``poss_pct``, ``shots_against_p90`` (fewer conceded = more control).
        Attacking directness/volume: ``shots_p90``, ``sot_pct``, ``goals_per_shot``, ``offsides_p90``
            (a high line / running in behind pushes offsides up).
        Width: ``crosses_p90``, ``crosses_against_p90`` (wide play conceded).
        Defensive engagement (NOT height -- see module docstring): ``interceptions_p90``,
            ``tackles_won_p90``, ``fouls_p90``.
    """
    std = _squad_table("stats", "stats_squads_standard_for")
    sht = _squad_table("stats", "stats_squads_shooting_for")
    misc = _squad_table("stats", "stats_squads_misc_for")
    sht_a = _squad_table("stats", "stats_squads_shooting_against")
    misc_a = _squad_table("stats", "stats_squads_misc_against")
    n90 = _num(std["Playing Time/90s"])

    f = pd.DataFrame(index=std.index)
    f["poss_pct"] = _num(std["Poss"])
    f["shots_p90"] = _num(sht["Standard/Sh"]) / n90
    f["sot_pct"] = _num(sht["Standard/SoT%"])
    f["goals_per_shot"] = _num(sht["Standard/G/Sh"])
    f["offsides_p90"] = _num(misc["Performance/Off"]) / n90
    f["crosses_p90"] = _num(misc["Performance/Crs"]) / n90
    f["interceptions_p90"] = _num(misc["Performance/Int"]) / n90
    f["tackles_won_p90"] = _num(misc["Performance/TklW"]) / n90
    f["fouls_p90"] = _num(misc["Performance/Fls"]) / n90
    f["shots_against_p90"] = _num(sht_a["Standard/Sh"]) / n90
    f["crosses_against_p90"] = _num(misc_a["Performance/Crs"]) / n90
    return f


def style_pca(features: pd.DataFrame, n_components: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Z-score the features across teams and run PCA.

    Returns:
        coords: per-team component scores (index = team, columns ``PC1..PCk``).
        loadings: feature loadings per component (columns ``PC1..PCk``), for naming the axes.
        explained: explained-variance ratio per component.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    x = StandardScaler().fit_transform(features.to_numpy(float))
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(x)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    coords = pd.DataFrame(scores, index=features.index, columns=cols)
    loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=cols)
    return coords, loadings, pca.explained_variance_ratio_


def main() -> None:
    pd.set_option("display.width", 200)
    f = load_features()
    coords, loadings, ev = style_pca(f)
    print("Explained variance:", [round(float(e), 3) for e in ev])
    print("\nLoadings (feature -> component):")
    print(loadings.round(2).to_string())
    print("\nCorpus team style coordinates (z-scored PCA, all-20-PL fit):")
    sub = coords.loc[[s for s in CORPUS if s in coords.index]].rename(index=CORPUS)
    print(sub.round(2).to_string())
    print("\nCorpus raw features:")
    print(f.loc[[s for s in CORPUS if s in f.index]].rename(index=CORPUS).round(2).to_string())


if __name__ == "__main__":
    main()
