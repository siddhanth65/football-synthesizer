"""Parse a FIFA Post-Match Summary Report (PMSR) PDF into structured ground truth.

Extracts the pages our pipeline can validate against: **phases of play** (the % time each team spends in
each in/out-of-possession phase -- the C6 oracle for our phase classifier), **key statistics**
(possession, xG, line breaks, receptions, pressures), and the **header** (teams + score). Player lineups
are kept in a hand-curated roster (``data/france_roster.json``) because the lineup page interleaves the
two teams' columns; phases/stats are clean.

    python tools/parse_pmsr.py --pdf PMSR-M17-FRA-V-SEN.pdf --out outputs/pmsr/fra_sen.json
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.getLogger("pdfminer").setLevel(logging.ERROR)

PHASE_RE = re.compile(r"^(\d+)%\s+(.+?)\s+(\d+)%$")
# Pitch-graphic number tokens on line-height pages, e.g. "38m".
NUMBER_TOKEN_RE = re.compile(r"^(\d+)m$")
# Column order (left to right) for each line-height page kind.
_IN_POSSESSION_PHASES = ["build_up_low", "build_up_mid", "final_third"]
_DEFENSIVE_PHASES = ["high_block", "mid_block", "low_block"]
# Header-row words to ignore when clustering phase columns (extra labels, punctuation).
_HEADER_SKIP_WORDS = {"Phase", "/"}
# (key, regex with home + away groups) over the key-statistics page text.
STATS = [
    ("possession_pct", r"Total\s+([\d.]+)%.*?([\d.]+)%\s+Total"),
    ("goals", r"\n(\d+)\s+Goals\s+(\d+)"),
    ("xg", r"([\d.]+)\s+xG \(Expected Goals\)\s+([\d.]+)"),
    ("attempts", r"(\d+) \(\d+\)\s+Attempts at Goal.*? (\d+) \(\d+\)"),
    ("attempts_on_target", r"\d+ \((\d+)\)\s+Attempts at Goal.*? \d+ \((\d+)\)"),
    ("passes", r"(\d+) \(\d+\)\s+Total Passes.*? (\d+) \(\d+\)"),
    ("pass_completion_pct", r"(\d+) %\s+Pass Completion %\s+(\d+) %"),
    ("completed_line_breaks", r"(\d+)\s+Completed Line Breaks\s+(\d+)"),
    ("defensive_line_breaks", r"(\d+)\s+Defensive Line Breaks\s+(\d+)"),
    ("receptions_final_third", r"(\d+)\s+Receptions in the Final Third\s+(\d+)"),
    ("crosses", r"(\d+)\s+Crosses\s+(\d+)"),
    ("ball_progressions", r"(\d+)\s+Ball Progressions\s+(\d+)"),
    ("defensive_pressures", r"(\d+) \(\d+\)\s+Defensive Pressures.*? (\d+) \(\d+\)"),
    ("forced_turnovers", r"(\d+)\s+Forced Turnovers\s+(\d+)"),
    ("distance_km", r"([\d.]+) km\s+Total Distance Covered\s+([\d.]+) km"),
]


def _pages_text(pdf_path: str, pages=range(1, 4)) -> dict[int, str]:
    import pdfplumber  # noqa: PLC0415

    out = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i in pages:
            if i < len(pdf.pages):
                out[i] = pdf.pages[i].extract_text() or ""
    return out


def parse_header(text: str) -> dict:
    """Teams + score from the summary page (``A  s1  s2  B``)."""
    m = re.search(r"\n(\d+)\s+(\d+)\n([A-Za-z ]+?)\s+([A-Za-z ]+?)\n", text)
    if m:
        return {"home": m.group(3).strip(), "away": m.group(4).strip(),
                "score": [int(m.group(1)), int(m.group(2))]}
    return {}


def parse_phases(text: str) -> dict[str, list[int]]:
    """``phase -> [home%, away%]`` from the phases-of-play page."""
    phases = {}
    for line in text.splitlines():
        m = PHASE_RE.match(line.strip())
        if m:
            phases[m.group(2).strip().lower().replace(" ", "_")] = [int(m.group(1)), int(m.group(3))]
    return phases


def parse_stats(text: str) -> dict[str, list[float]]:
    """``stat -> [home, away]`` from the key-statistics page (best-effort per known label)."""
    stats = {}
    for key, rx in STATS:
        m = re.search(rx, text, re.DOTALL)
        if m:
            stats[key] = [float(m.group(1)), float(m.group(2))]
    return stats


def _group_rows(words: list[dict], tol: float = 2.0) -> list[list[dict]]:
    """Cluster ``page.extract_words()`` output into text rows by ``top`` (y) coordinate."""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["top"]):
        if rows and abs(w["top"] - rows[-1][0]["top"]) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w["x0"])
    return rows


def _cluster_columns(row: list[dict], gap: float = 80.0) -> list[list[dict]]:
    """Split a header row into left-to-right column word clusters by x-gap."""
    clusters: list[list[dict]] = []
    for w in row:
        if clusters and (w["x0"] - clusters[-1][-1]["x1"]) <= gap:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    return clusters


def _parse_line_height_page(page, phase_names: list[str]) -> tuple[str, dict[str, dict]] | None:
    """Decode one line-height pitch-graphic page.

    Locates the 3 phase-column headers (e.g. "Build Up Low"/"Build Up Mid"/"Final
    Third"), then assigns every "<N>m" number token on the page to its nearest
    column by x-position. The line height for a column is the number nearest the
    bottom of the page (largest ``top``) -- the deepest defensive line in that phase.

    Args:
        page: A ``pdfplumber`` page.
        phase_names: Column names in left-to-right order for this page kind.

    Returns:
        ``(team, {phase: {"line_height", "front_line", "team_length"}})``, or
        ``None`` if the page doesn't look like a line-height page.
    """
    words = page.extract_words()
    rows = _group_rows(words)
    if not rows:
        return None
    title_row = next((r for r in rows if any(w["text"] == "Length" for w in r)), None)
    if title_row is None:
        return None
    title_top = title_row[0]["top"]
    length_word = next(w for w in title_row if w["text"] == "Length")
    team = " ".join(w["text"] for w in title_row if w["x0"] > length_word["x0"]).strip()

    header_row = next((r for r in rows if r[0]["top"] > title_top + 5), None)
    if header_row is None:
        return None
    header_words = [w for w in header_row if w["text"] not in _HEADER_SKIP_WORDS]
    clusters = _cluster_columns(header_words)
    if len(clusters) != len(phase_names):
        logging.getLogger(__name__).warning(
            "line-height header on %r: expected %d columns, found %d",
            team, len(phase_names), len(clusters),
        )
    centroids = [sum(w["x0"] for w in c) / len(c) for c in clusters]

    numbers = []
    for w in words:
        m = NUMBER_TOKEN_RE.match(w["text"])
        if m:
            numbers.append((int(m.group(1)), w["top"], w["x0"]))

    columns: list[list[tuple[int, float]]] = [[] for _ in centroids]
    for value, top, x0 in numbers:
        idx = min(range(len(centroids)), key=lambda i: abs(x0 - centroids[i]))
        columns[idx].append((value, top))

    result = {}
    for name, values in zip(phase_names, columns):
        if len(values) != 3:
            logging.getLogger(__name__).warning(
                "line-height column %r (%s): expected 3 numbers, found %d -> %r",
                name, team, len(values), values,
            )
        if not values:
            result[name] = {"line_height": None, "front_line": None, "team_length": None}
            continue
        line_height = max(values, key=lambda v: v[1])[0]
        front_line = min(values, key=lambda v: v[1])[0]
        team_length = max(v[0] for v in values) - line_height
        result[name] = {
            "line_height": line_height,
            "front_line": front_line,
            "team_length": team_length,
        }
    return team, result


def parse_line_heights(pdf_path: str, home: str, away: str) -> dict:
    """Extract per-phase line height & team length for both teams.

    Scans every page of the PMSR PDF for "<In Possession|Defensive> Line Height &
    Team Length <Team>" pitch graphics and decodes each one via
    :func:`_parse_line_height_page`.

    Args:
        pdf_path: Path to the PMSR PDF.
        home: Home team name, as parsed from the header page.
        away: Away team name, as parsed from the header page.

    Returns:
        ``{"line_height": {...}, "team_length": {...}}`` where each value is
        ``{"in_possession": {phase: [home, away]}, "defensive": {phase: [home, away]}}``.
        Missing values are ``None``.
    """
    import pdfplumber  # noqa: PLC0415

    line_height = {
        "in_possession": {p: [None, None] for p in _IN_POSSESSION_PHASES},
        "defensive": {p: [None, None] for p in _DEFENSIVE_PHASES},
    }
    team_length = {
        "in_possession": {p: [None, None] for p in _IN_POSSESSION_PHASES},
        "defensive": {p: [None, None] for p in _DEFENSIVE_PHASES},
    }

    def _slot(team: str) -> int | None:
        t = team.strip().lower()
        if t == home.strip().lower():
            return 0
        if t == away.strip().lower():
            return 1
        return None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "Line Height & Team Length" not in text:
                continue
            if "In Possession" in text:
                kind, phase_names = "in_possession", _IN_POSSESSION_PHASES
            elif "Defensive" in text:
                kind, phase_names = "defensive", _DEFENSIVE_PHASES
            else:
                continue
            parsed = _parse_line_height_page(page, phase_names)
            if parsed is None:
                continue
            team, phases = parsed
            slot = _slot(team)
            if slot is None:
                logging.getLogger(__name__).warning(
                    "line-height page team %r doesn't match home=%r/away=%r", team, home, away,
                )
                continue
            for name, vals in phases.items():
                line_height[kind][name][slot] = vals["line_height"]
                team_length[kind][name][slot] = vals["team_length"]

    return {"line_height": line_height, "team_length": team_length}


def parse_pmsr(pdf_path: str) -> dict:
    """Full structured parse: header + key stats + phases + line-height tables."""
    pages = _pages_text(pdf_path)
    txt = "\n".join(pages.values())
    header = parse_header(pages.get(1, ""))
    line_heights = parse_line_heights(pdf_path, header.get("home", ""), header.get("away", ""))
    return {"source": Path(pdf_path).name, **header,
            "key_stats": parse_stats(pages.get(2, "") + "\n" + txt),
            "phases": parse_phases(pages.get(3, "") + "\n" + txt),
            **line_heights}


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = parse_pmsr(args.pdf)
    print(json.dumps(data, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
