"""Generate the FIFA-EFI-style team report (the product), A3.

Renders the per-team 4-facet fingerprint (:mod:`fingerprint.facet_metrics`) + the in/out-of-possession
phase split (:mod:`fingerprint.phase_metrics`) into a single EFI-style HTML: a two-team comparison
(mirrored bars per metric, grouped by Attacking / Defending / Passing / Goalkeeping / Physical) plus the
possession-phase line-height contrast and an honest-scope footer. No Jinja/Matplotlib dependency -- pure
string templating so it runs anywhere.

The synthesizer's *forecast* (:func:`build_team_report`) layers on later; this is the descriptive half.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORT_DIR = Path("results/reports")
_RED, _BLUE, _INK = "#e6194B", "#4363d8", "#1a2342"

# output column -> (label, unit). Grouped by facet for the report sections.
_FACET_LABELS: dict[str, list[tuple[str, str, str]]] = {
    "Attacking": [("att_buildup_height", "Build-up height", "m"), ("att_third_share", "Attacking-third share", ""),
                  ("att_surface_m2", "Territory (hull)", "m²"), ("att_centre_share", "Central-lane share", ""),
                  ("att_wing_share", "Wing share", ""), ("att_threat_xt", "Threat created (xT)", ""),
                  ("att_success_p", "Attack success", ""), ("att_option_richness", "Passing-option richness", "")],
    "Defending": [("def_line_height", "Defensive line height", "m"), ("def_compactness_m", "Block compactness", "m"),
                  ("def_block_width_m", "Block width", "m"), ("def_recovery_p", "Ball recovery", ""),
                  ("def_press_decisiveness", "Press decisiveness", ""), ("def_lane_suppression", "Lane suppression", "")],
    "Territory (pitch control)": [("space_control", "Space control", ""), ("att_third_control", "Attacking-third control", "")],
    "Passing": [("pass_nearest_mate_m", "Nearest team-mate", "m"), ("pass_depth_m", "Team depth", "m")],
    "Goalkeeping": [("gk_sweeper_height_m", "Sweeper height", "m"), ("gk_lateral_range_m", "Keeper lateral range", "m")],
    "Coordination": [("velocity_synchrony", "Movement synchrony", "")],
    "Physical": [("phys_top_speed_kmh", "Top speed", "km/h"), ("phys_sprint_share", "Sprint share", "")],
}


def _fmt(v: float, unit: str) -> str:
    if pd.isna(v):
        return "&ndash;"
    return (f"{v:.0%}" if unit == "" and abs(v) <= 1 else f"{v:.3f}" if unit == "" else f"{v:.1f}{unit}")


def _row(label: str, a: float, b: float, unit: str) -> str:
    """A mirrored comparison row: team-A bar grows left, team-B bar grows right, scaled to the larger."""
    hi = max(abs(a) if not pd.isna(a) else 0, abs(b) if not pd.isna(b) else 0) or 1.0
    wa, wb = (0 if pd.isna(a) else abs(a) / hi * 100), (0 if pd.isna(b) else abs(b) / hi * 100)
    return (
        f'<div class="row"><div class="val">{_fmt(a, unit)}</div>'
        f'<div class="barwrap"><div class="bar a" style="width:{wa:.0f}%"></div></div>'
        f'<div class="label">{label}</div>'
        f'<div class="barwrap r"><div class="bar b" style="width:{wb:.0f}%"></div></div>'
        f'<div class="val">{_fmt(b, unit)}</div></div>'
    )


def _phase_block(phases: pd.DataFrame, team: int, colour: str) -> str:
    if phases.empty:
        return ""
    t = phases[phases["team"] == team].set_index("phase")
    cells = []
    for phase, title in [("in_poss", "In possession"), ("out_poss", "Out of possession")]:
        if phase not in t.index:
            continue
        r = t.loc[phase]
        cells.append(f'<div class="phasecard"><h4>{title}</h4>'
                     f'<div>Build-up / line: <b>{r["buildup_height"]:.0f}</b> / <b>{r["def_line_height"]:.0f}</b> m</div>'
                     f'<div>Width: <b>{r["width"]:.0f}</b> m &nbsp; Compact: <b>{r["compactness"]:.1f}</b> m</div></div>')
    return f'<div class="phaserow" style="border-color:{colour}">{"".join(cells)}</div>'


def render_match_report(facets: pd.DataFrame, phases: pd.DataFrame, *,
                        team_names: tuple[str, str] = ("Team 0", "Team 1"),
                        match_title: str = "Broadcast match — CV team report",
                        out_path: Path = REPORT_DIR / "match_report.html") -> Path:
    """Render the descriptive EFI-style two-team report to ``out_path`` and return it."""
    f = facets.set_index("team")
    t0, t1 = 0, 1

    def cell(col):
        return (f.loc[t0, col] if col in f.columns and t0 in f.index else float("nan"),
                f.loc[t1, col] if col in f.columns and t1 in f.index else float("nan"))

    sections = []
    for facet, metrics in _FACET_LABELS.items():
        rows = "".join(_row(lbl, *cell(col), unit) for col, lbl, unit in metrics if col in f.columns)
        if rows:
            sections.append(f'<section><h3>{facet}</h3>{rows}</section>')
    phase_html = (f'<section><h3>Line height by phase</h3>'
                  f'<div class="phasecols"><div><h4 style="color:{_RED}">{team_names[0]}</h4>'
                  f'{_phase_block(phases, t0, _RED)}</div>'
                  f'<div><h4 style="color:{_BLUE}">{team_names[1]}</h4>'
                  f'{_phase_block(phases, t1, _BLUE)}</div></div></section>') if not phases.empty else ""

    html = _TEMPLATE.format(
        title=match_title, red=_RED, blue=_BLUE, ink=_INK, a=team_names[0], b=team_names[1],
        sections="".join(sections), phases=phase_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f4f6fb;color:{ink};margin:0;padding:0 0 60px}}
header{{background:{ink};color:#fff;padding:28px 40px}}
header h1{{margin:0;font-size:24px;letter-spacing:.5px}}
header .teams{{margin-top:10px;font-size:20px;font-weight:700}}
header .teams .a{{color:{red}}} header .teams .b{{color:{blue}}}
.wrap{{max-width:920px;margin:24px auto;padding:0 20px}}
section{{background:#fff;border-radius:12px;padding:18px 24px;margin:18px 0;box-shadow:0 2px 8px rgba(20,30,70,.06)}}
section h3{{margin:0 0 14px;color:{ink};border-bottom:2px solid #eef1f7;padding-bottom:8px;font-size:18px}}
.row{{display:grid;grid-template-columns:64px 1fr 220px 1fr 64px;align-items:center;gap:8px;margin:7px 0}}
.val{{font-variant-numeric:tabular-nums;font-weight:700;font-size:13px}} .row .val:first-child{{text-align:right;color:{red}}}
.row .val:last-child{{text-align:left;color:{blue}}}
.label{{text-align:center;font-size:13px;color:#516}}
.barwrap{{height:12px;background:#eef1f7;border-radius:6px;overflow:hidden;display:flex;justify-content:flex-end}}
.barwrap.r{{justify-content:flex-start}}
.bar{{height:100%}} .bar.a{{background:{red}}} .bar.b{{background:{blue}}}
.phasecols{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.phaserow{{border-left:4px solid;padding-left:12px}} .phasecard{{margin:8px 0;font-size:13px}}
.phasecard h4{{margin:2px 0;font-size:13px;color:{ink}}}
footer{{max-width:920px;margin:10px auto;padding:0 24px;color:#889;font-size:12px;line-height:1.5}}
</style></head><body>
<header><h1>{title}</h1><div class="teams"><span class="a">{a}</span> vs <span class="b">{b}</span></div></header>
<div class="wrap">{sections}{phases}</div>
<footer><b>Scope &amp; caveats.</b> Metrics are derived from broadcast video (partial visibility, ~7&ndash;15 of
22 players per frame; metre-scale positions). Attacking/Defending combine positional shape with the trained
relational model (xT / success / recovery, attributed via ball-carrier frames). Passing &amp; goalkeeping are
<i>positional proxies</i> &mdash; true pass networks, line breaks and GK distribution need ball-event tracking
(in progress). Phase split is a coarse carry-forward from sparse ball-carrier detections. Team identities are
colour-anchored (not named).</footer>
</body></html>"""


def build_team_report(team: str, opponent: str | None = None, *, out_dir: Path = REPORT_DIR) -> Path:
    """Forecast report for ``team`` (optionally vs ``opponent``) from the synthesizer. TODO (needs C5)."""
    raise NotImplementedError("Predictive forecast needs the C5 synthesizer; use render_match_report "
                              "for the descriptive report.")


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facets", required=True, help="match_facets parquet")
    ap.add_argument("--phases", default=None, help="match_phases parquet (optional)")
    ap.add_argument("--out", default=str(REPORT_DIR / "match_report.html"))
    ap.add_argument("--names", default="Team 0,Team 1", help="comma-separated team names")
    args = ap.parse_args()
    phases = pd.read_parquet(args.phases) if args.phases else pd.DataFrame()
    a, b = (args.names.split(",", 1) + ["Team 1"])[:2]
    out = render_match_report(pd.read_parquet(args.facets), phases,
                              team_names=(a.strip(), b.strip()), out_path=Path(args.out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
