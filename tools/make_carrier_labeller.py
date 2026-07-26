"""Generate the offline carrier-labelling tool (a single self-contained ``label.html``).

Sid double-clicks ``results/carrier_attr/labelpack/label.html`` and labels the 90 sampled pass
moments from ``labels.csv``: one moment per screen, the boxed context frames large, the carrier crop
alongside, and the match's real 22-man Sofascore roster as clickable buttons.

The rosters live in cached Sofascore parquets which a browser cannot read, so this generator inlines
them as JSON. What is inlined is deliberately narrow -- roster identity only (name, shirt, position,
substitute flag, minutes played). **No model prediction of any kind is read or written here**: the
only probe artefact touched is ``labels.csv`` itself (blank label columns), never
``*_events.parquet`` / ``*_query_emb.npy`` / ``*_player_counts.csv``.

Run (CPU, seconds)::

    python -m tools.make_carrier_labeller
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

from core import registry
from tools.action_spot_probe import SOFASCORE_MATCH_ID

#: Labelling pack produced by ``tools.carrier_attribution_probe --stage labelpack``.
LABELPACK_DIR = Path("results/carrier_attr/labelpack")
#: Chunk videos are cut at a fixed 600 s; the last chunk of a half is short (see registry videos).
CHUNK_SECONDS = 600.0
#: Registry team name -> Sofascore club name, for the two-way roster split.
TEAM_ALIASES = {"man utd": "manchester united", "man city": "manchester city",
                "spurs": "tottenham hotspur"}
#: Nominal half length used to turn Sofascore ``minutesPlayed`` into an on-pitch window.
HALF_MINUTES = 45.0


def _norm(name: str) -> str:
    """Lowercase a club name and drop punctuation/``fc`` so registry and Sofascore names meet."""
    s = "".join(c if c.isalnum() else " " for c in name.lower())
    s = " ".join(t for t in s.split() if t != "fc")
    return TEAM_ALIASES.get(s, s)


def team_index(teams: tuple[str, ...], sofa_name: str) -> int:
    """Index of a Sofascore club inside the registry's kit-anchor team order.

    Args:
        teams: registry ``match.teams`` (``teams[0]`` is the anchored kit).
        sofa_name: Sofascore ``teamName``.

    Returns:
        0 or 1.

    Raises:
        ValueError: if the club matches neither or both registry teams.
    """
    target = _norm(sofa_name)
    hits = [i for i, t in enumerate(teams) if _norm(t) in target or target in _norm(t)]
    if len(hits) != 1:
        raise ValueError(f"cannot map Sofascore team {sofa_name!r} onto {teams!r}")
    return hits[0]


def minute_estimate(chunk_key: str, frame: int, fps: float = 25.0) -> int:
    """Approximate match minute of a chunk-local frame.

    Chunks are fixed 600 s cuts of a half-length video that starts at kickoff, so the minute is
    ``(600 * chunk_index + frame / fps) / 60`` plus 45 for the second half.

    Args:
        chunk_key: ``h1_chunk_002`` style key.
        frame: chunk-local frame index.
        fps: chunk frame rate.

    Returns:
        Estimated match minute (approximate to about +-2 min: stoppage time and the exact
        half-split point are not modelled).
    """
    half, num = chunk_key.split("_chunk_")
    seconds = CHUNK_SECONDS * int(num) + frame / fps
    return int(seconds / 60.0) + (int(HALF_MINUTES) if half == "h2" else 0)


def roster_for(match_id: str) -> dict:
    """Roster of one match from the cached Sofascore oracle, split by registry team index.

    Args:
        match_id: registry match id.

    Returns:
        ``{"teams": [name0, name1], "players": [...]}`` where each player carries ``name``,
        ``shirt``, ``pos``, ``team`` (0/1), ``unused`` (never came on) and the approximate on-pitch
        window ``on_from`` / ``on_to`` in minutes.
    """
    match = registry.get(match_id)
    path = Path(f"outputs/oracle/sofascore/player_stats_{SOFASCORE_MATCH_ID[match_id]}.parquet")
    df = pd.read_parquet(path)
    players: list[dict] = []
    for r in df.itertuples(index=False):
        mins = float(getattr(r, "minutesPlayed", float("nan")) or 0.0)
        sub = bool(getattr(r, "substitute", False))
        unused = math.isnan(mins) or mins <= 0
        if unused:
            on_from, on_to = 0.0, 0.0
        elif sub:
            on_from, on_to = max(0.0, 2 * HALF_MINUTES - mins), 999.0
        else:
            on_from, on_to = 0.0, 999.0 if mins >= 2 * HALF_MINUTES else mins
        players.append({"name": str(r.name), "shirt": int(getattr(r, "shirtNumber", 0) or 0),
                        "pos": str(getattr(r, "position", "") or ""),
                        "team": team_index(match.teams, str(r.teamName)),
                        "sub": sub, "unused": unused,
                        "on_from": round(on_from), "on_to": round(on_to)})
    return {"teams": list(match.teams), "players": players}


def build_payload(pack_dir: Path = LABELPACK_DIR) -> dict:
    """Inline data for the HTML: CSV columns, the 90 moments and the per-match rosters.

    Args:
        pack_dir: labelling-pack directory holding ``labels.csv`` and the images.

    Returns:
        JSON-serialisable payload.
    """
    with (pack_dir / "labels.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    moments = [{"id": r["id"], "match": r["match"],
                "minute": minute_estimate(r["chunk"], int(r["frame"])), "row": r} for r in rows]
    rosters = {mid: roster_for(mid) for mid in sorted({r["match"] for r in rows})}
    return {"columns": columns, "moments": moments, "rosters": rosters}


def build_html(payload: dict) -> str:
    """Render the single-file labelling app with the payload inlined.

    Args:
        payload: :func:`build_payload` output.

    Returns:
        Complete HTML document.
    """
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if "</script" in blob:
        raise ValueError("payload would close the script tag")
    return _TEMPLATE.replace("/*__PAYLOAD__*/null", blob)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(LABELPACK_DIR), help="labelling-pack directory")
    a = ap.parse_args()
    pack = Path(a.dir)
    payload = build_payload(pack)
    out = pack / "label.html"
    out.write_text(build_html(payload), encoding="utf-8")
    readme = pack / "README.txt"
    if readme.exists() and "label.html" not in (text := readme.read_text(encoding="utf-8")):
        readme.write_text(text + "\nEASIEST WAY: double-click label.html (offline, no server) and "
                          "label with the roster buttons; it exports labels_filled.csv, which "
                          "tools/score_carrier_labels.py reads directly.\n", encoding="utf-8")
    n_players = sum(len(r["players"]) for r in payload["rosters"].values())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): "
          f"{len(payload['moments'])} moments, {n_players} roster entries, "
          f"{len(payload['rosters'])} matches")


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Carrier labelling</title>
<style>
:root { --bg:#14171c; --panel:#1d222a; --line:#2c333e; --fg:#e8edf4; --dim:#8b95a4;
        --acc:#ffd23f; --ok:#3ddc84; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:13px/1.35 system-ui,Segoe UI,sans-serif; }
header { display:flex; gap:10px; align-items:center; padding:6px 10px; background:var(--panel);
         border-bottom:1px solid var(--line); flex-wrap:wrap; }
header .grow { flex:1; }
h1 { font-size:14px; margin:0 8px 0 0; }
button { background:#262d38; color:var(--fg); border:1px solid var(--line); border-radius:5px;
         padding:4px 8px; cursor:pointer; font:inherit; }
button:hover { background:#313a47; }
button.on { background:var(--acc); color:#14171c; border-color:var(--acc); font-weight:600; }
main { display:flex; gap:8px; padding:8px; align-items:flex-start; }
#left { flex:1 1 auto; min-width:0; }
#ctx { width:100%; max-height:66vh; object-fit:contain; background:#000; display:block;
       border:1px solid var(--line); border-radius:6px; }
#strip { display:flex; gap:4px; margin-top:5px; }
#strip button { padding:2px 6px; }
#right { flex:0 0 430px; display:flex; flex-direction:column; gap:6px; }
#crop { max-height:150px; image-rendering:pixelated; border:1px solid var(--acc); border-radius:4px;
        background:#000; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px; }
#answer { font-size:16px; font-weight:600; min-height:22px; }
#answer .none { color:var(--dim); font-weight:400; }
.rowbtns { display:flex; gap:5px; flex-wrap:wrap; }
.teamhead { color:var(--dim); margin:6px 0 3px; font-weight:600; letter-spacing:.4px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:3px; }
.grid button { text-align:left; padding:3px 5px; font-size:12px; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; }
.grid button .sh { color:var(--acc); font-weight:600; margin-right:4px; }
.grid button.off { opacity:.42; }
.grid button.unused { opacity:.25; text-decoration:line-through; cursor:not-allowed; }
.grid button.hide { display:none; }
input[type=text] { width:100%; background:#11141a; color:var(--fg); border:1px solid var(--line);
                   border-radius:5px; padding:4px 6px; font:inherit; }
footer { color:var(--dim); padding:4px 10px 10px; }
kbd { background:#262d38; border:1px solid var(--line); border-radius:3px; padding:0 4px; }
.done { color:var(--ok); }
</style>
</head>
<body>
<header>
  <h1>Carrier labelling</h1>
  <button id="prev">&#8592; Back</button>
  <span id="progress"></span>
  <button id="next">Next &#8594;</button>
  <span id="meta" style="color:var(--dim)"></span>
  <span id="warn" style="color:#ff6b6b;font-weight:600"></span>
  <span class="grow"></span>
  <button id="resume">Resume</button>
  <button id="export">Export CSV</button>
  <button id="importbtn">Import CSV</button>
  <input type="file" id="import" accept=".csv" style="display:none">
</header>
<main>
  <div id="left">
    <img id="ctx" alt="context frame">
    <div id="strip"></div>
  </div>
  <div id="right">
    <div class="card" style="text-align:center">
      <img id="crop" alt="carrier crop"><div style="color:var(--dim)">crop the model saw</div>
    </div>
    <div class="card"><div id="answer"></div></div>
    <div class="card rowbtns">
      <button data-sp="UNKNOWN">[U] UNKNOWN</button>
      <button data-sp="WRONG_PLAYER_BOXED">[W] WRONG BOX</button>
      <button data-sp="NOT_A_PASS">[X] NOT A PASS</button>
      <button id="clear">[0] clear</button>
    </div>
    <div class="card rowbtns">confidence
      <button data-cf="1">1</button><button data-cf="2">2</button><button data-cf="3">3</button>
      <input type="text" id="notes" placeholder="notes [N]" style="flex:1;min-width:120px">
    </div>
    <div class="card">
      <input type="text" id="filter" placeholder="filter players  [/]  Enter picks the only match">
      <div id="rosters"></div>
    </div>
  </div>
</main>
<footer>
<kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> context frame &#183; <kbd>Enter</kbd> save + next &#183;
<kbd>Backspace</kbd> previous &#183; <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> confidence &#183;
<kbd>U</kbd> unknown &#183; <kbd>W</kbd> wrong box &#183; <kbd>X</kbd> not a pass &#183;
<kbd>0</kbd> clear &#183; <kbd>/</kbd> filter players &#183; <kbd>N</kbd> notes &#183;
<kbd>Esc</kbd> leave a text box &#183; <kbd>G</kbd> jump to first unlabelled.
Picking a player sets confidence 3 by default -- press 1 or 2 to lower it.
Greyed = provably off the pitch (unused sub); faded = Sofascore minutes say off at this minute
(approximate, still clickable).
</footer>
<script>
const DATA = /*__PAYLOAD__*/null;
const KEY = "carrier_labels_v1";
const OFFSETS = [-2,-1,0,1,2];
const SPECIAL = ["UNKNOWN","WRONG_PLAYER_BOXED","NOT_A_PASS"];

// Some browsers refuse localStorage on file:// -- degrade to memory and say so loudly.
const $ = id => document.getElementById(id);
function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { warn(); return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { warn(); } }
function warn() {
  $("warn").textContent = "storage blocked -- EXPORT CSV before closing this tab!";
}

let store = JSON.parse(lsGet(KEY) || "{}");
let idx = Math.min(+(lsGet(KEY+"_idx") || 0), DATA.moments.length-1);
let ctxOff = 2;

const cur = () => DATA.moments[idx];
const rec = () => store[cur().id] || {};

function persist() { lsSet(KEY, JSON.stringify(store)); }

function save(patch) {
  const m = cur();
  store[m.id] = Object.assign({player_name:"",confidence_1to3:"",notes:""}, store[m.id], patch);
  persist();
  render();
}

function labelled() {
  return DATA.moments.filter(m => (store[m.id]||{}).player_name).length;
}

function onPitch(p, minute) {
  if (p.unused) return false;
  return minute >= p.on_from - 5 && minute <= p.on_to + 5;
}

function renderRoster() {
  const m = cur(), R = DATA.rosters[m.match], box = $("rosters");
  box.innerHTML = "";
  R.teams.forEach((tname, ti) => {
    const h = document.createElement("div");
    h.className = "teamhead"; h.textContent = tname;
    box.appendChild(h);
    const g = document.createElement("div");
    g.className = "grid";
    R.players.filter(p => p.team === ti).forEach(p => {
      const b = document.createElement("button");
      b.innerHTML = '<span class="sh">' + (p.shirt||"-") + "</span>";
      b.appendChild(document.createTextNode(p.name));
      b.dataset.name = p.name;
      b.title = p.name + " (" + p.pos + ")" +
        (p.unused ? " -- unused substitute" :
         p.sub ? " -- came on about " + p.on_from + "'" :
         p.on_to < 999 ? " -- off about " + p.on_to + "'" : " -- played the full match");
      if (p.unused) { b.className = "unused"; b.disabled = true; }
      else if (!onPitch(p, m.minute)) b.className = "off";
      if (rec().player_name === p.name) b.classList.add("on");
      b.onclick = () => pick(p.name);
      g.appendChild(b);
    });
    box.appendChild(g);
  });
  applyFilter();
}

function pick(name) {
  const r = rec();
  save({player_name:name, confidence_1to3: r.confidence_1to3 || "3"});
}

function render() {
  const m = cur(), r = rec();
  $("progress").innerHTML = "<b>" + (idx+1) + "</b> of " + DATA.moments.length +
    ' &#183; <span class="' + (labelled()===DATA.moments.length ? "done" : "") + '">' +
    labelled() + " labelled</span>";
  $("meta").textContent = m.match + "  " + m.id.slice(0,3) + "  ~" + m.minute + "'";
  $("ctx").src = m.id + "_ctx" + (OFFSETS[ctxOff] >= 0 ? "+" : "") + OFFSETS[ctxOff] + ".jpg";
  $("crop").src = m.id + "_crop.jpg";
  $("strip").innerHTML = "";
  OFFSETS.forEach((o,i) => {
    const b = document.createElement("button");
    b.textContent = (o>0?"+":"") + o + (o===0 ? " (kick)" : "");
    if (i === ctxOff) b.className = "on";
    b.onclick = () => { ctxOff = i; render(); };
    $("strip").appendChild(b);
  });
  $("answer").innerHTML = r.player_name
    ? r.player_name + ' <span style="color:var(--dim)">conf ' + (r.confidence_1to3||"-") + "</span>"
    : '<span class="none">not labelled yet</span>';
  document.querySelectorAll("[data-sp]").forEach(b =>
    b.classList.toggle("on", b.dataset.sp === r.player_name));
  document.querySelectorAll("[data-cf]").forEach(b =>
    b.classList.toggle("on", b.dataset.cf === (r.confidence_1to3||"")));
  if ($("notes") !== document.activeElement) $("notes").value = r.notes || "";
  renderRoster();
  lsSet(KEY+"_idx", idx);
  const nx = DATA.moments[idx+1];
  if (nx) OFFSETS.forEach(o => {
    (new Image()).src = nx.id + "_ctx" + (o>=0?"+":"") + o + ".jpg";
  });
}

function go(d) {
  idx = Math.max(0, Math.min(DATA.moments.length-1, idx+d));
  ctxOff = 2;
  $("filter").value = "";
  render();
}

function applyFilter() {
  const q = $("filter").value.trim().toLowerCase();
  document.querySelectorAll("#rosters .grid button").forEach(b =>
    b.classList.toggle("hide", q !== "" && !b.dataset.name.toLowerCase().includes(q)));
}

// --- csv ------------------------------------------------------------------
function csvCell(v) {
  v = (v === undefined || v === null) ? "" : String(v);
  return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}

function exportCsv() {
  const cols = DATA.columns;
  const lines = [cols.join(",")];
  DATA.moments.forEach(m => {
    const r = Object.assign({}, m.row, store[m.id] || {});
    lines.push(cols.map(c => csvCell(r[c])).join(","));
  });
  const blob = new Blob(["\uFEFF" + lines.join("\r\n") + "\r\n"],
                        {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_filled.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", q = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i+1] === '"') { cell += '"'; i++; }
      else if (c === '"') q = false;
      else cell += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (c !== "\r") cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

function importCsv(text) {
  const rows = parseCsv(text).filter(r => r.length > 1);
  if (!rows.length) return;
  const head = rows[0];
  if (head.indexOf("id") < 0) { alert("no id column found"); return; }
  let n = 0;
  rows.slice(1).forEach(r => {
    const o = {};
    head.forEach((h,i) => o[h] = r[i]);
    if (!o.id || !o.player_name) return;
    store[o.id] = {player_name:o.player_name, confidence_1to3:o.confidence_1to3||"",
                   notes:o.notes||""};
    n++;
  });
  persist();
  alert("imported " + n + " labels");
  render();
}

// --- wiring ---------------------------------------------------------------
$("prev").onclick = () => go(-1);
$("next").onclick = () => go(1);
$("clear").onclick = () => { delete store[cur().id]; persist(); render(); };
$("export").onclick = exportCsv;
$("importbtn").onclick = () => $("import").click();
$("import").onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const fr = new FileReader();
  fr.onload = () => importCsv(fr.result);
  fr.readAsText(f, "utf-8");
  e.target.value = "";
};
$("resume").onclick = () => {
  const i = DATA.moments.findIndex(m => !(store[m.id]||{}).player_name);
  idx = i < 0 ? DATA.moments.length-1 : i;
  ctxOff = 2; render();
};
document.querySelectorAll("[data-sp]").forEach(b =>
  b.onclick = () => save({player_name:b.dataset.sp, confidence_1to3:rec().confidence_1to3||"3"}));
document.querySelectorAll("[data-cf]").forEach(b =>
  b.onclick = () => save({confidence_1to3:b.dataset.cf}));
$("notes").oninput = () => save({notes:$("notes").value});
$("filter").oninput = applyFilter;
$("filter").onkeydown = e => {
  if (e.key === "Escape") { $("filter").value = ""; applyFilter(); $("filter").blur(); }
  if (e.key === "Enter") {
    const vis = [...document.querySelectorAll("#rosters .grid button")]
      .filter(b => !b.classList.contains("hide") && !b.disabled);
    if (vis.length === 1) { pick(vis[0].dataset.name); $("filter").value = ""; applyFilter(); }
    e.preventDefault();
  }
};
$("notes").onkeydown = e => { if (e.key === "Escape" || e.key === "Enter") $("notes").blur(); };

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (e.key === "ArrowLeft")  { ctxOff = Math.max(0, ctxOff-1); render(); }
  else if (e.key === "ArrowRight") { ctxOff = Math.min(OFFSETS.length-1, ctxOff+1); render(); }
  else if (e.key === "Enter") go(1);
  else if (e.key === "Backspace") go(-1);
  else if (k === "1" || k === "2" || k === "3") save({confidence_1to3:k});
  else if (k === "u") save({player_name:SPECIAL[0], confidence_1to3:rec().confidence_1to3||"3"});
  else if (k === "w") save({player_name:SPECIAL[1], confidence_1to3:rec().confidence_1to3||"3"});
  else if (k === "x") save({player_name:SPECIAL[2], confidence_1to3:rec().confidence_1to3||"3"});
  else if (k === "0") { delete store[cur().id]; persist(); render(); }
  else if (k === "/") { $("filter").focus(); }
  else if (k === "n") { $("notes").focus(); }
  else if (k === "g") { $("resume").click(); }
  else return;
  e.preventDefault();
});
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
