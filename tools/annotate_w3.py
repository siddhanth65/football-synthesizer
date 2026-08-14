"""Local, keyboard-first labelling UI for the W3 tracklet annotation queue.

``python -m tools.annotate_w3`` starts a localhost server, opens a browser, and serves one page
that walks ``outputs/gsr/w3_annotation/labelling_manifest.csv`` row by row with its contact sheet.
Every commit rewrites that CSV atomically, so the CSV stays the single source of truth and
``tools.gsr_w3_corpus --ingest`` keeps working unchanged.

Validation is not re-implemented here: commits go through the ingest parsers themselves
(:func:`tools.gsr_w3_corpus._parse_a` / ``_parse_b``), so anything the UI accepts is by construction
something the ingest accepts.

Contact-sheet geometry comes from ``tools.gsr_w3_corpus._sheet``: ``SHEET_ROWS x SHEET_COLS``
cells of ``CELL_H x CELL_W`` px tiling the whole JPEG, cell 1 top-left, filled row-major. The
overlay is therefore a plain percentage grid and needs no pixel arithmetic.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.gsr_w3_corpus import OUT_ROOT, SHEET_COLS, SHEET_ROWS, _parse_a, _parse_b

MANIFEST = OUT_ROOT / "labelling_manifest.csv"
SHEETS = OUT_ROOT / "sheets"
#: First port tried; the next few are tried in turn if it is busy.
PORT = 8377
MAX_NOTE = 500


class Store:
    """The manifest held in memory, with atomic write-back on every accepted commit.

    Attributes:
        path: The manifest CSV.
        fields: Column order exactly as read, so the file round-trips byte-compatibly.
        rows: One dict per manifest row, in file order.
    """

    def __init__(self, path: Path = MANIFEST) -> None:
        """Load the manifest, keeping any labels already typed in Excel."""
        self.path = path
        self.audit = path.with_name("labels_audit.jsonl")
        self.lock = threading.Lock()
        with path.open(encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            self.fields = list(rd.fieldnames or [])
            self.rows = list(rd)
        self.index = {r["queue_id"]: i for i, r in enumerate(self.rows)}

    def bad_existing(self) -> list[tuple[str, str]]:
        """Rows whose pre-existing label would not survive the ingest parsers."""
        bad = []
        for r in self.rows:
            if not r["label"].strip():
                continue
            try:
                canonical(r["tier"], r["label"], int(r["n_cells"]))
            except ValueError as exc:
                bad.append((r["queue_id"], f"{r['label']!r}: {exc}"))
        return bad

    def commit(self, queue_id: str, label: str, note: str, ts: str) -> dict[str, str]:
        """Validate and store one label, rewriting the CSV atomically.

        Args:
            queue_id: Row id, e.g. ``A0007``.
            label: Raw label text from the UI.
            note: Free-text note (may be empty).
            ts: Client-side timestamp for the audit trail.

        Returns:
            The updated row.

        Raises:
            ValueError: If the id is unknown or the label is not an allowed value.
        """
        if queue_id not in self.index:
            raise ValueError(f"unknown queue_id {queue_id!r}")
        row = self.rows[self.index[queue_id]]
        new = canonical(row["tier"], label, int(row["n_cells"]))
        note = " ".join(note.split())[:MAX_NOTE]
        with self.lock:
            old, old_note = row["label"], row["note"]
            row["label"], row["note"] = new, note
            self._write()
            with self.audit.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": ts or _now(), "queue_id": queue_id, "old_label": old,
                                     "new_label": new, "old_note": old_note,
                                     "new_note": note}) + "\n")
        return row

    def _write(self) -> None:
        """Rewrite the manifest via a temp file + ``os.replace`` (same dir, same encoding)."""
        tmp = self.path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)


def canonical(tier: str, label: str, n_cells: int) -> str:
    """Canonical stored form of one answer, or ``ValueError``.

    Args:
        tier: ``A`` or ``B``.
        label: Raw answer text.
        n_cells: Number of cells on this row's sheet (12 for every current row).

    Returns:
        ``"7"`` / ``"none"`` / ``"unsure"`` for tier A; ``"1,5,9"`` / ``"all"`` / ``"none"`` /
        ``"unsure"`` for tier B.

    Raises:
        ValueError: On an empty or disallowed answer.
    """
    s = " ".join(label.split()).lower()
    if not s:
        raise ValueError("empty label")
    if tier == "A":
        num, verdict = _parse_a(s)
        return num if num else verdict
    if s in ("all", "none", "unsure"):
        return s
    cells = _parse_b(s, n_cells)
    if not cells:
        raise ValueError(f"tier-B label must be cells 1-{n_cells} / all / none / unsure")
    return ",".join(str(c + 1) for c in sorted(set(cells)))


def _now() -> str:
    """Server-side ISO timestamp, used only when the client sends none."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to ``store``."""

    sheet_names = {r["sheet"] for r in store.rows}
    keys = ("queue_id", "tier", "sheet", "sequence", "tracklet_id", "n_frames", "role",
            "gt_number", "n_cells", "median_view_h", "label", "note")
    page = PAGE.replace("__ROWS__", json.dumps([{k: r[k] for k in keys} for r in store.rows]))
    page_bytes = page.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        """Serves the single page, the sheet JPEGs and the label POST endpoint."""

        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            """Serve the page, or one sheet JPEG from the manifest's own file list."""
            if self.path in ("/", "/index.html"):
                self._send(200, page_bytes, "text/html; charset=utf-8")
            elif self.path.startswith("/sheets/"):
                name = self.path[len("/sheets/"):]
                if name not in sheet_names:  # whitelist, not a path join
                    self._send(404, b"no such sheet", "text/plain")
                    return
                self._send(200, (SHEETS / name).read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            """Accept one label commit; reply ``{ok, label}`` or ``{ok:false, error}``."""
            if self.path != "/label":
                self._send(404, b"not found", "text/plain")
                return
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                row = store.commit(str(req.get("id", "")), str(req.get("label", "")),
                                   str(req.get("note", "")), str(req.get("ts", "")))
            except (ValueError, KeyError) as exc:
                self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                           "application/json")
                return
            self._send(200, json.dumps({"ok": True, "label": row["label"],
                                        "note": row["note"]}).encode("utf-8"), "application/json")

        def log_message(self, fmt: str, *args: object) -> None:
            """Stay quiet: the browser page is the operator's feedback channel."""

    return Handler


def serve(port: int = PORT, *, open_browser: bool = True) -> None:
    """Run the annotation server until Ctrl-C."""
    store = Store()
    for bad_id, why in store.bad_existing():
        print(f"WARNING: existing label in {bad_id} is not ingest-valid -- {why}")
    for p in range(port, port + 10):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), make_handler(store))
        except OSError:
            continue
        break
    else:
        raise SystemExit(f"no free port in {port}..{port + 9}")
    url = f"http://127.0.0.1:{p}/"
    a = sum(1 for r in store.rows if r["tier"] == "A" and r["label"].strip())
    b = sum(1 for r in store.rows if r["tier"] == "B" and r["label"].strip())
    print(f"manifest {store.path} ({len(store.rows)} rows; A {a} / B {b} already labelled)")
    print(f"serving {url}  --  Ctrl-C to stop")
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; manifest is saved after every commit")
    finally:
        httpd.server_close()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>W3 labelling</title><style>
:root { color-scheme: dark; }
body { background:#141618; color:#dfe3e6; font:14px/1.45 system-ui,sans-serif; margin:0;
       display:flex; flex-direction:column; height:100vh; overflow:hidden; }
header { padding:6px 12px; background:#1c1f22; border-bottom:1px solid #2b2f33;
         display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
#pos { font-weight:700; font-size:16px; }
.gk { color:#ffd166; font-weight:700; }
#gt { font-size:26px; font-weight:800; color:#7fd1ff; }
#saved { color:#8de08d; }
#pend { font-size:22px; font-weight:800; color:#ffd166; min-width:70px; }
#err.bad { color:#ff8b7a; font-weight:600; }
#err.ok { color:#8de08d; font-weight:600; }
#err.hint { color:#9aa3ab; }
#mode { color:#ffd166; font-weight:700; }
main { flex:1; display:flex; align-items:center; justify-content:center; overflow:hidden; }
#wrap { position:relative; overflow:hidden; }
#inner { position:relative; display:inline-block; transform-origin:var(--ox) var(--oy); }
#wrap.zoom #inner { transform:scale(2.6); }
#sheet { display:block; height:calc(100vh - 150px); width:auto; }
#ov { position:absolute; inset:0; display:grid;
      grid-template-columns:repeat(__COLS__,1fr); grid-template-rows:repeat(__ROWS_N__,1fr); }
#ov.hide { display:none; }
#ov div { border:1px solid rgba(255,255,255,.16); position:relative; cursor:pointer; }
#ov div.sel { background:rgba(80,220,130,.28); border:3px solid #4ade80; }
#ov div span { position:absolute; top:2px; right:4px; font-size:11px; color:#9aa3ab; }
#ov div.sel span { color:#eafff0; font-weight:700; }
footer { padding:5px 12px; background:#1c1f22; border-top:1px solid #2b2f33; color:#9aa3ab;
         font-size:12px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
input { background:#0f1113; color:#dfe3e6; border:1px solid #3a4045; border-radius:3px;
        padding:4px 6px; font:13px system-ui,sans-serif; width:320px; }
input:focus { outline:none; border-color:#ffd166; box-shadow:0 0 0 2px rgba(255,209,102,.35); }
</style></head><body>
<header>
  <span id="pos"></span><span id="meta"></span><span id="gt"></span>
  <span id="saved"></span><span id="pend"></span><span id="err"></span>
  <span style="margin-left:auto" id="prog"></span>
</header>
<main><div id="wrap"><div id="inner">
  <img id="sheet" alt="contact sheet"><div id="ov"></div>
</div></div></main>
<footer>
  <span>note (Tab):</span><input id="note" placeholder="optional -- wrong person / wrong number">
  <span id="mode"></span><span id="keys"></span>
</footer>
<script>
const R = __ROWS__;
const NCELL = __NCELL__;
const KEYMAP = ["1","2","3","4","5","6","7","8","9","0","-","="];
let i = 0, pend = "", cells = new Set();
const $ = id => document.getElementById(id);

function nextUn(from) { for (let k = from; k < R.length; k++) if (!R[k].label) return k; return -1; }

function go(k) {
  i = Math.max(0, Math.min(R.length - 1, k));
  pend = ""; cells = new Set();
  const r = R[i];
  if (r.tier === "B" && r.label && !["all","none","unsure"].includes(r.label))
    r.label.split(",").forEach(c => cells.add(+c));
  $("note").value = r.note || "";
  $("err").textContent = ""; $("err").className = "";
  render();
}

// The answer buffer holds digits OR a word ("none"/"unsure"/"all"): whatever a key puts in it,
// Enter commits. Letters commit at once as well, so both habits produce the same label.
const word = w => { pend = w; render(); commit(w); };
const hint = m => { $("err").className = "hint"; $("err").textContent = m; };

function render() {
  const r = R[i], A = r.tier === "A";
  $("pos").textContent = `${r.queue_id}  (${i + 1}/${R.length})`;
  $("meta").innerHTML = `tier ${r.tier} &middot; ${r.sequence} t${r.tracklet_id} &middot; ` +
    (r.role === "goalkeeper" ? `<span class="gk">GOALKEEPER</span>` : r.role) +
    ` &middot; ${r.n_frames} frames`;
  $("gt").textContent = A ? "" : "GT " + r.gt_number;
  $("saved").textContent = r.label ? "saved: " + r.label : "";
  $("pend").textContent = pend ? (/^[0-9]+$/.test(pend) ? pend + "_" : pend)
    : (!A && cells.size ? [...cells].sort((a, b) => a - b).join(",") : "");
  const a = R.filter(x => x.tier === "A"), b = R.filter(x => x.tier === "B");
  $("prog").textContent = `A ${a.filter(x => x.label).length}/${a.length}  ` +
    `B ${b.filter(x => x.label).length}/${b.length}`;
  $("sheet").src = "/sheets/" + r.sheet;
  $("ov").classList.toggle("hide", A);
  $("keys").textContent = A
    ? "type 1-99 then Enter | n = none | u = unsure (n/u save on their own too) | <- -> move | "
      + "g = next blank | Tab = note | hold Shift = zoom"
    : "1-9 0 - = toggle cells 1-12 (or click) | Enter = commit | a all | n none | u unsure | "
      + "<- -> | g | Tab = note | hold Shift = zoom";
  if (!A) [...$("ov").children].forEach((d, k) => d.classList.toggle("sel", cells.has(k + 1)));
}

async function commit(label, stay) {
  const r = R[i];
  const res = await fetch("/label", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: r.queue_id, label: label, note: $("note").value,
                          ts: new Date().toISOString()})});
  const j = await res.json();
  if (!j.ok) { $("err").className = "bad"; $("err").textContent = "rejected: " + j.error; return; }
  r.label = j.label; r.note = j.note;
  const n = nextUn(i + 1);
  if (stay) render(); else go(n < 0 ? i + 1 : n);
  $("err").className = "ok";  // set after go(), which clears the line
  $("err").textContent = `${r.queue_id} = ${j.label} saved`;
}

function commitPending() {
  const r = R[i];
  if (pend) commit(pend);
  else if (r.tier === "B" && cells.size) commit([...cells].sort((a, b) => a - b).join(","));
  else if ($("err").className !== "ok")  // never nag straight after a successful save
    hint(r.tier === "A" ? "type a number, then Enter -- or n = none, u = unsure"
                        : "click or type cells, then Enter -- or a = all, n = none, u = unsure");
}

document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.target.id === "note") {  // a text field: every letter belongs to the note, not to labelling
    if (e.key === "Escape" || e.key === "Enter") { $("note").blur(); e.preventDefault(); }
    return;  // the blur handler saves the note
  }
  if (e.key === "Tab") { $("note").focus(); e.preventDefault(); return; }
  const r = R[i], A = r.tier === "A";
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (k === "ArrowLeft") go(i - 1);
  else if (k === "ArrowRight") go(i + 1);
  else if (k === "g") { const n = nextUn(i + 1); if (n >= 0) go(n); }
  else if (k === "n") word("none");
  else if (k === "u") word("unsure");
  else if (k === "Enter") commitPending();
  else if (A && k >= "0" && k <= "9") { pend = (pend + k).slice(-2); render(); }
  else if (A && k === "Backspace") { pend = pend.slice(0, -1); render(); }
  else if (!A && k === "a") word("all");
  else if (!A && KEYMAP.includes(k)) {
    const c = KEYMAP.indexOf(k) + 1;
    if (c <= +r.n_cells) { cells.has(c) ? cells.delete(c) : cells.add(c); render(); }
  } else return;
  e.preventDefault();
});

$("note").addEventListener("focus", () => {
  $("mode").textContent = "NOTE MODE -- letters go in the box; Esc returns to labelling";
});
$("note").addEventListener("blur", () => {
  $("mode").textContent = "";
  const r = R[i];  // an answered row would otherwise lose a note typed after the answer
  if (r.label && $("note").value !== r.note) commit(r.label, true);
});

const ov = $("ov");
for (let c = 1; c <= NCELL; c++) {
  const d = document.createElement("div");
  d.innerHTML = `<span>${c} (${KEYMAP[c - 1]})</span>`;
  d.onclick = () => { cells.has(c) ? cells.delete(c) : cells.add(c); render(); };
  ov.appendChild(d);
}
const wrap = $("wrap");
wrap.addEventListener("mousemove", e => {
  const b = wrap.getBoundingClientRect();
  wrap.style.setProperty("--ox", (e.clientX - b.left) + "px");
  wrap.style.setProperty("--oy", (e.clientY - b.top) + "px");
  wrap.classList.toggle("zoom", e.shiftKey);
});
wrap.addEventListener("mouseleave", () => wrap.classList.remove("zoom"));
document.addEventListener("keyup", e => { if (!e.shiftKey) wrap.classList.remove("zoom"); });
go(Math.max(0, nextUn(0)));
</script></body></html>
"""
PAGE = (PAGE.replace("__COLS__", str(SHEET_COLS)).replace("__ROWS_N__", str(SHEET_ROWS))
        .replace("__NCELL__", str(SHEET_COLS * SHEET_ROWS)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local labelling UI for the W3 annotation queue.")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    serve(args.port, open_browser=not args.no_browser)
