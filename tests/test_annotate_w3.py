"""The W3 annotation server writes a manifest the ingest still parses."""
from __future__ import annotations

import csv
import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

import tools.gsr_w3_corpus as w3
from tools.annotate_w3 import MANIFEST, Store, make_handler
from tools.gsr_crops import Crop
from tools.gsr_w3_corpus import _parse_a, _parse_b


def _post(url: str, body: dict) -> tuple[int, dict]:
    """POST JSON, returning ``(status, payload)`` for both accept and reject."""
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"),  # noqa: S310
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:  # noqa: S310
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["queue_id"]: r for r in csv.DictReader(fh)}


@pytest.mark.skipif(not MANIFEST.exists(), reason="labelling manifest not built")
def test_commit_roundtrip_and_ingest_contract(tmp_path: Path) -> None:
    """Valid tier-A/B commits land in the CSV; invalid ones leave it byte-identical."""
    man = tmp_path / "labelling_manifest.csv"
    shutil.copyfile(MANIFEST, man)
    store = Store(man)
    a_id = next(r["queue_id"] for r in store.rows if r["tier"] == "A")
    b_id = next(r["queue_id"] for r in store.rows if r["tier"] == "B")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/label"
    try:
        assert _post(url, {"id": a_id, "label": " 07 ", "note": "leading zero", "ts": "t0"})[0] == 200
        assert _post(url, {"id": b_id, "label": "9,1,5", "note": "", "ts": "t1"})[0] == 200
        good = man.read_bytes()

        for bad in ({"id": a_id, "label": "100"}, {"id": a_id, "label": ""},
                    {"id": b_id, "label": "13"}, {"id": b_id, "label": "1;5"},
                    {"id": "ZZZZ", "label": "7"}):
            code, payload = _post(url, bad)
            assert code == 400 and payload["error"], bad
        assert man.read_bytes() == good, "a rejected commit must not touch the CSV"
    finally:
        httpd.shutdown()
        httpd.server_close()

    out = _rows(man)
    assert out[a_id]["label"] == "7" and out[a_id]["note"] == "leading zero"
    assert out[b_id]["label"] == "1,5,9"
    assert len(out) == len(store.rows) == 499  # noqa: PLR2004
    assert good.count(b"\r\n") == len(out) + 1, "CRLF line endings must survive"
    assert list(_rows(MANIFEST)[a_id]) == list(out[a_id]), "column order must survive"

    audit = [json.loads(ln) for ln in
             (tmp_path / "labels_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(e["ts"], e["queue_id"], e["new_label"]) for e in audit] == [
        ("t0", a_id, "7"), ("t1", b_id, "1,5,9")]

    for r in out.values():  # the ingest contract, on every row of the file we wrote
        if r["tier"] == "A":
            _parse_a(r["label"])
        else:
            _parse_b(r["label"], int(r["n_cells"]))


def test_ingest_tier_b_blank_and_unsure_emit_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank or 'unsure' tier-B row must not fabricate 12 human-negative rows.

    Mirrors the tier-A guard (``verdict in ("unsure", "blank") -> continue``): tier-B rows the
    annotator never answered must not enter the corpus at all, while an actual "none" answer still
    produces its 12 human-verified no-number rows.
    """
    seq = "S1"
    frames = [f"{i:06d}" for i in range(12)]
    crops = [Crop(seq, f, 1, (0, 0, 40, 100), "player", "left", "7") for f in frames]
    monkeypatch.setattr(w3, "split_sequences", lambda split: [seq])  # noqa: ARG005
    monkeypatch.setattr(w3, "_all_boxes", lambda s: {1: crops})  # noqa: ARG005
    monkeypatch.setattr(w3, "leg_path", lambda tag: tmp_path / "no_such.parquet")  # noqa: ARG005
    monkeypatch.setattr(w3, "OUT_ROOT", tmp_path)

    fields = ["queue_id", "tier", "sheet", "sequence", "tracklet_id", "n_frames",
              "n_readable_frames", "role", "gt_number", "n_cells", "median_view_h",
              "max_view_h", "cell_frames", "label", "note"]
    rows = [{**dict.fromkeys(fields, ""), "queue_id": qid, "tier": "B", "sequence": seq,
             "tracklet_id": "1", "gt_number": "7", "n_cells": "12",
             "cell_frames": ";".join(frames), "label": label}
            for qid, label in (("B0001", ""), ("B0002", "unsure"), ("B0003", "none"))]
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    df = pd.read_parquet(w3.ingest(csv_path))
    assert len(df) == 12, "only the 'none' row's 12 cells may reach the corpus"
    assert set(df["label"]) == {"<none>"}
