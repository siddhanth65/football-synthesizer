# knowledge/ — the project's claims graph

`claims.json` is a flat list of claim nodes harvested from `STATUS.md`, `docs/*` and `results/*`.
It exists so the project's state is queryable and so two documents that disagree get surfaced,
not silently picked between. Query it with `tools/kb.py` (stdlib only, no new dependencies).

```
python -m tools.kb list --tag identity
python -m tools.kb list --status retracted
python -m tools.kb search "counterpress"
python -m tools.kb show teammap-003
python -m tools.kb check
```

## Schema

Every claim is one JSON object with these fields (see `claims.json` for real examples):

| field | meaning |
|---|---|
| `id` | short, prefixed, stable (`teammap-003`, `b4-002`, ...). Never reused. |
| `statement` | one sentence. The claim, in plain language. |
| `value` / `unit` | the number, if there is one, and its unit. `null` if the claim is not numeric. |
| `status` | `confirmed` \| `retracted` \| `superseded` \| `pending` \| `refuted_externally` (see below) |
| `evidence` | list of `"path/to/file.md#Section heading"` — every claim must trace to a real file |
| `method` | one sentence on how it was measured |
| `n` | sample size, if applicable, else `null` |
| `caveats` | list of strings — known limits, confounds, honest scope |
| `date` | when it was established/last recomputed (`YYYY-MM-DD`) |
| `supersedes` / `superseded_by` | ids of the claim(s) this replaces / was replaced by |
| `depends_on` | ids of claims this one is built on (e.g. a corrected-mapping claim depending on the retraction that produced it) |
| `tags` | e.g. `identity`, `imputation`, `pressing`, `block-height`, `possession`, `methodology`, `storage`, `scoop`, `negative-result`, `pending` |

### Status values, precisely

- **confirmed** — currently believed true, with evidence traced to a real file.
- **retracted** — the project published this and later withdrew it. The retraction reason lives
  in the statement/caveats, and `superseded_by` names the replacement claim if one exists.
- **superseded** — not wrong, but replaced by a better-derived number (e.g. an in-sample figure
  superseded by a held-out refit). Softer than `retracted`.
- **pending** — measured but the validation gate hasn't run yet (e.g. block-height Gate 1), or a
  direction reported honestly as fragile/small-n.
- **refuted_externally** — the project tested its own candidate claim and the project's own
  measurement rejected it (e.g. "block compactness is a Man Utd fingerprint" — tested, not
  distinguishable from the league). This is *not* for claims refuted by an outside paper; use
  `retracted` + a caveat naming the outside source for that (see the Choi-scoop claims, tagged
  `scoop`), since those are about a claim's *novelty*, not its truth.

## How to add a claim

1. Find the real line/section in the source file. Do not transcribe a claim you cannot point to.
2. If the document states something vaguely (no precise number), record it as `status: "pending"`
   with a caveat saying so — never invent precision.
3. Pick an id: `<short-tag>-NNN`, next free number in that prefix.
4. Fill every field. `caveats` should carry the scope limits stated in the source (n, confounds,
   "directional not significant", etc.) — do not silently drop them.
5. If this claim relates to an existing one (extends it, contradicts it, replaces it), add the
   `depends_on` / `supersedes` link both ways where applicable.
6. Run `python -m tools.kb check` and read the new contradiction candidates it prints. Either the
   new claim needs a `supersedes`/`depends_on` link to an old one, or the flagged pair is a
   genuine open disagreement — in which case leave both claims in the graph (see next section).

## How to retract a claim

**A retraction must name its replacement.** Do not set `status: "retracted"` and stop.

1. Add a new claim for the corrected finding (or point `superseded_by` at an existing one).
2. Set the old claim's `status` to `"retracted"` (or `"superseded"` if it's a softer replacement,
   e.g. an in-sample number replaced by a held-out one).
3. Set the old claim's `superseded_by` to the new claim's id.
4. Set the new claim's `depends_on` (or, if it directly replaces the old one, no extra field is
   needed — `superseded_by` on the old claim is the link) to point back if the new number was
   computed *using* the corrected input from the old claim's retraction (e.g. `teammap-003`
   depends on `teammap-002`, the mapping-flip retraction that made the recompute possible).
5. Write the *reason* for the retraction into the new claim's `statement` or `caveats` — a bare
   "retracted" with no reason is not acceptable (see `teammap-003`, `event-001b` for the pattern).

If a document in the repo still asserts the old number in prose (a STATUS.md entry, an old
results/*.md), that is fine — STATUS.md is a historical log and is allowed to contain superseded
numbers with a `[CORRECTED ...]` annotation inline. The claims graph is what a query should trust;
the prose log is what happened, in order.

## Where two documents disagree

`tools.kb check` finds candidate contradictions by two heuristics — same tag + same unit with
values more than 15% apart, or same tag with high statement-text overlap — restricted to pairs of
`confirmed` claims with no existing supersession/dependency link between them. **It never
auto-resolves anything.** Every hit is printed for a human to look at. Two legitimate outcomes:

- The pair is not actually a contradiction (e.g. `cv-001` 26.4% vs `cv-003` 80.5% — different
  denominators, both correct, just not linked yet). Add a `depends_on` note explaining the
  relationship, or leave them if the relationship is "these measure different things and both
  numbers get quoted with their denominator stated" (the common case here).
- The pair is a genuine open disagreement between two sources. **Record BOTH claims with a
  contradiction flag rather than silently picking one** — add a caveat on each naming the other
  claim's id and the discrepancy. That surfacing is the entire point of this tool.

## What this is not

No web service, no database, no ORM. `claims.json` is read with `json.loads`; `tools/kb.py` is
~180 lines of stdlib argparse. If the graph ever needs relational queries beyond tag/status/text
filtering, `sqlite3` (also stdlib) is the next rung — not before it is actually needed.
