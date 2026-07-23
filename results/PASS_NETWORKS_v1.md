# Pass networks v1 (Phase-0 build A)

Directed passing networks from the validated BAS pass stream + event-ledger carrier attribution (`tools.event_ledger.build_ledger`, reused). Edges link consecutive attributed carriers within a possession; nodes are named players where identity resolves, else a per-team abstain bucket. Structural metrics use the **named-player subgraph only** (the bucket collapses who-to-whom structure).

## GATE-0 verdicts

- **Pass-total band:** 9/12 matches inside the Sofascore 0.97-1.09x band (op PASS vs attempted, complete halves).
- **Named-edge coverage:** named pass rate 0.0-3.9% across matches (the known 4-12% player floor). Named-player *edges* (both endpoints named, consecutive) are rarer still -- see per-match counts; the player-level network is volume-thin, structure-poor.
- **Man Utd identifiability:** 3 ManU identity matches; mean intra-match player pass-volume cosine = 0.21 (the named-edge network is empty -- see limits; structural identifiability is not computable at this coverage).

## Per-match networks + gate

### brighton_manutd

Coverage -- PASS events 981 | team-attributed 414 (42.2%) | named 15 (1.5%).
GATE-0 band -- op PASS 981 / truth 988 = 0.993 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 84 | 0 | 0 | - | - | - | - | - | - |
| Brighton | 93 | 0 | 0 | - | - | - | - | - | - |

### manutd_liverpool

Coverage -- PASS events 981 | team-attributed 355 (36.2%) | named 20 (2.0%).
GATE-0 band -- op PASS 981 / truth 971 = 1.01 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 75 | 0 | 0 | - | - | - | - | - | - |
| Liverpool | 67 | 2 | 1 | 0.00 | 1.00 | 0.00 | 0.00 | - | - |

### manutd_fulham

Coverage -- PASS events 945 | team-attributed 310 (32.8%) | named 0 (0.0%).
GATE-0 band -- op PASS 945 / truth 866 = 1.091 (complete halves ['h1', 'h2']) -> **FAIL** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 58 | 0 | 0 | - | - | - | - | - | - |
| Fulham | 42 | 0 | 0 | - | - | - | - | - | - |

### palace_manutd

Coverage -- PASS events 965 | team-attributed 249 (25.8%) | named 0 (0.0%).
GATE-0 band -- op PASS 965 / truth 952 = 1.014 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Crystal Palace | 12 | 0 | 0 | - | - | - | - | - | - |
| Man Utd | 65 | 0 | 0 | - | - | - | - | - | - |

### manutd_tottenham

Coverage -- PASS events 1079 | team-attributed 338 (31.3%) | named 42 (3.9%).
GATE-0 band -- op PASS 1079 / truth 1031 = 1.047 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 42 | 0 | 0 | - | - | - | - | - | - |
| Tottenham | 95 | 0 | 0 | - | - | - | - | - | - |

### southampton_manutd

Coverage -- PASS events 1087 | team-attributed 425 (39.1%) | named 0 (0.0%).
GATE-0 band -- op PASS 1087 / truth 1091 = 0.996 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Southampton | 39 | 0 | 0 | - | - | - | - | - | - |
| Man Utd | 69 | 0 | 0 | - | - | - | - | - | - |

### liverpool_manutd

Coverage -- PASS events 886 | team-attributed 262 (29.6%) | named 0 (0.0%).
GATE-0 band -- op PASS 886 / truth 821 = 1.079 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Liverpool | 58 | 0 | 0 | - | - | - | - | - | - |
| Man Utd | 31 | 0 | 0 | - | - | - | - | - | - |

### manutd_brighton

Coverage -- PASS events 960 | team-attributed 402 (41.9%) | named 0 (0.0%).
GATE-0 band -- op PASS 960 / truth 900 = 1.067 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 75 | 0 | 0 | - | - | - | - | - | - |
| Brighton | 82 | 0 | 0 | - | - | - | - | - | - |

### fulham_manutd

Coverage -- PASS events 1017 | team-attributed 445 (43.8%) | named 0 (0.0%).
GATE-0 band -- op PASS 1017 / truth 987 = 1.03 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Fulham | 83 | 0 | 0 | - | - | - | - | - | - |
| Man Utd | 89 | 0 | 0 | - | - | - | - | - | - |

### manutd_palace

Coverage -- PASS events 943 | team-attributed 340 (36.1%) | named 0 (0.0%).
GATE-0 band -- op PASS 943 / truth 851 = 1.108 (complete halves ['h1', 'h2']) -> **FAIL** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 74 | 0 | 0 | - | - | - | - | - | - |
| Crystal Palace | 26 | 0 | 0 | - | - | - | - | - | - |

### manutd_southampton

Coverage -- PASS events 1066 | team-attributed 381 (35.7%) | named 0 (0.0%).
GATE-0 band -- op PASS 1066 / truth 1000 = 1.066 (complete halves ['h1', 'h2']) -> **PASS** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 65 | 0 | 0 | - | - | - | - | - | - |
| Southampton | 51 | 0 | 0 | - | - | - | - | - | - |

### tottenham_manutd

Coverage -- PASS events 1036 | team-attributed 387 (37.4%) | named 0 (0.0%).
GATE-0 band -- op PASS 1036 / truth 933 = 1.11 (complete halves ['h1', 'h2']) -> **FAIL** (band 0.97-1.09x).

| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | ABA ratio | ABA z |
|------|-----------|-------------|-------------|-------|------|------|------|-----------|-------|
| Man Utd | 61 | 0 | 0 | - | - | - | - | - | - |
| Tottenham | 112 | 0 | 0 | - | - | - | - | - | - |

## Man Utd cross-match identifiability (player pass-volume cosine)

| match A | match B | cosine |
|---------|---------|--------|
| brighton_manutd | manutd_liverpool | 0.293 |
| brighton_manutd | manutd_tottenham | 0.23 |
| manutd_liverpool | manutd_tottenham | 0.106 |

Mean intra-ManU cosine: **0.21** over 3 networks.

## Honest limits

- **Player network is not viable at current identity coverage.** Named passes are 0-42 per match across both teams (2-4% of the pass stream); consecutive named->named edges are a handful, so centrality/motif numbers on the named subgraph are volume artifacts, not structure. What is real is per-player pass *volume* (the node strengths, echoing `results/PLAYER_LEDGER.md`), not who-passes-to-whom.
- **Edges skip unattributed touches.** `src -> dst` links the next attributed carrier in a possession, so an edge may span 1-2 unlabeled passes -- an over-connection bias, not a direct-pass guarantee.
- **Team attribution is the nearest-carrier heuristic** (`tools.event_ledger`); it abstains ~60% of passes (no tracked player on the ball at the kick). The abstain bucket carries most edge weight; the named subgraph is the residue.
- **Formation proxy is the usable player-level artifact** -- it averages every trusted frame of a named fragment, not just pass events, so it has real support; see the `formation` records in each `outputs/<id>/facts/pass_network.json`.
- **Identifiability is under-powered:** 3 identity matches, distinct opponents (n=1 each) -> opponent self-consistency untestable; only ManU intra-match consistency is reported.
