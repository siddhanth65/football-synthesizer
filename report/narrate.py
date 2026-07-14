"""LLM narration layer: turn the grounded evidence bundle into natural pundit prose.

The template report (``report.pundit``) is the deterministic, auditable floor. This layer wraps the SAME
grounded facts in an LLM prompt so the output reads like a person wrote it -- while the prompt forbids any
claim not backed by a provided number. The LLM narrates evidence; it never invents tactics or stats.

Runs the Anthropic API when ``ANTHROPIC_API_KEY`` is set (``pip install anthropic``); otherwise it writes
the ready-to-send prompt to ``results/narrate_<match>_prompt.txt`` so it can be pasted into any LLM. The
grounding contract is identical either way.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from report.pundit import load_bundle

MODEL = os.environ.get("PUNDIT_MODEL", "claude-sonnet-4-6")

SYSTEM = (
    "You are an experienced, sharp football pundit and tactical analyst. You write concise, flowing, "
    "confident prose — the kind heard on a top broadcast. STRICT RULE: every tactical claim must be "
    "grounded in a number from the EVIDENCE provided. Never invent statistics, events, goals, or tactics "
    "not present in the evidence. Name players naturally. If the evidence is thin on something, don't "
    "cover it. 3–5 short paragraphs, no headings, no bullet points."
)


def build_evidence(match: str) -> dict:
    """Assemble the grounded fact bundle (CV shape + roster + FIFA) for one France match."""
    b = load_bundle(match)
    m, fifa, cv = b["roster"], b["fifa"], b["cv"]
    fi, oi = b["france_idx"], b["opp_idx"]

    def ph(k, idx):
        v = fifa.get("phases", {}).get(k)
        return v[idx] if isinstance(v, list) else None

    def ks(k, idx):
        v = fifa.get("key_stats", {}).get(k)
        return v[idx] if isinstance(v, list) else None

    ev = {
        "match": f"France {m['score'][0]}-{m['score'][1]} {m['opponent']}",
        "opponent": m["opponent"], "france_formation": m["formation"],
        "france_key_attackers": m["key_attackers"], "france_xi_by_role": m["xi"],
        "cv_shape_france": None if not cv else {
            "defensive_line_height_m": round(cv["def_line_height"]["mean"], 1),
            "build_up_height_m": round(cv["buildup_height"]["mean"], 1),
            "attacking_third_share": round(cv["attacking_third_share"]["mean"], 2),
            "wing_share": round(cv["wing_share"]["mean"], 2),
        },
        "fifa_france_phases_pct": {"build_up_unopposed": ph("build_up_unopposed", fi),
                                   "progression": ph("progression", fi), "final_third": ph("final_third", fi),
                                   "mid_block": ph("mid_block", fi), "low_block": ph("low_block", fi),
                                   "high_press": ph("high_press", fi)},
        "fifa_opponent_defensive_pct": {"low_block": ph("low_block", oi), "mid_block": ph("mid_block", oi)},
        "fifa_france_stats": {"xg": ks("xg", fi), "opponent_xg": ks("xg", oi),
                              "completed_line_breaks": ks("completed_line_breaks", fi),
                              "receptions_final_third": ks("receptions_final_third", fi),
                              "possession_pct": ks("possession_pct", fi),
                              "forced_turnovers": ks("forced_turnovers", fi)},
        "cross_match_note": ("France commits further forward against deeper-sitting opponents (measured "
                             "across its group-stage matches) — an opponent-conditioned tendency."),
    }
    facts = b.get("facts")
    if facts:  # the versioned fact store (report.facts): ball-tracked CV metrics for BOTH teams
        fc = facts["cv"]
        ev["cv_ball_tracked"] = {
            "metrics_version": facts["metrics_version"],
            "counterpress": fc.get("transitions"),
            "pressing_ppda_proxy": fc.get("passing"),
            "ball_xt_progression": fc.get("ball_xt"),
            "velocity_synchrony": fc.get("style", {}).get("velocity_synchrony"),
            "space_control": fc.get("space"),
            "phase_time_pct": fc.get("phases_pct"),
        }
    return ev


def build_prompt(match: str) -> str:
    """The full user prompt: instructions + the grounded evidence JSON."""
    ev = build_evidence(match)
    return ("Write a grounded pundit analysis of this France match. EVIDENCE (the ONLY facts you may "
            f"use, all measured — CV = our computer-vision tracking, FIFA = official match report):\n\n"
            f"{json.dumps(ev, indent=2)}\n\n"
            "Weave the numbers in naturally (e.g. 'behind a high line, roughly 52 metres up the pitch'). "
            "Lead with France's identity and their attackers' threat against the opponent's block, then "
            "how they defend, then a one-line honest note on what's measured vs indicative.")


def _http_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    """POST JSON and return the parsed response (stdlib only, no SDK needed)."""
    import json as _json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, data=_json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return _json.loads(r.read().decode())


def _call_ollama(prompt: str, model: str, host: str) -> str:
    """Local Ollama chat (free, no key; needs `ollama serve` + `ollama pull <model>`)."""
    r = _http_json(f"{host.rstrip('/')}/api/chat", {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    return r["message"]["content"]


def _call_openai_compat(prompt: str, model: str, base_url: str, key: str) -> str:
    """Any OpenAI-compatible chat endpoint (Groq / Gemini-openai / OpenRouter / local)."""
    r = _http_json(f"{base_url.rstrip('/')}/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": 1100}, headers={"Authorization": f"Bearer {key}"})
    return r["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, model: str, key: str) -> str:
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(model=model, max_tokens=1100, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def _guard(text: str, match: str) -> str:
    """Post-generation numeric guardrail: flag any number the LLM emitted that the fact store can't back.

    Every figure in the narration is checked against the match fact store; ungrounded numbers are
    ``[?…]``-annotated inline and summarised in a footer. This is what makes the narration *grounded* — a
    hallucinated stat cannot silently reach the reader (see ``report.guardrail`` / ``tools.guardrail_eval``).
    """
    from report.facts import load_facts  # noqa: PLC0415
    from report.guardrail import annotate, audit  # noqa: PLC0415

    facts = load_facts(match)
    if not facts:
        return text
    a = audit(text, facts)
    if a["n_unbacked"] == 0:
        return f"{text}\n\n---\n*Guardrail: all {a['n_numbers']} numbers grounded in the fact store.*"
    flags = ", ".join(u["text"] for u in a["unbacked"])
    return (f"{annotate(text, facts)}\n\n---\n*Guardrail: {a['n_unbacked']}/{a['n_numbers']} numbers "
            f"UNGROUNDED and [?]-flagged above ({flags}) -- not in the fact store, likely LLM fabrications.*")


def narrate(match: str, *, backend: str = "auto", model: str | None = None,
            base_url: str | None = None, guard: bool = True) -> str:
    """Narrate via the chosen backend; ``auto`` tries what's configured, else writes the prompt.

    Backends: ``ollama`` (local, free) | ``openai`` (OpenAI-compatible: Groq/Gemini/OpenRouter, needs
    ``LLM_API_KEY`` + ``base_url``) | ``anthropic`` (needs ``ANTHROPIC_API_KEY``) | ``auto`` (ollama if up,
    then anthropic, then openai, else prompt file). With ``guard`` (default), the numeric guardrail flags
    any un-grounded figure in the generated prose before returning it.
    """
    prompt = build_prompt(match)
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    raw = None
    try:
        if backend in ("ollama",) or (backend == "auto" and _ollama_up(ollama_host)):
            raw = _call_ollama(prompt, model or os.environ.get("OLLAMA_MODEL", "llama3.2"), ollama_host)
        elif backend == "anthropic" or (backend == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
            raw = _call_anthropic(prompt, model or MODEL, os.environ["ANTHROPIC_API_KEY"])
        elif backend == "openai" or (backend == "auto" and os.environ.get("LLM_API_KEY")):
            raw = _call_openai_compat(prompt, model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant"),
                                      base_url or os.environ.get("LLM_BASE_URL", ""),
                                      os.environ["LLM_API_KEY"])
    except Exception as e:  # noqa: BLE001
        return f"[LLM call failed ({backend}): {e}]\n\nPrompt was written; run a backend to narrate."
    if raw is not None:
        return _guard(raw, match) if guard else raw
    out = Path(f"results/narrate_{match}_prompt.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + prompt, encoding="utf-8")
    return (f"[No LLM backend available — wrote the grounded prompt to {out}.\n"
            f"  FREE options:\n"
            f"   • Local (recommended): install Ollama, `ollama pull llama3.2`, then re-run — auto-detected.\n"
            f"   • Groq (free key): --backend openai --base-url https://api.groq.com/openai/v1 "
            f"(set LLM_API_KEY, LLM_MODEL=llama-3.1-8b-instant)\n"
            f"   • Paste {out} into any chatbot.\n"
            f"  The grounding contract lives in the SYSTEM prompt, so any model stays honest.]")


def _ollama_up(host: str) -> bool:
    import urllib.request  # noqa: PLC0415

    try:
        urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=1)  # noqa: S310
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="france_iraq")
    ap.add_argument("--backend", default="auto", choices=["auto", "ollama", "openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None, help="for --backend openai (Groq/Gemini/OpenRouter)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = narrate(args.match, backend=args.backend, model=args.model, base_url=args.base_url)
    print(text)
    if args.out and not text.startswith("[No LLM"):
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
