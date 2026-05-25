"""
Development evaluator for the Agentic Movie Recommender.

Required for normal judging:
  OLLAMA_API_KEY - used by llm.py and by the same-model judge path

Optional for cross-model judging:
  ANTHROPIC_API_KEY - used only when --external-judge is passed
  ANTHROPIC_JUDGE_MODEL - optional override for the Claude judge model

This script is not part of the production recommendation path. It exists to
document and automate the Build-Measure-Learn loop for the README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Callable

import ollama

from baseline import baseline_get_recommendation
from llm import MODEL, TOP_MOVIES, _history_exclude_ids, get_recommendation


ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_CASES = ROOT / "Eval packages" / "eval_cases.json"
CACHE_PATH = ROOT / ".eval_cache.json"
DEFAULT_REPORT_PATH = ROOT / "eval_report.md"
DEFAULT_SUMMARY_PATH = ROOT / "eval_summary.json"
TIMEOUT_SECONDS = 20
DEFAULT_ANTHROPIC_JUDGE_MODEL = os.getenv(
    "ANTHROPIC_JUDGE_MODEL", "claude-3-5-haiku-latest"
)


FALLBACK_CASES = [
    {
        "label": "superhero action",
        "tier": "easy",
        "preferences": "I love action movies with superheroes and big emotional stakes.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "feel good comedy",
        "tier": "medium",
        "preferences": "I want something funny, warm, and feel-good.",
        "history": ["The Dark Knight Rises"],
        "history_ids": [49026],
    },
    {
        "label": "recent foreign subtitles",
        "tier": "hard",
        "preferences": "Recent foreign film with subtitles",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "missing decade thriller",
        "tier": "hard",
        "preferences": "A thriller from the 90s",
        "history": [],
        "history_ids": [],
    },
]


def _clean_text(value: str, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped or text[:limit]


def _hash_key(*parts) -> str:
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clamp_score(value) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 3
    return max(1, min(5, score))


def _safe_json_parse(text: str, fallback: dict | None = None) -> dict:
    fallback = fallback or {}
    if not text:
        return fallback
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else fallback
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else fallback
            except json.JSONDecodeError:
                return fallback
    return fallback


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _norm_title(title: str) -> str:
    return " ".join(str(title).lower().replace(":", " ").split())


def _title_to_id(title: str) -> int | None:
    wanted = _norm_title(title)
    for row in TOP_MOVIES.itertuples():
        if _norm_title(row.title) == wanted:
            return int(row.tmdb_id)
    return None


def _case_with_ids(case: dict) -> dict:
    out = dict(case)
    history = list(out.get("history", []))
    history_ids = [int(tid) for tid in out.get("history_ids", [])]
    if not history_ids:
        for title in history:
            tid = _title_to_id(title)
            if tid is not None:
                history_ids.append(tid)
    out["history"] = history
    out["history_ids"] = history_ids
    return out


def _load_cases(path: str | None, max_cases: int = 0) -> list[dict]:
    case_path = Path(path) if path else DEFAULT_EVAL_CASES
    if case_path.exists():
        cases = json.loads(case_path.read_text(encoding="utf-8"))
    else:
        cases = FALLBACK_CASES
    cases = [_case_with_ids(case) for case in cases]
    if max_cases and max_cases > 0:
        cases = cases[:max_cases]
    return cases


def _movie_meta(tmdb_id: int) -> dict:
    rows = TOP_MOVIES[TOP_MOVIES["tmdb_id"].astype(int) == int(tmdb_id)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "title": str(row["title"]),
        "year": int(row["year"]),
        "genres": str(row["genres"]),
        "overview": str(row["overview"]),
        "keywords": str(row.get("keywords", "")),
        "original_language": str(row.get("original_language", "")),
        "top_cast": str(row.get("top_cast", "")),
    }


def _build_ollama_client(timeout: float) -> ollama.Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set.")
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )


def _call_ollama_json(
    client: ollama.Client,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 500,
) -> dict:
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format="json",
        think=False,
        options={"temperature": 0, "num_predict": max_tokens},
    )
    return _safe_json_parse(response.message.content)


def _call_anthropic_json(prompt: str, model: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0,
        system=(
            "You are evaluating AI-generated movie recommendations. Return only "
            "valid JSON matching the requested schema. Do not include markdown."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        getattr(block, "text", "")
        for block in message.content
        if getattr(block, "type", "") == "text" or hasattr(block, "text")
    )
    return _safe_json_parse(text)


SCORING_PROMPT = """
You are a skeptical, discerning user evaluating a movie recommendation.
Be tough but fair. Generic praise should score 2-3; specific, grounded,
preference-anchored work can score 5.

Case label: {label}
Tier: {tier}
User preferences: {preferences}
Watch history: {history}
History IDs to avoid: {history_ids}

Recommended movie metadata:
{movie_json}

Recommendation JSON:
{rec_json}

Score these dimensions from 1 to 5:
- relevance: how well the movie matches the stated preference
- novelty: avoids watched movies and uses history in a useful way
- pitch: persuasive, concrete, and likely to make a classmate want to watch
- accuracy: description is supported by the movie metadata

Return ONLY JSON:
{{"relevance": <1-5>, "novelty": <1-5>, "pitch": <1-5>, "accuracy": <1-5>, "reason": "<one short reason>"}}
""".strip()


def _judge_scores(
    client: ollama.Client,
    args: argparse.Namespace,
    case: dict,
    rec: dict,
    meta: dict,
    cache: dict,
) -> dict:
    provider = "anthropic" if args.external_judge else "ollama"
    model = args.anthropic_model if args.external_judge else args.judge_model
    cache_key = _hash_key("score", provider, model, case.get("label"), rec, meta)
    if not args.no_cache and cache_key in cache:
        return cache[cache_key]

    prompt = SCORING_PROMPT.format(
        label=case.get("label", ""),
        tier=case.get("tier", ""),
        preferences=case.get("preferences", ""),
        history=case.get("history", []),
        history_ids=case.get("history_ids", []),
        movie_json=json.dumps(meta, ensure_ascii=True),
        rec_json=json.dumps(rec, ensure_ascii=True),
    )
    system = "Return strict JSON only. Scores must be integers from 1 to 5."

    if args.external_judge:
        parsed = _call_anthropic_json(prompt, args.anthropic_model)
    else:
        parsed = _call_ollama_json(client, args.judge_model, system, prompt, max_tokens=260)

    result = {
        "relevance": _clamp_score(parsed.get("relevance")),
        "novelty": _clamp_score(parsed.get("novelty")),
        "pitch": _clamp_score(parsed.get("pitch")),
        "accuracy": _clamp_score(parsed.get("accuracy")),
        "reason": _clean_text(parsed.get("reason", ""), 300),
    }
    cache[cache_key] = result
    return result


AB_PROMPT = """
You are a skeptical, discerning user choosing which movie recommendation you
would rather receive. Judge fit to the user's request plus persuasiveness.
Avoid position bias: A and B were randomized.

User preferences: {preferences}
Watch history: {history}
History IDs to avoid: {history_ids}

Recommendation A:
{a_json}

Recommendation B:
{b_json}

Return ONLY JSON:
{{"winner": "A"|"B"|"tie", "reason": "<one short reason>"}}
""".strip()


def _assign_ab(
    rng: random.Random, ours: dict, baseline: dict
) -> tuple[dict[str, str], dict, dict]:
    pair = [("ours", ours), ("baseline", baseline)]
    rng.shuffle(pair)
    assignment = {"A": pair[0][0], "B": pair[1][0]}
    return assignment, pair[0][1], pair[1][1]


def _judge_ab(
    client: ollama.Client,
    args: argparse.Namespace,
    case: dict,
    ours: dict,
    baseline: dict,
    assignment: dict[str, str],
    a_rec: dict,
    b_rec: dict,
    cache: dict,
) -> dict:
    provider = "anthropic" if args.external_judge else "ollama"
    model = args.anthropic_model if args.external_judge else args.judge_model
    cache_key = _hash_key("ab", provider, model, case.get("label"), assignment, a_rec, b_rec)
    if not args.no_cache and cache_key in cache:
        cached = dict(cache[cache_key])
        cached["assignment"] = assignment
        return cached

    prompt = AB_PROMPT.format(
        preferences=case.get("preferences", ""),
        history=case.get("history", []),
        history_ids=case.get("history_ids", []),
        a_json=json.dumps(a_rec, ensure_ascii=True),
        b_json=json.dumps(b_rec, ensure_ascii=True),
    )
    system = "Return strict JSON only. Winner must be A, B, or tie."

    if args.external_judge:
        parsed = _call_anthropic_json(prompt, args.anthropic_model)
    else:
        parsed = _call_ollama_json(client, args.judge_model, system, prompt, max_tokens=220)

    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"

    result = {
        "winner": winner,
        "winner_source": assignment.get(winner, "tie") if winner != "tie" else "tie",
        "assignment": assignment,
        "reason": _clean_text(parsed.get("reason", ""), 300),
    }
    cache[cache_key] = {k: v for k, v in result.items() if k != "assignment"}
    return result


def _validate_result(
    result: dict,
    history: list[str],
    history_ids: list[int],
    valid_ids: set[int],
    latency_s: float,
) -> tuple[dict, list[str]]:
    issues: list[str] = []
    checks = {
        "dict": isinstance(result, dict),
        "has_required_keys": False,
        "valid_id": False,
        "not_seen": False,
        "description_length": False,
        "under_timeout": latency_s < TIMEOUT_SECONDS,
        "references_history_when_present": True,
    }

    if not isinstance(result, dict):
        issues.append("result is not a dict")
        return checks, issues

    checks["has_required_keys"] = set(result.keys()) == {"tmdb_id", "description"}
    if not checks["has_required_keys"]:
        issues.append("result keys are not exactly tmdb_id and description")

    try:
        tmdb_id = int(result.get("tmdb_id"))
    except (TypeError, ValueError):
        tmdb_id = -1
        issues.append("tmdb_id is not an int")

    checks["valid_id"] = tmdb_id in valid_ids
    if not checks["valid_id"]:
        issues.append("tmdb_id is not in CSV")

    excluded = _history_exclude_ids(history, history_ids)
    checks["not_seen"] = tmdb_id not in excluded
    if not checks["not_seen"]:
        issues.append("tmdb_id is in watch history")

    description = result.get("description", "")
    checks["description_length"] = isinstance(description, str) and len(description) <= 500
    if not checks["description_length"]:
        issues.append("description is not a <=500 char string")

    if not checks["under_timeout"]:
        issues.append("latency >= 20s")

    if history and isinstance(description, str):
        checks["references_history_when_present"] = any(
            title.lower() in description.lower() for title in history
        )
        if not checks["references_history_when_present"]:
            issues.append("description does not mention any watched title")

    return checks, issues


def _safe_call(
    fn: Callable[[str, list[str], list[int]], dict],
    case: dict,
) -> tuple[dict, float]:
    start = time.perf_counter()
    try:
        rec = fn(
            case.get("preferences", ""),
            case.get("history", []),
            case.get("history_ids", []),
        )
    except Exception as exc:
        rec = {"tmdb_id": -1, "description": f"error: {exc.__class__.__name__}"}
    return rec, time.perf_counter() - start


def _extract_prompt_source() -> str:
    source = (ROOT / "llm.py").read_text(encoding="utf-8")
    start = source.find("def _build_messages(")
    end = source.find("\ndef _extract_json", start)
    return source[start:end] if start >= 0 and end > start else source[:6000]


def _optimizer_suggestions(
    client: ollama.Client,
    optimizer_model: str,
    report: dict,
    max_examples: int = 6,
) -> dict:
    weak_examples = sorted(
        report["results"],
        key=lambda item: (
            item.get("judge", {}).get("relevance", 5)
            + item.get("judge", {}).get("pitch", 5)
            + item.get("judge", {}).get("accuracy", 5),
            item.get("elapsed_seconds", 0),
        ),
    )[:max_examples]
    payload = {
        "summary": report["summary"],
        "weak_examples": weak_examples,
        "current_prompt_code": _extract_prompt_source(),
    }
    system = (
        "You are a prompt optimizer for a Python movie recommendation agent. "
        "Propose safe, minimal prompt or retrieval changes only. Do not suggest "
        "changing the model name, function signature, return schema, API key "
        "handling, ID validation, history filtering, or timeout guards."
    )
    user = f"""
Review these judge results and the current prompt-building code from llm.py.

Return JSON only with:
- diagnosis: brief explanation of the main failure pattern.
- prompt_patch: concrete replacement wording or bullets to add.
- retrieval_patch: concrete safe candidate-filtering idea, if any.
- risk_checks: disqualification risks the change must preserve.
- expected_gain: one sentence.

Evaluation payload:
{json.dumps(payload, ensure_ascii=True)}
""".strip()
    return _call_ollama_json(client, optimizer_model, system, user, max_tokens=800)


def _write_markdown_report(report: dict, path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Generator: `{summary['generator_model']}`",
        f"- Judge: `{summary['judge_model']}`",
        f"- External judge: `{summary['external_judge']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Validity: `{summary['validity_passed']}/{summary['case_count']}` (`{summary['validity_rate']:.1%}`)",
        f"- Mean relevance: `{summary['mean_relevance']}`",
        f"- Mean novelty: `{summary['mean_novelty']}`",
        f"- Mean pitch: `{summary['mean_pitch']}`",
        f"- Mean accuracy: `{summary['mean_accuracy']}`",
        f"- A/B wins/losses/ties: `{summary['ab_ours_wins']}/{summary['ab_baseline_wins']}/{summary['ab_ties']}`",
        f"- A/B win rate: `{summary['ab_win_rate']:.1%}`",
        "",
        "## Per-Case Results",
        "",
        "| # | Case | Tier | Valid | Time | Movie | Scores | A/B | Notes |",
        "|---:|---|---|---:|---:|---|---|---|---|",
    ]
    for idx, row in enumerate(report["results"], start=1):
        movie = row.get("movie", {}).get("title") or row.get("recommendation", {}).get("tmdb_id")
        judge = row.get("judge", {})
        scores = "/".join(
            str(judge.get(key, "-")) for key in ("relevance", "novelty", "pitch", "accuracy")
        )
        notes = "; ".join(row.get("issues", [])) or "OK"
        lines.append(
            f"| {idx} | {row['case']} | {row.get('tier', '')} | {int(row['valid'])} | "
            f"{row['elapsed_seconds']} | {movie} | {scores} | "
            f"{row.get('ab', {}).get('winner_source', '-')} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_suggestions(path: Path, suggestions: dict, summary: dict) -> None:
    lines = [
        "# Prompt Optimizer Suggestions",
        "",
        "## Evaluation Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Suggested Update",
        "",
        "```json",
        json.dumps(suggestions, indent=2),
        "```",
        "",
        "Review these manually before editing llm.py. Keep the model, signature, output schema, ID validation, history filtering, and timeout guards unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict:
    cases = _load_cases(args.cases_file, args.max_cases)
    cache = _load_cache()
    client = _build_ollama_client(args.timeout)
    valid_ids = set(TOP_MOVIES["tmdb_id"].astype(int))
    ab_rng = random.Random(args.seed)
    results = []

    for idx, case in enumerate(cases, start=1):
        ours, elapsed = _safe_call(get_recommendation, case)
        baseline, baseline_elapsed = _safe_call(baseline_get_recommendation, case)

        hard_checks, issues = _validate_result(
            ours,
            case.get("history", []),
            case.get("history_ids", []),
            valid_ids,
            elapsed,
        )
        valid = all(
            hard_checks[key]
            for key in (
                "dict",
                "has_required_keys",
                "valid_id",
                "not_seen",
                "description_length",
                "under_timeout",
                "references_history_when_present",
            )
        )
        tmdb_id = int(ours.get("tmdb_id", -1)) if isinstance(ours, dict) else -1
        meta = _movie_meta(tmdb_id)

        if args.skip_judge:
            judge = {}
            ab = {}
        else:
            # Invalid cases get floor scores rather than being excluded — they
            # should drag the mean down, not silently disappear from the average.
            if valid:
                judge = _judge_scores(client, args, case, ours, meta, cache)
            else:
                judge = {
                    "relevance": 1,
                    "novelty": 1,
                    "pitch": 1,
                    "accuracy": 1,
                    "reason": "failed hard validity checks: " + "; ".join(issues),
                }
            # A/B runs regardless of validity — baseline may still win, which is signal.
            assignment, a_rec, b_rec = _assign_ab(ab_rng, ours, baseline)
            ab = _judge_ab(client, args, case, ours, baseline, assignment, a_rec, b_rec, cache)

        row = {
            "case": case.get("label", f"case_{idx}"),
            "tier": case.get("tier", ""),
            "preferences": case.get("preferences", ""),
            "elapsed_seconds": round(elapsed, 3),
            "baseline_elapsed_seconds": round(baseline_elapsed, 3),
            "valid": valid,
            "issues": issues,
            "hard_checks": hard_checks,
            "recommendation": ours,
            "baseline": baseline,
            "movie": meta,
            "judge": judge,
            "ab": ab,
        }
        results.append(row)
        print(
            f"[{idx}/{len(cases)}] {row['case']}: "
            f"valid={valid} rec={ours} judge={judge} ab={ab.get('winner_source', '-')}"
        )

    _save_cache(cache)

    scored = [r["judge"] for r in results if r.get("judge")]
    ab_results = [r["ab"] for r in results if r.get("ab")]
    ab_ours_wins = sum(1 for item in ab_results if item.get("winner_source") == "ours")
    ab_baseline_wins = sum(1 for item in ab_results if item.get("winner_source") == "baseline")
    ab_ties = sum(1 for item in ab_results if item.get("winner_source") == "tie")
    compared = ab_ours_wins + ab_baseline_wins + ab_ties

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": MODEL,
        "judge_model": args.anthropic_model if args.external_judge else args.judge_model,
        "external_judge": bool(args.external_judge),
        "case_count": len(results),
        "validity_passed": sum(1 for r in results if r["valid"]),
        "validity_rate": sum(1 for r in results if r["valid"]) / max(len(results), 1),
        "invalid_id_rate": sum(not r["hard_checks"].get("valid_id", False) for r in results) / max(len(results), 1),
        "repeat_rate": sum(not r["hard_checks"].get("not_seen", False) for r in results) / max(len(results), 1),
        "timeout_rate": sum(not r["hard_checks"].get("under_timeout", False) for r in results) / max(len(results), 1),
        "mean_relevance": round(statistics.mean(j["relevance"] for j in scored), 3) if scored else None,
        "mean_novelty": round(statistics.mean(j["novelty"] for j in scored), 3) if scored else None,
        "mean_pitch": round(statistics.mean(j["pitch"] for j in scored), 3) if scored else None,
        "mean_accuracy": round(statistics.mean(j["accuracy"] for j in scored), 3) if scored else None,
        "ab_ours_wins": ab_ours_wins,
        "ab_baseline_wins": ab_baseline_wins,
        "ab_ties": ab_ties,
        "ab_win_rate": round(ab_ours_wins / max(compared, 1), 3),
    }
    report = {"summary": summary, "results": results}

    if args.optimize_prompt and not args.skip_judge:
        suggestions = _optimizer_suggestions(client, args.optimizer_model, report)
        report["optimizer_suggestions"] = suggestions
        _write_suggestions(Path(args.suggestions_output), suggestions, summary)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.report_output:
        _write_markdown_report(report, Path(args.report_output))

    return report


def run_dry_run_ab(iterations: int, cases_file: str | None, seed: int) -> dict:
    cases = _load_cases(cases_file, 0)
    rng = random.Random(seed)
    counts = {"ours_as_a": 0, "ours_as_b": 0, "baseline_as_a": 0, "baseline_as_b": 0}
    for _ in range(max(iterations, 0)):
        for _case in cases:
            assignment, _a, _b = _assign_ab(rng, {"tmdb_id": 1}, {"tmdb_id": 2})
            counts["ours_as_a" if assignment["A"] == "ours" else "baseline_as_a"] += 1
            counts["ours_as_b" if assignment["B"] == "ours" else "baseline_as_b"] += 1
    total = max(iterations, 0) * len(cases)
    return {
        "iterations": max(iterations, 0),
        "case_count": len(cases),
        "total_assignments": total,
        **counts,
        "ours_as_a_rate": round(counts["ours_as_a"] / max(total, 1), 4),
        "ours_as_b_rate": round(counts["ours_as_b"] / max(total, 1), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation.")
    parser.add_argument("--cases-file", help="JSON case file. Defaults to Eval packages/eval_cases.json.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases.")
    parser.add_argument("--judge-model", default=MODEL, help="Ollama judge model for same-model sanity checks.")
    parser.add_argument("--optimizer-model", default=MODEL, help="Ollama model for prompt optimizer suggestions.")
    parser.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_JUDGE_MODEL, help="Anthropic judge model.")
    parser.add_argument("--external-judge", action="store_true", help="Use Anthropic as cross-model judge.")
    parser.add_argument("--skip-judge", action="store_true", help="Only run hard validity checks.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached judge results.")
    parser.add_argument("--seed", type=int, default=42, help="A/B randomization seed.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Ollama client timeout for evaluation calls.")
    parser.add_argument("--output", default=str(ROOT / "eval_report.json"), help="Full JSON report path.")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH), help="Summary JSON path.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_PATH), help="Markdown report path.")
    parser.add_argument("--optimize-prompt", action="store_true", help="Ask an optimizer LLM for improvement ideas.")
    parser.add_argument("--suggestions-output", default=str(ROOT / "prompt_suggestions.md"))
    parser.add_argument("--dry-run-ab", type=int, default=0, help="Check A/B label randomization without LLM calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run_ab > 0:
        result = run_dry_run_ab(args.dry_run_ab, args.cases_file, args.seed)
        print("=== Dry-Run A/B Assignment ===")
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    report = run_evaluation(args)
    print("\n=== Evaluation Summary ===")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
