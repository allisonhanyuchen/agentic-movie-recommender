# Agentic Movie Recommender

Collaborators: Celina Cao, Venssa Liu, Vansh Kharyal

A movie recommendation agent built around `gemma4:31b-cloud`. Given a user's preferences and watch history, `get_recommendation()` returns a single persuasive pitch for a movie drawn from the TMDB Top 1000 corpus.

Under the hood, the agent combines a local TF-IDF retrieval pass, a content-filtering tool that scores candidates against the prompt, and optional TMDB API enrichment before a final generation step. When TMDB is unavailable — no key, timeout, rate limit — the system degrades cleanly to the local CSV path and still returns a valid in-dataset movie.

## Quick Start

**Requires Python 3.10 or newer** (the code uses `dict | None` union syntax). macOS ships with Python 3.9 by default — if `python3 --version` reports 3.9.x, install a newer interpreter (`brew install python@3.12`) and use `python3.12` instead of `python3` in the first command below.

macOS / Linux (bash/zsh):

```bash
python3 -m venv .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# The Ollama key is the only one you must export yourself.
export OLLAMA_API_KEY="your_key_here"

python test.py
```

The bundled `.env` holds non-Ollama keys (`TMDB_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_JUDGE_MODEL`). `llm.py` auto-loads it via `python-dotenv` at import time, so you don't need to export those manually. To run without the bundled keys, delete `.env` or overwrite it with your own — the recommender falls back gracefully.

Windows PowerShell equivalents: `.venv\Scripts\Activate.ps1`, `$env:OLLAMA_API_KEY="..."`.

For an interactive run:

```bash
python llm.py --preferences "I want a tense sci-fi thriller with big ideas" --history "Inception"
```

## The `get_recommendation` Contract

`llm.py` exposes:

```python
def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    ...
```

It returns:

```python
{"tmdb_id": <int>, "description": "<=500 character pitch"}
```

## Design Guardrails

A few rules are non-negotiable — the model name, the 500-character cap, the 20-second latency ceiling, and the requirement that the returned `tmdb_id` actually exists in the corpus. Each row below maps a rule to the specific mechanism in `llm.py` that enforces it.

| Rule | Guard |
|---|---|
| Uses `gemma4:31b-cloud` | `MODEL` is fixed at the top of `llm.py` and never overridden. |
| No hardcoded API keys | `OLLAMA_API_KEY`, `TMDB_API_KEY`, and `ANTHROPIC_API_KEY` are read only from environment variables (directly or via `.env`). |
| Valid TMDB ID only | The final ID is checked against `VALID_IDS`, a frozen set built from the movie CSV. |
| No watched movie | Both `history_ids` and normalized history titles are excluded before retrieval and after generation. |
| Response under 20 seconds | The LLM call uses a single request with a 12-second client timeout; optional TMDB calls use a 2-second timeout and failures fall back to a deterministic local choice. |
| Description length | Output text is normalized and capped at 500 characters. |
| No team leak | The prompt forbids any class/team/hidden-rule disclosure, and suspicious private-context wording triggers a local fallback pitch. |

## Architecture

```text
preferences + watch history
        |
        v
title/id history filter
        |
        v
local RAG retrieval over the full movie corpus
        |
        v
filter_movies tool: genre + keyword + mood enrichment
        |
        v
rank top candidates by preference fit and quality
        |
        v
optional TMDB API actor/movie enrichment, intersected with local valid IDs
        |
        v
gemma4:31b-cloud writes one persuasive JSON recommendation
        |
        v
validate id/history/length, otherwise deterministic fallback
```

## Creativity Strategy

Two local tools run before the LLM writes the final answer.

`_rag_retrieve()` is a retrieval-augmented generation step over every movie synopsis, genre, keyword, tagline, director, and cast field. It uses TF-IDF vector similarity when scikit-learn is available and falls back to pure keyword-plus-quality scoring when it's not.

`_tool_filter_movies()` is a content-filtering tool. It extracts genre, mood, theme, actor, and quality signals from the user request, searches the full movie table, and ranks matches by preference fit plus rating, vote count, and popularity confidence.

When `TMDB_API_KEY` is set, the agent can call TMDB's person and movie endpoints to verify actor filmographies and enrich the selected movie's metadata. API results never escape the local candidate list: every returned `tmdb_id` must still exist in `tmdb_top1000_movies.csv`. That constraint lets the agent be creative with tool use without risking a hallucinated or out-of-list recommendation.

The prompt itself uses a compact internal selection rubric — preference fit, novelty against history, evidence from metadata, and pitch quality — plus a few pitch-style examples, all while requiring JSON-only output so the grader can parse a single line.

## Evaluation Strategy

`evaluate.py` drives the build-measure-learn loop we used to tune the recommender. Movie pitch quality is inherently subjective, so hard validity checks alone can't tell us whether a recommendation is actually *persuasive* — we lean on LLM-as-a-judge scoring in addition to the validity gates.

The default case set lives in `Eval packages/eval_cases.json`. It covers the real failure modes we saw during iteration: single-genre matching, rom-com disambiguation, funny horror, French grief drama, recent foreign/subtitled films, and prompts with missing-data constraints.

Run a local hard-check smoke test:

```bash
export OLLAMA_API_KEY="your_key_here"
python evaluate.py --skip-judge --max-cases 4
```

Run the same-model judge, using `gemma4:31b-cloud` as a sanity check:

```bash
python evaluate.py --max-cases 8 --output eval_report.json --report-output eval_report.md
```

Run the stronger cross-model judge with Anthropic Claude:

```bash
# ANTHROPIC_API_KEY is read from the bundled .env automatically.
python evaluate.py --external-judge --max-cases 8 --anthropic-model claude-haiku-4-5-20251001
```

The Anthropic key is used only by `evaluate.py` for LLM-as-a-judge scoring. `llm.py` never reads it during recommendation generation, so anyone running the public `get_recommendation()` API doesn't need an Anthropic key at all.

The evaluator has three layers:

| Layer | What it checks |
|---|---|
| Hard disqualification checks | Exact output keys, valid CSV `tmdb_id`, not in watch history, description <=500 chars, latency <20 seconds. |
| LLM-as-a-judge scoring | Scores relevance, novelty, pitch quality, and metadata accuracy from 1-5. |
| Randomized A/B judging | Compares our recommender against `baseline.py` with randomized A/B labels, then records wins, losses, ties, and judge reasoning. |

| Metric | Meaning |
|---|---|
| `relevance` | Does the movie match the stated preference? |
| `novelty` | Does it avoid repeating or over-copying watch history? |
| `pitch` | Is the description persuasive and specific? |
| `accuracy` | Does the pitch stay consistent with the movie metadata? |

We weight the A/B comparison heavily because competition day is also pairwise — classmates pick one of two recommendations. The evaluator uses a seeded randomizer for A/B placement and ships a dry-run verifier to catch position bias:

```bash
python evaluate.py --dry-run-ab 100
```

### Results

Target thresholds versus measured results from the 8-case external-judge run on 2026-04-22 (generator `gemma4:31b-cloud`, judge `claude-haiku-4-5-20251001`):

| Metric | Target | Measured | Status |
|---|---|---|---|
| Mean relevance | >= 4.0 / 5 | **4.25** | pass |
| Mean novelty | >= 4.0 / 5 | **4.875** | pass |
| Mean pitch | >= 4.0 / 5 | **3.375** | below target |
| Mean accuracy | >= 4.0 / 5 | **4.0** | pass (borderline) |
| Invalid ID rate | 0% | **0%** | pass |
| Repeat rate | 0% | **0%** | pass |
| Timeout rate | 0% on provided tests | **0%** (max observed 12.47s) | pass |
| Validity (hard DQ checks) | 100% | **8/8 (100%)** | pass |
| A/B win rate vs baseline | >50% | **6/2/0 (75%)** | pass |

Full artifacts: `eval_report.json`, `eval_report.md`, `eval_summary.json`.

### Methodological Caveats

Two honest limitations shape how these numbers should be read.

First, the A/B comparator is the bundled `baseline.py`, which is *not* a static fallback — it asks the same `gemma4:31b-cloud` model to pick one movie from the top 5 by vote count (Interstellar, The Avengers, Deadpool, Avengers: Infinity War, Guardians of the Galaxy). On broad, popular-genre prompts it can land on a strong pick: in this run the baseline won the `single_genre_action` case (it picked Avengers: Infinity War and the judge preferred its "epic scale" pitch over ours) and the `mood_uplifting` case (baseline picked Guardians of the Galaxy with a direct emotional hook; ours picked About Time with a subtler pitch). A 75% win rate against that specific candidate pool is evidence our retrieval + filtering layer earns its keep on harder prompts — French grief drama, recent foreign film, funny horror, dark sci-fi — where the top-5 vote-count list can't answer the request. It does not predict the peer-vote outcome on competition day, where opponents are running their own tuned agents.

Second, pitch is still below the 4.0 target at 3.375. The judge consistently penalized descriptions that led with a tagline ("Love can only survive in the shadow of secrets.") instead of dropping straight into the movie's conflict, and marked down hard-tier cases (French grief, recent foreign) where the pitch felt generic relative to the specific request. Accuracy landed right at 4.0 — a small improvement from the previous run (3.75) driven by tighter grounding in CSV metadata — but borderline enough that we're not claiming it as solved.

Earlier in development, an evaluation round reported 100% A/B wins that turned out to be a **broken-baseline artifact** — the baseline call was silently failing on Ollama quota exhaustion and returning an error stub, which the judge of course preferred our real recommendation over. We now spot-check per-case baseline output before trusting any A/B number. The 75% reported above has been verified: the baseline returned real movie IDs with substantive pitch text in every case, and both of its wins were decided on pitch quality, not default fallbacks.

For prompt optimization, the evaluator can ask the fixed Ollama model for advisory suggestions after the judge run:

```bash
python evaluate.py --max-cases 8 --optimize-prompt
```

This writes `prompt_suggestions.md`. The optimizer is advisory only — every suggestion is reviewed manually, and the fixed model name, function signature, return schema, API key handling, ID validation, history filtering, and timeout guards stay as they are.

If a metric misses its target, we tune the weakest part of the loop: retrieval terms for relevance, history filtering for novelty, prompt examples for pitch quality, or metadata checks for accuracy. Before/after scores live in `eval_summary.json`.

## What's in This Submission

Three files form the core deliverable:

- `llm.py`
- `requirements.txt`
- `README.md`

The rest of the zip supports reproducibility, so anyone unzipping it can re-run the full evaluation pipeline without guesswork:

- `test.py` — the team's validity check referenced in Quick Start. Runs a short battery of recommendations and verifies each one meets the grading criteria (valid `tmdb_id`, description under 500 chars, latency under 20 seconds, not in watch history).
- `baseline.py` — the A/B comparator called by `evaluate.py`. Also present because `test.py`'s AST import scan otherwise flags `baseline` as a missing requirement.
- `evaluate.py` and `Eval packages/eval_cases.json` — the evaluation harness and the 8-case test set that produced the measured numbers above.
- `eval_report.json`, `eval_report.md`, `eval_summary.json` — the verified evaluation artifacts.
- `tmdb_top1000_movies.csv` — the movie corpus used for RAG, filtering, and `tmdb_id` validation.
- `.env` — the non-Ollama API keys (`TMDB_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_JUDGE_MODEL`). `llm.py` auto-loads it via `python-dotenv` at import time. The Ollama key is never bundled — the grader supplies their own.

Not included: `.venv/`, `__pycache__/`, `.chroma_db/`, `.eval_cache.json`, editor scratchpads, and anything containing a hardcoded `OLLAMA_API_KEY`.
