"""
Simple baseline recommender used only by evaluate.py for A/B comparison.

This mirrors the starter idea: show the LLM a tiny high-vote candidate list and
ask it for one JSON recommendation. It is intentionally less agentic than
llm.py, which makes it useful as a development benchmark.
"""

import json
import os

import ollama
import pandas as pd


MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_MOVIES = pd.read_csv(DATA_PATH).nlargest(5, "vote_count")


def _fallback(history_ids: list[int] | None = None) -> dict:
    blocked = {int(tid) for tid in (history_ids or [])}
    for row in TOP_MOVIES.itertuples():
        if int(row.tmdb_id) in blocked:
            continue
        desc = (
            f"{row.title} is a popular baseline pick with broad appeal. "
            f"It combines {row.genres.lower()} elements and gives the user a reliable, high-profile option."
        )
        return {"tmdb_id": int(row.tmdb_id), "description": desc[:500]}
    row = TOP_MOVIES.iloc[0]
    return {"tmdb_id": int(row["tmdb_id"]), "description": "Popular baseline recommendation."}


def baseline_get_recommendation(
    preferences: str, history: list[str], history_ids: list[int] = []
) -> dict:
    movie_list = "\n".join(
        f'- tmdb_id={row.tmdb_id} | "{row.title}" ({row.year}) | genres: {row.genres} | overview: {str(row.overview)[:200]}'
        for row in TOP_MOVIES.itertuples()
    )
    history_text = (
        ", ".join(f'"{name}" (tmdb_id={tid})' for name, tid in zip(history, history_ids))
        if history
        else "none"
    )
    prompt = f"""You are a movie recommendation assistant.

A user is looking for a movie to watch. Here are their preferences:
"{preferences}"

Movies they have already watched (do NOT recommend these):
{history_text}

Below is the list of candidate movies you may recommend. You MUST pick exactly one.

{movie_list}

Respond with ONLY a JSON object in this exact format:
{{"tmdb_id": <integer>, "description": "<a compelling blurb <=500 chars>"}}
"""

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
            timeout=20,
        )
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            think=False,
            options={"num_predict": 140, "temperature": 0.2},
        )
        result = json.loads(response.message.content)
        tmdb_id = int(result["tmdb_id"])
        if tmdb_id in {int(tid) for tid in (history_ids or [])}:
            return _fallback(history_ids)
        return {"tmdb_id": tmdb_id, "description": str(result.get("description", ""))[:500]}
    except Exception:
        return _fallback(history_ids)


def get_recommendation(
    preferences: str, history: list[str], history_ids: list[int] = []
) -> dict:
    return baseline_get_recommendation(preferences, history, history_ids)
