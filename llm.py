"""
REQUIRED ENV VARS:
  OLLAMA_API_KEY - API key for the Ollama cloud service (injected by grader)
OPTIONAL ENV VARS:
  TMDB_API_KEY - TMDB v3 API key or read-access token for actor/movie enrichment
  ANTHROPIC_API_KEY - development-only, used by evaluate.py external judge;
    get_recommendation() never reads it
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import ollama
import pandas as pd

# Auto-load non-Ollama API keys from .env if present.
# Must run before any os.getenv() / os.environ.get() calls below.
# Defensive: if python-dotenv is missing, silently continue (grader may export vars directly).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
_RAG_TOPN = 10
_MAX_CANDIDATES = 8
_REQUEST_TIMEOUT_SECONDS = 12.0
_TMDB_TIMEOUT_SECONDS = 2.0
_MAX_DESCRIPTION_CHARS = 500
_TMDB_BASE_URL = "https://api.themoviedb.org/3"
_TMDB_CACHE: dict[str, dict | None] = {}


# ---------------------------------------------------------------------------
# Dataset - expose the full set so test.py builds a correct VALID_IDS set
# ---------------------------------------------------------------------------
_raw = pd.read_csv(DATA_PATH)
_raw["tmdb_id"] = _raw["tmdb_id"].astype(int)
TOP_MOVIES = _raw
MOVIES_DF = TOP_MOVIES
VALID_IDS = frozenset(MOVIES_DF["tmdb_id"].tolist())


def _s(val, n: int | None = None) -> str:
    out = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:n] if n else out


def _get(row, name: str, default=None):
    if hasattr(row, name):
        return getattr(row, name)
    if hasattr(row, "get"):
        return row.get(name, default)
    return default


def _normalize_title(title: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", _s(title).lower()).strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return re.sub(r"\s+", " ", text)


_TITLE_TO_IDS: dict[str, set[int]] = {}
_TITLE_COMPACT_TO_IDS: dict[str, set[int]] = {}
for _row in MOVIES_DF.itertuples():
    for _name in {_s(_get(_row, "title")), _s(_get(_row, "original_title"))}:
        _norm = _normalize_title(_name)
        if _norm:
            _tid = int(_get(_row, "tmdb_id"))
            _TITLE_TO_IDS.setdefault(_norm, set()).add(_tid)
            _compact = re.sub(r"\s+", "", _norm)
            # Require >=5 chars to avoid clashes with short words like "up", "it".
            if len(_compact) >= 5:
                _TITLE_COMPACT_TO_IDS.setdefault(_compact, set()).add(_tid)


def _normalize_person(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", _s(name).lower()).strip()
    return re.sub(r"\s+", " ", text)


def _split_people(value: str) -> list[str]:
    return [_s(part) for part in _s(value).split(",") if _s(part)]


_ACTOR_TO_IDS: dict[str, set[int]] = {}
_ACTOR_DISPLAY: dict[str, str] = {}
for _row in MOVIES_DF.itertuples():
    for _actor in _split_people(_get(_row, "top_cast")):
        _norm = _normalize_person(_actor)
        if _norm:
            _ACTOR_TO_IDS.setdefault(_norm, set()).add(int(_get(_row, "tmdb_id")))
            _ACTOR_DISPLAY.setdefault(_norm, _actor)
_KNOWN_ACTORS = sorted(_ACTOR_TO_IDS, key=lambda name: (-len(name), name))


def _detect_actor_names(text: str) -> list[str]:
    normalized = f" {_normalize_person(text)} "
    # Compact form handles inputs like "adrienbrody" (no space between names).
    compact = re.sub(r"\s+", "", _normalize_person(text))
    matches: list[str] = []
    consumed: set[str] = set()
    for actor_norm in _KNOWN_ACTORS:
        if len(actor_norm.split()) < 2:
            continue
        actor_tokens = set(actor_norm.split())
        if actor_tokens & consumed:
            continue
        actor_compact = re.sub(r"\s+", "", actor_norm)
        spaced_hit = f" {actor_norm} " in normalized
        # Only accept a compact match if it's long enough to be distinctive (>=8
        # chars) — guards against accidental substring collisions.
        compact_hit = len(actor_compact) >= 8 and actor_compact in compact
        if not (spaced_hit or compact_hit):
            continue
        matches.append(_ACTOR_DISPLAY[actor_norm])
        consumed.update(actor_tokens)
        if len(matches) >= 3:
            break
    return matches


def _person_in_text(person: str, text: str) -> bool:
    return f" {_normalize_person(person)} " in f" {_normalize_person(text)} "


def _tmdb_get_json(path: str, params: dict | None = None) -> dict | None:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return None

    params = dict(params or {})
    headers = {"Accept": "application/json"}
    # TMDB accepts either a v3 API key as api_key=... or a v4 read token as Bearer.
    if api_key.startswith("eyJ") or api_key.count(".") >= 2:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    query = urllib.parse.urlencode(params)
    url = f"{_TMDB_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"

    if url in _TMDB_CACHE:
        return _TMDB_CACHE[url]

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_TMDB_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        payload = None

    _TMDB_CACHE[url] = payload
    return payload


def _tmdb_person_movie_ids(person_name: str) -> set[int]:
    search = _tmdb_get_json(
        "/search/person",
        {"query": person_name, "include_adult": "false", "language": "en-US", "page": 1},
    )
    if not search or not search.get("results"):
        return set()

    wanted = _normalize_person(person_name)
    results = search.get("results", [])
    exact = [
        result
        for result in results
        if _normalize_person(result.get("name", "")) == wanted
    ]
    person = exact[0] if exact else results[0]
    person_id = person.get("id")
    if not person_id:
        return set()

    credits = _tmdb_get_json(f"/person/{person_id}/movie_credits", {"language": "en-US"})
    if not credits:
        return set()

    ids: set[int] = set()
    for section in ("cast", "crew"):
        for item in credits.get(section, []):
            try:
                ids.add(int(item.get("id")))
            except (TypeError, ValueError):
                continue
    return ids & VALID_IDS


def _tmdb_actor_candidates(actor_names: list[str], exclude: set[int]) -> list[dict]:
    out: dict[int, dict] = {}
    for actor in actor_names:
        for tmdb_id in _tmdb_person_movie_ids(actor):
            if tmdb_id in exclude:
                continue
            rows = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
            if rows.empty:
                continue
            meta = _to_meta(rows.iloc[0])
            confirmed = set(_split_people(meta.get("api_confirmed_actors", "")))
            confirmed.add(actor)
            meta["api_confirmed_actors"] = ", ".join(sorted(confirmed))
            out[tmdb_id] = meta
    return list(out.values())


def _tmdb_enrich_movie_meta(meta: dict) -> dict:
    enriched = dict(meta)
    tmdb_id = enriched.get("tmdb_id")
    if not tmdb_id:
        return enriched

    details = _tmdb_get_json(f"/movie/{tmdb_id}", {"language": "en-US"})
    if not details:
        return enriched

    enriched["overview"] = _s(enriched.get("overview")) or _s(details.get("overview"), 320)
    enriched["tagline"] = _s(enriched.get("tagline")) or _s(details.get("tagline"), 140)
    if not enriched.get("tmdb_url"):
        enriched["tmdb_url"] = f"https://www.themoviedb.org/movie/{tmdb_id}"
    return enriched


def _tmdb_lookup_movie(title: str) -> dict | None:
    """Search TMDB for a movie by title; return genre/keyword/overview context."""
    search = _tmdb_get_json(
        "/search/movie",
        {"query": title, "language": "en-US", "page": 1, "include_adult": "false"},
    )
    if not search or not search.get("results"):
        return None

    wanted = _normalize_title(title)
    results = search["results"]
    exact = [
        r for r in results
        if _normalize_title(r.get("title", "")) == wanted
        or _normalize_title(r.get("original_title", "")) == wanted
    ]
    movie = exact[0] if exact else results[0]

    movie_id = movie.get("id")
    if not movie_id:
        return None

    details = _tmdb_get_json(
        f"/movie/{movie_id}",
        {"language": "en-US", "append_to_response": "keywords,credits"},
    )
    if not details:
        return None

    genres = ", ".join(g["name"] for g in details.get("genres", []))
    keywords = ", ".join(
        k["name"] for k in details.get("keywords", {}).get("keywords", [])[:10]
    )
    overview = _s(details.get("overview"), 400)
    year_raw = (details.get("release_date") or "")[:4]
    year = int(year_raw) if year_raw.isdigit() else 0
    director = next(
        (p["name"] for p in details.get("credits", {}).get("crew", []) if p.get("job") == "Director"),
        "",
    )
    return {
        "title": _s(details.get("title")),
        "year": year,
        "genres": genres,
        "keywords": keywords,
        "overview": overview,
        "director": director,
    }


def _movie_corpus(row) -> str:
    return " ".join(
        [
            _s(_get(row, "title")),
            _s(_get(row, "original_title")),
            _s(_get(row, "genres")),
            _s(_get(row, "keywords"), 240),
            _s(_get(row, "tagline"), 140),
            _s(_get(row, "overview"), 500),
            _s(_get(row, "director")),
            _s(_get(row, "top_cast"), 160),
            _s(_get(row, "us_rating")),
        ]
    )


def _to_meta(row) -> dict:
    year = _get(row, "year", 0)
    vote_average = _get(row, "vote_average", 0)
    vote_count = _get(row, "vote_count", 0)
    popularity = _get(row, "popularity", 0)
    return {
        "tmdb_id": int(_get(row, "tmdb_id")),
        "title": _s(_get(row, "title")),
        "year": int(year) if pd.notna(year) else 0,
        "genres": _s(_get(row, "genres")),
        "overview": _s(_get(row, "overview"), 320),
        "tagline": _s(_get(row, "tagline"), 140),
        "director": _s(_get(row, "director")),
        "top_cast": _s(_get(row, "top_cast"), 140),
        "vote_average": round(float(vote_average), 1) if pd.notna(vote_average) else 0.0,
        "vote_count": int(vote_count) if pd.notna(vote_count) else 0,
        "popularity": float(popularity) if pd.notna(popularity) else 0.0,
        "keywords": _s(_get(row, "keywords"), 160),
        "tmdb_url": _s(_get(row, "tmdb_url")),
        "api_confirmed_actors": _s(_get(row, "api_confirmed_actors")),
        "original_language": _s(_get(row, "original_language")),
        "production_countries": _s(_get(row, "production_countries")),
    }


# ---------------------------------------------------------------------------
# Local RAG index - fast, deterministic, and safe under the 20-second limit
# ---------------------------------------------------------------------------
_tfidf_vec = None
_tfidf_mat = None
_tfidf_ids: list[int] = []


def _init_rag() -> None:
    global _tfidf_vec, _tfidf_mat, _tfidf_ids

    if not _SKLEARN_OK:
        return

    try:
        corpus = [_movie_corpus(row) for row in MOVIES_DF.itertuples()]
        vec = TfidfVectorizer(
            max_features=12000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        _tfidf_mat = vec.fit_transform(corpus)
        _tfidf_vec = vec
        _tfidf_ids = list(MOVIES_DF["tmdb_id"].astype(int))
    except Exception:
        _tfidf_vec = None
        _tfidf_mat = None
        _tfidf_ids = []


_init_rag()


def _quality_score(meta: dict) -> float:
    rating = float(meta.get("vote_average", 0) or 0)
    votes = int(meta.get("vote_count", 0) or 0)
    popularity = float(meta.get("popularity", 0) or 0)
    vote_confidence = min(math.log10(votes + 1) / 4.0, 1.0)
    popularity_signal = min(math.log10(popularity + 1) / 4.0, 1.0)
    return (rating / 10.0) * 1.5 + vote_confidence + popularity_signal * 0.4


def _best_by_quality(exclude: set[int], n: int) -> list[dict]:
    metas = [
        _to_meta(row)
        for row in MOVIES_DF.itertuples()
        if int(_get(row, "tmdb_id")) not in exclude
    ]
    metas.sort(key=_quality_score, reverse=True)
    return metas[:n]


def _rag_retrieve(query: str, exclude: set[int], n: int = _RAG_TOPN) -> list[dict]:
    """Return relevant unwatched movies from the local movie corpus."""
    query = _s(query)
    if not query:
        return _best_by_quality(exclude, n)

    if _tfidf_vec is not None and _tfidf_mat is not None:
        try:
            qvec = _tfidf_vec.transform([query])
            sims = _cos_sim(qvec, _tfidf_mat)[0]
            order = sims.argsort()[::-1]
            out = []
            for idx in order:
                tid = int(_tfidf_ids[int(idx)])
                if tid in exclude:
                    continue
                meta = _to_meta(MOVIES_DF.iloc[int(idx)])
                out.append(meta)
                if len(out) >= n:
                    break
            if out:
                return out
        except Exception:
            pass

    tokens = re.findall(r"[a-zA-Z0-9]{4,}", query.lower())
    scored: list[tuple[float, dict]] = []
    for row in MOVIES_DF.itertuples():
        tid = int(_get(row, "tmdb_id"))
        if tid in exclude:
            continue
        meta = _to_meta(row)
        hay = _movie_corpus(row).lower()
        token_score = sum(hay.count(token) for token in tokens)
        scored.append((token_score + _quality_score(meta), meta))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [meta for _, meta in scored[:n]]


# ---------------------------------------------------------------------------
# Local tools for preference parsing and candidate enrichment
# ---------------------------------------------------------------------------
_GENRE_KEYWORDS: dict[str, list[str]] = {
    "Action": ["action", "fight", "battle", "explosion", "combat", "chase", "superhero"],
    "Adventure": ["adventure", "quest", "journey", "survival", "exploration"],
    "Animation": ["animated", "animation", "cartoon", "pixar", "disney", "family"],
    "Comedy": ["comedy", "funny", "humor", "hilarious", "laugh", "lighthearted"],
    "Crime": ["crime", "heist", "detective", "murder", "gangster", "noir"],
    "Drama": ["drama", "emotional", "powerful", "touching", "moving", "character"],
    "Fantasy": ["fantasy", "magic", "wizard", "dragon", "mythical", "fairy tale"],
    "Horror": ["horror", "scary", "terrifying", "creepy", "haunted", "slasher"],
    "Mystery": ["mystery", "whodunit", "investigation", "puzzle", "detective"],
    "Romance": ["romance", "romantic", "love story", "relationship", "date night"],
    "Science Fiction": ["sci-fi", "science fiction", "scifi", "space", "alien", "robot", "future"],
    "Thriller": ["thriller", "suspense", "tense", "twist", "psychological"],
    "War": ["war", "soldier", "battlefield", "military"],
}

_MOOD_KEYWORDS: dict[str, list[str]] = {
    "feel-good": ["feel good", "uplifting", "comfort", "warm", "optimistic", "heartwarming", "rough week", "had a rough", "cheer"],
    "mind-bending": ["mind bending", "philosophical", "cerebral", "surreal", "twist", "dream"],
    "dark": ["dark", "gritty", "bleak", "violent", "morally complex"],
    "fast-paced": ["fast paced", "exciting", "adrenaline", "intense", "nonstop"],
    "slow-burn": ["slow burn", "atmospheric", "moody", "patient", "subtle"],
    "fun": ["fun", "playful", "light", "breezy", "entertaining"],
}

_SPECIAL_TERMS = [
    "time travel",
    "found family",
    "serial killer",
    "space opera",
    "coming of age",
    "based on a true story",
    "super hero",
    "superhero",
    "post apocalyptic",
    "romantic comedy",
]

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "but",
    "can",
    "for",
    "from",
    "have",
    "like",
    "love",
    "movie",
    "movies",
    "need",
    "really",
    "seen",
    "show",
    "similar",
    "something",
    "that",
    "the",
    "this",
    "want",
    "watch",
    "with",
    "actor",
    "actress",
    "film",
    "films",
    "star",
    "stars",
    "starring",
    "featuring",
    "foreign",
    "international",
    "recent",
    "subtitle",
    "subtitles",
    "subtitled",
}

_SIMILARITY_CUES = [
    "more like",
    "similar to",
    "same vibe",
    "reminds me of",
    "another movie like",
    "another film like",
    "something like",
    "movies like",
    "films like",
]


def _extract_reference_titles(preferences: str) -> list[str]:
    lower = _s(preferences).lower()
    titles: list[str] = []
    for cue in _SIMILARITY_CUES:
        idx = lower.find(cue)
        if idx == -1:
            continue
        after = preferences[idx + len(cue):].strip()
        segment = re.split(r'[.!?\n]|\s+(?:and|or|but)\b', after)[0]
        words = segment.split()[:5]
        title = " ".join(words).strip(" ,;:")
        if title and title.lower() not in _STOPWORDS and title not in titles:
            titles.append(title)
    return titles[:2]


def _tmdb_reference_context(preferences: str) -> str:
    titles = _extract_reference_titles(preferences)
    if not titles:
        return ""
    parts: list[str] = []
    for title in titles:
        meta = _tmdb_lookup_movie(title)
        if not meta:
            continue
        parts.append(" ".join(filter(None, [meta["genres"], meta["keywords"], meta["overview"]])))
    return " ".join(parts)


def _requested_person_names(preferences: str) -> list[str]:
    actors = _detect_actor_names(preferences)
    if actors:
        return actors

    lower = _normalize_person(preferences)
    non_person = set(_STOPWORDS)
    for terms in list(_GENRE_KEYWORDS.values()) + list(_MOOD_KEYWORDS.values()):
        for term in terms:
            non_person.update(_normalize_person(term).split())

    candidates: list[str] = []
    patterns = [
        r"(?:starring|featuring|with|directed\s+by)\s+([a-z0-9]+\s+[a-z0-9]+(?:\s+[a-z0-9]+)?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            words = [
                word
                for word in match.group(1).split()
                if word not in non_person and len(word) > 1
            ]
            if len(words) >= 2:
                name = " ".join(word.capitalize() for word in words[:3])
                if name not in candidates:
                    candidates.append(name)
            if len(candidates) >= 2:
                return candidates
    return candidates


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if str(item)]


_LANGUAGE_MAP: dict[str, str] = {
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "zh",
    "korean": "ko",
    "japanese": "ja",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "german": "de",
    "hindi": "hi",
    "bollywood": "hi",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "danish": "da",
    "swedish": "sv",
    "norwegian": "no",
    "thai": "th",
    "turkish": "tr",
}

_LANGUAGE_DISPLAY: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "zh": "Chinese",
}

_COUNTRY_MAP: dict[str, str] = {
    "chinese": "China",
    "korean": "South Korea",
    "japanese": "Japan",
    "french": "France",
    "italian": "Italy",
    "german": "Germany",
    "hindi": "India",
    "bollywood": "India",
    "thai": "Thailand",
    "turkish": "Turkey",
}


def _language_label(code: str | None) -> str:
    code = _s(code).lower()
    return _LANGUAGE_DISPLAY.get(code, code.upper() if code else "non-English")


def _detect_language(text: str) -> tuple[str | None, str | None]:
    lower = text.lower()
    for word, lang_code in _LANGUAGE_MAP.items():
        if re.search(rf"\b{word}\b", lower):
            country = _COUNTRY_MAP.get(word)
            return lang_code, country
    return None, None


def _detect_foreign_request(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(foreign|international|world\s+cinema|non[-\s]?english|subtitled?|subtitles)\b",
            lower,
        )
    )


_ERA_NOSTALGIA_TERMS: frozenset[str] = frozenset(
    {
        "nostalgia", "period piece", "retro", "classic", "vintage",
        "historical", "old-fashioned", "coming of age", "childhood",
        "based on novel", "fairy tale", "folklore", "old west", "medieval",
    }
)

_DECADE_PATTERNS: list[tuple[re.Pattern, int, int]] = [
    (re.compile(r"\b(1920s|silent\s+era)\b"), 1920, 1929),
    (re.compile(r"\b1930s\b"), 1930, 1939),
    (re.compile(r"\b1940s\b"), 1940, 1949),
    (re.compile(r"\b1950s\b"), 1950, 1959),
    (re.compile(r"\b1960s\b"), 1960, 1969),
    (re.compile(r"\b1970s\b"), 1970, 1979),
    (re.compile(r"\b(1980s|80s|eighties|the\s+80s|from\s+the\s+80s)\b"), 1980, 1989),
    (re.compile(r"\b(1990s|90s|nineties|the\s+90s|from\s+the\s+90s)\b"), 1990, 1999),
    (re.compile(r"\b(2000s|early\s+2000s)\b"), 2000, 2009),
    (re.compile(r"\b2010s\b"), 2010, 2019),
    (re.compile(r"\b(recent\s+(?:film|movie|foreign)|2020s)\b"), 2020, 2099),
]


def _detect_era(text: str) -> tuple[int | None, int | None]:
    lower = text.lower()
    for pattern, year_min, year_max in _DECADE_PATTERNS:
        if pattern.search(lower):
            return year_min, year_max
    m = re.search(r"\b(19[5-9]\d|20[012]\d)\b", lower)
    if m:
        y = int(m.group(1))
        return y, y
    return None, None


def _preference_profile(preferences: str) -> dict:
    lower = _s(preferences).lower()
    actors = _detect_actor_names(preferences)
    requested_people = _requested_person_names(preferences)
    actor_tokens = {
        token
        for actor in actors
        for token in _normalize_person(actor).split()
    }

    # Collapse doubled consonants before matching so common typos like
    # "rommance" or "horro" still trigger the right genre/mood.
    _collapsed = re.sub(r'([bcdfghjklmnpqrstvwxyz])\1+', r'\1', lower)

    def _term_in(term: str) -> bool:
        return term in lower or term in _collapsed

    genres = [
        genre
        for genre, terms in _GENRE_KEYWORDS.items()
        if any(_term_in(term) for term in terms)
    ]
    moods = [
        mood
        for mood, terms in _MOOD_KEYWORDS.items()
        if any(_term_in(term) for term in terms)
    ]

    keywords: list[str] = []
    for phrase in _SPECIAL_TERMS:
        if phrase in lower and phrase not in keywords:
            keywords.append(phrase)

    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", lower):
        token = token.replace("-", " ")
        if token not in _STOPWORDS and token not in actor_tokens and token not in keywords:
            keywords.append(token)
        if len(keywords) >= 8:
            break

    lang_code, country = _detect_language(preferences)
    year_min, year_max = _detect_era(preferences)
    foreign_requested = _detect_foreign_request(preferences)

    return {
        "genres": genres[:3],
        "moods": moods[:3],
        "keywords": keywords[:8],
        "actors": actors[:3],
        "requested_people": requested_people[:3],
        "language": lang_code,
        "country": country,
        "year_min": year_min,
        "year_max": year_max,
        "foreign_requested": foreign_requested,
    }


def _candidate_match_score(meta: dict, profile: dict) -> float:
    corpus = " ".join(
        [
            meta.get("title", ""),
            meta.get("genres", ""),
            meta.get("overview", ""),
            meta.get("keywords", ""),
            meta.get("tagline", ""),
            meta.get("director", ""),
            meta.get("top_cast", ""),
            meta.get("api_confirmed_actors", ""),
        ]
    ).lower()
    genres_text = meta.get("genres", "").lower()

    score = _quality_score(meta)
    for genre in profile.get("genres", []):
        if genre.lower() in genres_text:
            score += 4.0
    for keyword in profile.get("keywords", []):
        if keyword.lower() in corpus:
            score += 1.25
    for mood in profile.get("moods", []):
        terms = _MOOD_KEYWORDS.get(mood, [])
        if any(term in corpus for term in terms):
            score += 1.5
    actor_matches = 0
    for actor in profile.get("actors", []):
        actor_evidence = f"{meta.get('top_cast', '')} {meta.get('api_confirmed_actors', '')}"
        if _person_in_text(actor, actor_evidence):
            score += 20.0
            actor_matches += 1
    if profile.get("actors") and actor_matches == 0:
        score -= 20.0
    return score


def _tool_filter_movies(
    genres: list[str] | str | None = None,
    keywords: list[str] | str | None = None,
    moods: list[str] | str | None = None,
    actors: list[str] | str | None = None,
    min_vote_average: float | None = None,
    exclude: set[int] | None = None,
    limit: int = 15,
    language: str | None = None,
    country: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    foreign_requested: bool = False,
) -> list[dict]:
    """
    Local content-filtering tool.

    It searches the complete movie table by genre, keyword, and mood, then ranks
    matches by preference fit plus quality/popularity confidence.
    """
    genre_list = _as_list(genres)
    keyword_list = _as_list(keywords)
    mood_list = _as_list(moods)
    actor_list = _as_list(actors)
    exclude = exclude or set()
    profile = {
        "genres": genre_list,
        "keywords": keyword_list,
        "moods": mood_list,
        "actors": actor_list,
    }

    scored: list[tuple[float, dict]] = []
    for row in MOVIES_DF.itertuples():
        tid = int(_get(row, "tmdb_id"))
        if tid in exclude:
            continue

        meta = _to_meta(row)
        corpus = _movie_corpus(row).lower()
        genres_text = meta["genres"].lower()

        row_language = _s(_get(row, "original_language")).lower()
        if foreign_requested and (not row_language or row_language == "en"):
            continue
        if language and row_language != language:
            if not country or country.lower() not in _s(_get(row, "production_countries")).lower():
                continue
        movie_year = int(_get(row, "year") or 0)
        if year_min and movie_year and movie_year < year_min:
            continue
        if year_max and movie_year and movie_year > year_max:
            continue
        if "animation" in genres_text and "Animation" not in (genre_list or []):
            continue
        # Prevent animated films from surfacing for romance/romcom requests
        # unless the user explicitly asked for animation or family content.
        _non_anim_romance = (
            "romance" in [g.lower() for g in (genre_list or [])]
            and "Animation" not in (genre_list or [])
            and "Family" not in (genre_list or [])
        )
        if _non_anim_romance and "animation" in genres_text:
            continue
        if actor_list and not any(_person_in_text(actor, meta["top_cast"]) for actor in actor_list):
            continue
        if min_vote_average is not None and meta["vote_average"] < float(min_vote_average):
            continue
        if genre_list and not any(genre.lower() in genres_text for genre in genre_list):
            continue
        if keyword_list and not any(keyword.lower() in corpus for keyword in keyword_list):
            if not genre_list and not actor_list:
                continue
        if mood_list:
            mood_terms = [term for mood in mood_list for term in _MOOD_KEYWORDS.get(mood, [])]
            if mood_terms and not any(term in corpus for term in mood_terms):
                if not genre_list and not keyword_list and not actor_list:
                    continue

        scored.append((_candidate_match_score(meta, profile), meta))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [meta for _, meta in scored[: min(int(limit), 30)]]


def _row_matches_structural_constraints(
    row,
    profile: dict,
    relax_year: bool = False,
    relax_genre: bool = False,
) -> bool:
    row_language = _s(_get(row, "original_language")).lower()
    row_country = _s(_get(row, "production_countries")).lower()

    if profile.get("foreign_requested") and (not row_language or row_language == "en"):
        return False

    lang_filter = profile.get("language")
    country_filter = _s(profile.get("country")).lower()
    if lang_filter and row_language != lang_filter:
        if not country_filter or country_filter not in row_country:
            return False

    movie_year = int(_get(row, "year") or 0)
    if not relax_year:
        year_min = profile.get("year_min")
        year_max = profile.get("year_max")
        if movie_year and year_min and movie_year < year_min:
            return False
        if movie_year and year_max and movie_year > year_max:
            return False

    if not relax_genre and profile.get("genres"):
        genres_text = _s(_get(row, "genres")).lower()
        if not any(genre.lower() in genres_text for genre in profile.get("genres", [])):
            return False

    return True


def _candidate_respects_final_constraints(candidate: dict | None, profile: dict) -> bool:
    if not candidate:
        return False

    row_language = _s(candidate.get("original_language")).lower()
    row_country = _s(candidate.get("production_countries")).lower()

    if profile.get("foreign_requested") and (not row_language or row_language == "en"):
        return False

    lang_filter = profile.get("language")
    country_filter = _s(profile.get("country")).lower()
    if lang_filter and row_language != lang_filter:
        if not country_filter or country_filter not in row_country:
            return False

    # A constraint_note means the local dataset could not satisfy the exact
    # year/genre intersection, so a deliberate fallback is allowed. Otherwise,
    # decade/year requests are treated as hard filters all the way to output.
    if not candidate.get("constraint_note"):
        movie_year = int(candidate.get("year") or 0)
        year_min = profile.get("year_min")
        year_max = profile.get("year_max")
        if movie_year and year_min and movie_year < year_min:
            return False
        if movie_year and year_max and movie_year > year_max:
            return False

    return True


def _natural_join(parts: list[str]) -> str:
    parts = [part for part in parts if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _year_range_text(profile: dict) -> str:
    year_min = profile.get("year_min")
    year_max = profile.get("year_max")
    if year_min and year_max:
        if year_min == year_max:
            return str(year_min)
        return f"{year_min}-{year_max}"
    if year_min:
        return f"after {year_min}"
    if year_max:
        return f"before {year_max}"
    return ""


def _dataset_has_year_range(profile: dict) -> bool:
    year_min = profile.get("year_min")
    year_max = profile.get("year_max")
    if not (year_min or year_max):
        return True
    years = MOVIES_DF["year"].fillna(0).astype(int)
    mask = pd.Series(True, index=MOVIES_DF.index)
    if year_min:
        mask &= years >= int(year_min)
    if year_max:
        mask &= years <= int(year_max)
    return bool(mask.any())


def _preserved_constraint_text(profile: dict) -> str:
    preserved: list[str] = []
    genres = [genre.lower() for genre in profile.get("genres", [])]
    if genres:
        preserved.append(f"the {_natural_join(genres)} signal")
    if profile.get("language"):
        preserved.append(f"the {_language_label(profile.get('language'))} language signal")
    elif profile.get("foreign_requested"):
        preserved.append("the non-English/subtitles signal")
    return _natural_join(preserved)


def _constraint_fallback_note(profile: dict, relaxed_year: bool = False) -> str:
    preserved = _preserved_constraint_text(profile)
    preserved_clause = f", so I preserved {preserved}" if preserved else ", so I stayed inside the current library"

    if relaxed_year:
        year_text = _year_range_text(profile)
        if not _dataset_has_year_range(profile):
            return (
                f"Those {year_text} titles aren't available in our current library{preserved_clause} "
                "— here's the closest match we have."
            )
        return (
            f"We don't have an exact {year_text} match for the full request in our current library{preserved_clause} "
            "— here's the nearest available title."
        )

    if profile.get("genres"):
        return (
            f"The local CSV has no exact genre match after the hard filters{preserved_clause} "
            "and picked the strongest available title."
        )

    return f"The local CSV cannot satisfy the exact request{preserved_clause}."


def _closest_safe_link(candidate: dict, preferences: str) -> str:
    profile = _preference_profile(preferences)
    title = candidate.get("title", "This movie")
    year = candidate.get("year", 0)
    genres = [genre.lower() for genre in profile.get("genres", [])]
    if genres:
        return (
            f"{title} ({year}) is the closest safe pick because it keeps "
            f"{_natural_join(genres)} momentum inside the allowed dataset"
        )
    if profile.get("language"):
        return (
            f"{title} ({year}) is the closest safe pick because it keeps the "
            f"{_language_label(profile.get('language'))} language signal inside the allowed dataset"
        )
    if profile.get("foreign_requested"):
        return (
            f"{title} ({year}) is the closest safe pick because it keeps the "
            "non-English/subtitles constraint inside the allowed dataset"
        )
    year_min = profile.get("year_min")
    if year_min:
        decade = f"{(year_min // 10) * 10}s"
        candidate_keywords = candidate.get("keywords", "").lower()
        candidate_genres = candidate.get("genres", "")
        has_nostalgic_feel = any(term in candidate_keywords for term in _ERA_NOSTALGIA_TERMS)
        if has_nostalgic_feel:
            return (
                f"{title} ({year}) is the closest safe pick — "
                f"its nostalgic tone and timeless themes echo the spirit of {decade} cinema"
            )
        first_genre = candidate_genres.split(",")[0].strip().lower() if candidate_genres else ""
        if first_genre:
            return (
                f"{title} ({year}) is the closest safe pick — "
                f"its {first_genre} heart and classic storytelling carry the feel of the {decade}"
            )
        return (
            f"{title} ({year}) is the closest safe pick — "
            f"it's the nearest available title in our library to the {decade}, "
            "sharing the timeless appeal of films from that era"
        )
    return f"{title} ({year}) is the closest safe pick because it stays inside the allowed dataset"


def _structural_fallback_candidates(profile: dict, exclude: set[int], limit: int) -> list[dict]:
    has_structural_constraint = bool(
        profile.get("year_min")
        or profile.get("year_max")
        or profile.get("language")
        or profile.get("foreign_requested")
    )
    if not has_structural_constraint:
        return []

    def collect(relax_year: bool = False, relax_genre: bool = False) -> list[dict]:
        scored: list[tuple[float, dict]] = []
        midpoint = None
        if profile.get("year_min") and profile.get("year_max"):
            midpoint = (int(profile["year_min"]) + int(profile["year_max"])) / 2
        for row in MOVIES_DF.itertuples():
            tid = int(_get(row, "tmdb_id"))
            if tid in exclude:
                continue
            if not _row_matches_structural_constraints(row, profile, relax_year, relax_genre):
                continue
            meta = _to_meta(row)
            score = _candidate_match_score(meta, profile)
            if relax_year and midpoint is not None:
                year = int(_get(row, "year") or 0)
                if year:
                    score -= abs(year - midpoint) * 10.0
                era_corpus = f"{meta.get('keywords', '')} {meta.get('overview', '')}".lower()
                if any(term in era_corpus for term in _ERA_NOSTALGIA_TERMS):
                    score += 8.0
            scored.append((score, meta))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [meta for _, meta in scored[:limit]]

    exact = collect()
    if exact:
        return exact

    relaxed = []
    if profile.get("year_min") or profile.get("year_max"):
        relaxed = collect(relax_year=True)
        if relaxed:
            note = _constraint_fallback_note(profile, relaxed_year=True)
            for meta in relaxed:
                meta["constraint_note"] = note
            return relaxed

    if profile.get("genres"):
        relaxed = collect(relax_genre=True)
        if relaxed:
            note = _constraint_fallback_note(profile)
            for meta in relaxed:
                meta["constraint_note"] = note
            return relaxed

    return []


def _history_exclude_ids(history: list[str] | None, history_ids: list[int] | None) -> set[int]:
    exclude: set[int] = set()
    for item in history_ids or []:
        try:
            exclude.add(int(item))
        except (TypeError, ValueError):
            continue

    for title in history or []:
        norm = _normalize_title(title)
        if norm in _TITLE_TO_IDS:
            exclude.update(_TITLE_TO_IDS[norm])
            continue
        compact = re.sub(r"\s+", "", norm)
        if compact and compact in _TITLE_COMPACT_TO_IDS:
            exclude.update(_TITLE_COMPACT_TO_IDS[compact])
    return exclude


def _history_similarity_context(preferences: str, history: list[str] | None) -> str:
    lower = _s(preferences).lower()
    if not any(cue in lower for cue in _SIMILARITY_CUES):
        return ""

    snippets = []
    for title in history or []:
        ids = _TITLE_TO_IDS.get(_normalize_title(title), set())
        for tmdb_id in ids:
            rows = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
            if rows.empty:
                continue
            meta = _to_meta(rows.iloc[0])
            snippets.append(
                " ".join(
                    [
                        meta["genres"],
                        meta["keywords"],
                        meta["tagline"],
                        meta["overview"][:220],
                    ]
                )
            )
            break
    return " ".join(snippets[:3])


def _build_candidates(
    preferences: str, exclude: set[int], history: list[str] | None = None
) -> tuple[list[dict], dict]:
    retrieval_query = " ".join(
        part for part in [preferences, _history_similarity_context(preferences, history), _tmdb_reference_context(preferences)] if part
    )
    profile = _preference_profile(retrieval_query)

    pool: dict[int, dict] = {}
    actor_candidates: list[dict] = []
    actor_queries = profile["actors"] or profile["requested_people"]
    if actor_queries:
        profile["actors"] = actor_queries[:3]
        actor_candidates = _tool_filter_movies(
            actors=actor_queries,
            genres=profile["genres"],
            keywords=profile["keywords"],
            moods=profile["moods"],
            exclude=exclude,
            language=profile.get("language"),
            country=profile.get("country"),
            year_min=profile.get("year_min"),
            year_max=profile.get("year_max"),
            foreign_requested=profile.get("foreign_requested", False),
            limit=_MAX_CANDIDATES,
        )
        if not actor_candidates:
            actor_candidates = _tool_filter_movies(
                actors=actor_queries,
                exclude=exclude,
                language=profile.get("language"),
                country=profile.get("country"),
                year_min=profile.get("year_min"),
                year_max=profile.get("year_max"),
                foreign_requested=profile.get("foreign_requested", False),
                limit=_MAX_CANDIDATES,
            )
        tmdb_candidates = _tmdb_actor_candidates(actor_queries, exclude)
        if tmdb_candidates:
            profile["tmdb_api_used"] = True
            seen = {meta["tmdb_id"] for meta in actor_candidates}
            for meta in tmdb_candidates:
                if not _candidate_respects_final_constraints(meta, profile):
                    continue
                if meta["tmdb_id"] not in seen:
                    actor_candidates.append(meta)
                    seen.add(meta["tmdb_id"])
        else:
            profile["tmdb_api_used"] = False
        if actor_candidates:
            profile["actor_match_available"] = True
            actor_candidates.sort(
                key=lambda meta: _candidate_match_score(meta, profile),
                reverse=True,
            )
            return actor_candidates[:_MAX_CANDIDATES], profile
        profile["actor_match_available"] = False

    for meta in _rag_retrieve(retrieval_query, exclude, n=_RAG_TOPN):
        row_df = MOVIES_DF[MOVIES_DF["tmdb_id"] == meta["tmdb_id"]]
        if row_df.empty:
            pool[meta["tmdb_id"]] = meta
            continue
        r = row_df.iloc[0]
        if not _row_matches_structural_constraints(r, profile, relax_genre=True):
            continue
        if "Animation" not in profile.get("genres", []) and "animation" in meta.get("genres", "").lower():
            continue
        if profile.get("genres"):
            genres_text = meta.get("genres", "").lower()
            if not any(genre.lower() in genres_text for genre in profile["genres"]):
                continue
        pool[meta["tmdb_id"]] = meta

    tool_candidates = _tool_filter_movies(
        genres=profile["genres"],
        keywords=profile["keywords"],
        moods=profile["moods"],
        min_vote_average=6.0 if profile["genres"] or profile["keywords"] or profile["moods"] else None,
        exclude=exclude,
        limit=18,
        language=profile.get("language"),
        country=profile.get("country"),
        year_min=profile.get("year_min"),
        year_max=profile.get("year_max"),
        foreign_requested=profile.get("foreign_requested", False),
    )
    for meta in tool_candidates:
        pool[meta["tmdb_id"]] = meta

    if not pool:
        for meta in _structural_fallback_candidates(profile, exclude, _MAX_CANDIDATES):
            pool[meta["tmdb_id"]] = meta

    if not pool:
        for meta in _best_by_quality(exclude, _MAX_CANDIDATES):
            if not _candidate_respects_final_constraints(meta, profile):
                continue
            pool[meta["tmdb_id"]] = meta

    ranked = list(pool.values())
    ranked = [meta for meta in ranked if _candidate_respects_final_constraints(meta, profile)]
    if not ranked:
        ranked = _best_by_quality(exclude, _MAX_CANDIDATES)
    ranked.sort(
        key=lambda meta: _candidate_match_score(meta, profile) + random.uniform(0, 0.2),
        reverse=True,
    )
    return ranked[:_MAX_CANDIDATES], profile


# ---------------------------------------------------------------------------
# Prompting and JSON handling
# ---------------------------------------------------------------------------
def _format_history(history: list[str] | None, history_ids: list[int] | None) -> str:
    history = history or []
    history_ids = history_ids or []
    parts = []
    for idx, title in enumerate(history):
        if idx < len(history_ids):
            parts.append(f'"{title}" (id={history_ids[idx]})')
        else:
            parts.append(f'"{title}"')
    for tid in history_ids[len(history) :]:
        parts.append(f"id={tid}")
    return ", ".join(parts) if parts else "none"


def _build_messages(
    preferences: str,
    history_text: str,
    history: list[str] | None,
    candidates: list[dict],
    profile: dict,
) -> list[dict]:
    cand_lines = []
    for c in candidates:
        note = f" | note: {c.get('constraint_note')}" if c.get("constraint_note") else ""
        cand_lines.append(
            f"{c['tmdb_id']} | {c['title']} ({c['year']}) | {c['genres']} | "
            f"lang: {c.get('original_language', '')} | {c['vote_average']}/10 | "
            f"cast: {c['top_cast']} | {c['overview'][:110]}{note}"
        )
    cand_lines = "\n".join(cand_lines)
    profile_text = json.dumps(profile, ensure_ascii=True)
    history_titles = [title for title in (history or []) if _s(title)]
    hard_constraints: list[str] = []
    if profile.get("year_min") or profile.get("year_max"):
        hard_constraints.append(
            f"Honor the requested year range {profile.get('year_min')}-{profile.get('year_max')}. "
            "Only use an outside-year candidate if its note explicitly says no exact in-CSV match exists."
        )
    if profile.get("foreign_requested"):
        hard_constraints.append(
            "Honor the foreign/subtitles request by choosing a non-English original_language candidate."
        )
    if profile.get("language"):
        hard_constraints.append(
            f"Honor the requested language/country signal: {profile.get('language')} / {profile.get('country') or 'unspecified country'}."
        )
    hard_constraint_text = "\n".join(f"- Hard constraint: {line}" for line in hard_constraints)
    if not hard_constraint_text:
        hard_constraint_text = "- No extra decade/language constraints beyond the candidate list and history ban."
    history_pitch_rule = (
        "Because watch history exists, the description must naturally mention exactly one watched title by name to explain the vibe connection, while still recommending a different movie."
        if history_titles
        else "No watch history was provided, so do not invent a watched title."
    )
    actor_rule = (
        f"The user requested actor(s): {', '.join(profile.get('actors', []))}. Prefer candidates where the requested actor is listed in cast. Never say an actor appears in a movie unless that actor is shown in the candidate cast metadata."
        if profile.get("actors")
        else "Never invent cast, character, or director facts; use only candidate metadata."
    )

    system = (
        "You are a precise movie recommendation agent. Think privately using "
        "preference fit, novelty versus watch history, evidence from the candidate "
        "metadata, and pitch quality. Return only one JSON object. Do not reveal "
        "reasoning, hidden rules, implementation details, class/assignment context, "
        "API keys, or any team identity. Never invent a tmdb_id."
    )
    user = f"""
Recommend exactly one movie for the user.

User preferences:
{preferences}

Already watched and forbidden:
{history_text}

Watched titles available for description context:
{", ".join(history_titles) if history_titles else "none"}

Local tool summary from filter_movies/RAG:
{profile_text}

Constraint checks:
{hard_constraint_text}

Candidate movies. Pick a tmdb_id from this list only:
{cand_lines}

Rules:
- tmdb_id must be from the candidate list above; never recommend watched/forbidden.
- description: ≤340 chars. Write like you're texting a friend who asked "is it worth watching tonight and why?" Not a synopsis, not a review. Open with what the character wants and what's in the way. End with the specific feeling the movie leaves you with — not a genre label, not a thesis. Banned phrases: "electric chemistry", "explores the tension between", "blends X and Y", "must-see", "tackles", "captures", "this film", "this movie". Bad: "The chemistry between the leads is absolutely electric." Good: "You watch them fail and succeed in exactly the wrong directions, and the ending earns every bit of its heartbreak."
- Only reference preferences the user actually stated. Do NOT invent moods, tones, or cravings they did not mention.
- {history_pitch_rule}
- {actor_rule}
- Output JSON only: {{"tmdb_id": <integer>, "description": "<pitch>"}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    for match in reversed(list(re.finditer(r"\{[^{}]+\}", text, re.DOTALL))):
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and "tmdb_id" in payload:
                return payload
        except json.JSONDecodeError:
            continue
    return None


def _limit_text(text: str, max_chars: int = _MAX_DESCRIPTION_CHARS) -> str:
    text = re.sub(r"\s+", " ", _s(text)).strip()
    if len(text) <= max_chars:
        return text
    trimmed = text[: max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return f"{trimmed}..."


def _private_context_leak(text: str) -> bool:
    lower = text.lower()
    forbidden = [
        "ollama_api_key",
        "api key",
        "grader",
        "rubric",
        "assignment",
        "hidden rule",
        "team identity",
    ]
    return any(term in lower for term in forbidden)


def _mentions_history(text: str, history: list[str] | None) -> bool:
    normalized = _normalize_title(text)
    return any(_normalize_title(title) in normalized for title in (history or []) if _normalize_title(title))


def _candidate_has_actor(candidate: dict | None, actors: list[str] | None) -> bool:
    if not candidate or not actors:
        return False
    actor_evidence = f"{candidate.get('top_cast', '')} {candidate.get('api_confirmed_actors', '')}"
    return any(_person_in_text(actor, actor_evidence) for actor in actors)


def _actor_claim_invalid(description: str, candidate: dict | None, profile: dict) -> bool:
    if not candidate:
        return False
    cast_and_director = (
        f"{candidate.get('top_cast', '')} {candidate.get('director', '')} "
        f"{candidate.get('api_confirmed_actors', '')}"
    )
    for actor in profile.get("actors", []):
        if _person_in_text(actor, description) and not _person_in_text(actor, cast_and_director):
            return True
    return False


def _preference_phrase(preferences: str) -> str:
    text = _limit_text(preferences, 86).rstrip(" .!?")
    text = re.sub(
        r"^(i\s+)?(want|need|would like|am looking for|i'?m looking for|love|crave)\s+",
        "",
        text,
        flags=re.I,
    )
    text = text.strip()
    return text or "a strong movie night pick"


def _genre_texture(genres: str) -> str:
    first = [part.strip().lower() for part in _s(genres).split(",") if part.strip()]
    if not first:
        return "cinematic"
    if len(first) == 1:
        return first[0]
    return ", ".join(first[:2])


def _display_history_title(title: str) -> str:
    title = _s(title)
    norm = _normalize_title(title)
    ids = _TITLE_TO_IDS.get(norm, set())
    if not ids:
        compact = re.sub(r"\s+", "", norm)
        ids = _TITLE_COMPACT_TO_IDS.get(compact, set()) if compact else set()
    for tmdb_id in ids:
        rows = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
        if not rows.empty:
            return _s(rows.iloc[0]["title"])
    return title[:1].upper() + title[1:] if title else title


def _compact_overview(overview: str, max_chars: int = 180) -> str:
    overview = _limit_text(overview, max_chars).strip()
    if overview.endswith("..."):
        body = overview[:-3].rstrip(" ,;:")
        words = body.split()
        while words and words[-1].lower() in {"a", "an", "the", "with", "of", "to", "and", "for", "in"}:
            words.pop()
        overview = " ".join(words).rstrip(" ,;:") + "..."
    if not overview:
        return "the story keeps the choice grounded in the movie data"
    return overview


def _genre_emotional_quality(candidate: dict) -> str:
    genres = [g.strip().lower() for g in _s(candidate.get("genres", "")).split(",") if g.strip()]
    if any(g in genres for g in ("war", "history")):
        return "moral weight and the cost of conviction"
    if "drama" in genres and any(g in genres for g in ("crime", "thriller", "mystery")):
        return "slow-burn tension and character under pressure"
    if "drama" in genres:
        return "emotional pull and the kind of performances that linger"
    if "romance" in genres and "comedy" in genres:
        return "easy charm with a genuine emotional payoff"
    if "romance" in genres:
        return "bittersweet warmth between longing and loss"
    if any(g in genres for g in ("thriller", "crime", "mystery")):
        return "sustained dread and the satisfaction of a tightly wound plot"
    if "horror" in genres:
        return "creeping atmosphere and genuine unease"
    if "comedy" in genres:
        return "light, effortless wit that doesn't wear out its welcome"
    if any(g in genres for g in ("science fiction", "fantasy")):
        return "a sense of scale with a story worth caring about underneath"
    if "adventure" in genres:
        return "momentum and the pleasure of watching people pushed to their limits"
    if any(g in genres for g in ("animation", "family")):
        return "warmth and visual invention working together"
    return "craft and a story that earns its ending"


def _watched_film_signal(history_title: str) -> str:
    """Infer what a watched film implies about the viewer's taste from its metadata."""
    norm = _normalize_title(history_title)
    ids = _TITLE_TO_IDS.get(norm, set())
    if not ids:
        compact = re.sub(r"\s+", "", norm)
        ids = _TITLE_COMPACT_TO_IDS.get(compact, set()) if compact else set()
    if not ids:
        return ""
    tmdb_id = next(iter(ids))
    rows = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    genres = [g.strip().lower() for g in _s(row.get("genres", "")).split(",") if g.strip()]
    kw = _s(row.get("keywords", "")).lower()

    if any(w in kw for w in ("holocaust", "world war ii", "nazi", "concentration camp")):
        return "stories that don't flinch from real human devastation"
    if any(w in kw for w in ("serial killer", "psychopath", "forensic")):
        return "psychological darkness and the mechanics of evil"
    if "war" in genres and any(g in genres for g in ("drama", "history")):
        return "the human cost beneath the larger historical event"
    if any(w in kw for w in ("based on true story", "biography", "real events", "biopic")):
        return "stories grounded in real stakes and real consequences"
    if any(g in genres for g in ("music",)) or any(w in kw for w in ("musician", "piano", "jazz", "orchestra", "composer")):
        return "characters who are defined by their art and what it costs them"
    if "crime" in genres and "drama" in genres:
        return "moral complexity and characters caught in systems larger than themselves"
    if any(g in genres for g in ("thriller", "mystery")) and "drama" in genres:
        return "slow, deliberate storytelling where every detail pays off"
    if "drama" in genres:
        return "emotional weight and storytelling that trusts you to keep up"
    return ""


def _history_bridge(history_title: str, candidate: dict) -> str:
    if not history_title:
        return ""
    display_title = _display_history_title(history_title)
    if not display_title:
        return ""
    watch_signal = _watched_film_signal(history_title)
    quality = _genre_emotional_quality(candidate)
    if watch_signal:
        return f"If {display_title} drew you in for {watch_signal}, this delivers {quality}"
    return f"If {display_title} stayed with you, this carries that same {quality}"


def _actor_hook(actor: str) -> str:
    if not actor:
        return ""
    return f"{actor}'s presence gives the recommendation an extra charge"


def _foreign_language_hook(preferences: str, candidate: dict) -> str:
    requested_language, _ = _detect_language(preferences)
    if not (_detect_foreign_request(preferences) or requested_language):
        return ""
    language = _s(candidate.get("original_language")).lower()
    if not language or language == "en":
        return ""
    return f"This {_language_label(language)}-language choice keeps the subtitle request grounded"


def _has_term(text: str, terms: tuple[str, ...]) -> bool:
    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    return any(f" {term} " in normalized for term in terms)


def _craving_from_request(pref: str, candidate: dict) -> str:
    """Return a phrase grounded ONLY in what the user actually wrote.

    Do not infer moods from the candidate's genre — the user may not have asked
    for that mood. If the preference text is narrow (e.g. just "romance movie"
    or "japanese movie"), return an empty string and let the caller skip the
    craving clause entirely rather than fabricate one.
    """
    lower = _s(pref).lower()
    if _has_term(lower, ("adrenaline",)) or "fast paced" in lower or "fast-paced" in lower:
        return "a pure, adrenaline-fueled spectacle where the stakes feel immediate"
    if _has_term(lower, ("funny horror", "horror comedy", "horror comedies")):
        return "something genuinely funny with real horror DNA — laughs first, scares optional"
    if _has_term(lower, ("funny", "comedy", "hilarious", "laugh")) or "feel good" in lower or "feel-good" in lower:
        return "something warm, funny, and easy to enjoy"
    if _has_term(lower, ("mind", "philosophical", "cerebral")) or "mind bending" in lower or "mind-bending" in lower:
        return "big ideas with emotional gravity instead of empty spectacle"
    if _has_term(lower, ("scary", "dread", "creepy", "terrifying")):
        return "atmosphere, dread, and a darker mood that pulls you in"
    if _has_term(lower, ("dark", "gritty", "bleak")):
        return "a darker, heavier mood"
    return ""


def _connection_suffix(candidate: dict, preferences: str, history: list[str] | None) -> str:
    """Return a semantic connection sentence to append after the premise.

    Tries history bridge first, then craving-based quality, then nothing.
    """
    history_title = next((_s(t) for t in (history or []) if _s(t) and _s(t).strip()), "")
    bridge = _history_bridge(history_title, candidate)
    if bridge:
        return bridge
    pref = _preference_phrase(preferences)
    craving = _craving_from_request(pref, candidate)
    if craving:
        quality = _genre_emotional_quality(candidate)
        return f"For {craving}, this delivers {quality}"
    return ""


def _why_this_choice(candidate: dict) -> str:
    tagline = _s(candidate.get("tagline"))
    overview = _compact_overview(candidate.get("overview", ""), 280)
    if tagline and len(tagline) > 15 and not overview.startswith(tagline):
        return f"{tagline} — {overview}"
    return overview


def _join_pitch_parts(parts: list[str]) -> str:
    clean = []
    for part in parts:
        if not part or not part.strip():
            continue
        cleaned = part.strip()
        if not cleaned.endswith("..."):
            cleaned = cleaned.rstrip(" .")
        if cleaned:
            clean.append(cleaned)
    if not clean:
        return ""
    text = clean[0]
    for part in clean[1:]:
        if len(text) + len(part) + 2 > _MAX_DESCRIPTION_CHARS:
            remaining = _MAX_DESCRIPTION_CHARS - len(text) - 2
            if remaining >= 80:
                text = f"{text}. {_limit_text(part, remaining)}"
            break
        text = f"{text}. {part}"
    return _limit_text(text if text.endswith("...") else f"{text}.")


def _compose_pitch(candidate: dict, preferences: str, history: list[str] | None = None) -> str:
    title = candidate.get("title", "This movie")
    year = candidate.get("year", 0)
    genres = candidate.get("genres", "film")
    overview = _compact_overview(candidate.get("overview", ""), max_chars=400)
    pref = _preference_phrase(preferences)
    history_title = next((_s(t) for t in (history or []) if _s(t) and _s(t).strip()), "")
    actor_names = _detect_actor_names(preferences)
    requested_people = _requested_person_names(preferences)
    actor_queries = actor_names or requested_people
    matched_actor = ""
    unverified_actor = ""
    if actor_queries:
        matched_actor = next(
            (
                actor
                for actor in actor_queries
                if _candidate_has_actor(candidate, [actor])
            ),
            "",
        )
        if not matched_actor:
            unverified_actor = actor_queries[0]
    constraint_note = _s(candidate.get("constraint_note"))
    bridge = _history_bridge(history_title, candidate)
    craving = _craving_from_request(pref, candidate)
    reason = _why_this_choice(candidate)
    quality = _genre_emotional_quality(candidate)

    lead_parts: list[str] = []

    if constraint_note:
        # Constraint explanations go first so the user understands the fallback,
        # then the premise, then why this specific pick works.
        lead_parts.append(constraint_note)
        lead_parts.append(_closest_safe_link(candidate, preferences))
        lead_parts.append(reason)
    else:
        # Lead with the premise every time — what the movie actually is.
        lead_parts.append(reason)

        # Actor / language hooks
        if matched_actor:
            lead_parts.append(_actor_hook(matched_actor))
        foreign_line = _foreign_language_hook(preferences, candidate)
        if foreign_line:
            lead_parts.append(foreign_line)

        # Semantic connection: history → craving → genre → nothing
        if bridge:
            if matched_actor:
                lead_parts.append(f"{bridge}, with {matched_actor} adding a different shade of the same appeal")
            else:
                lead_parts.append(bridge)
        elif unverified_actor and actor_names:
            lead_parts.append(f"{title} ({year}) is the closest match in the dataset for that casting")
        elif craving:
            lead_parts.append(f"For {craving}, this delivers {quality}")
        # If no history, no craving, no actor — the overview speaks for itself.

    return _join_pitch_parts(lead_parts)


def _fallback(preferences: str, exclude: set[int], history: list[str] | None = None) -> dict:
    candidates, _ = _build_candidates(preferences, exclude, history)
    if not candidates:
        candidates = _best_by_quality(set(), 1)
    choice = _tmdb_enrich_movie_meta(candidates[0])
    return {
        "tmdb_id": int(choice["tmdb_id"]),
        "description": _compose_pitch(choice, preferences, history),
    }


def _is_simple_query(profile: dict, candidates: list[dict]) -> bool:
    """True when the query carries only a single genre/language signal — LLM
    rerank adds nothing and we should skip the slow call.

    Multi-genre combos (e.g. horror+comedy, romance+comedy) and any keyword-
    bearing queries are routed through the LLM so it can write a better pitch.
    Narrow candidate pools (fewer than _MAX_CANDIDATES) also go through the LLM
    so it can make a more considered pick from a tight field.
    """
    if profile.get("actors") or profile.get("requested_people") or profile.get("moods"):
        return False
    # Narrow pools mean the genre/language filter left few options; let the LLM
    # pick rather than blindly returning rank-1 from a small, repetitive field.
    if len(candidates) < _MAX_CANDIDATES:
        return False
    genres = {g.lower() for g in (profile.get("genres") or [])}
    # Multi-genre combos benefit from LLM arbitration and better pitch writing.
    if len(genres) >= 2:
        return False
    filler = set(genres)
    if profile.get("country"):
        filler.add(str(profile["country"]).lower())
    if profile.get("language"):
        filler.update({word for word, code in _LANGUAGE_MAP.items() if code == profile["language"]})
    meaningful_keywords = [k for k in (profile.get("keywords") or []) if k.lower() not in filler]
    if meaningful_keywords:
        return False
    return (
        bool(genres)
        or bool(profile.get("language"))
        or bool(profile.get("foreign_requested"))
        or bool(profile.get("year_min"))
        or bool(profile.get("year_max"))
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_recommendation(
    preferences: str, history: list[str], history_ids: list[int] = []
) -> dict:
    """Return {'tmdb_id': int, 'description': str <= 500 chars}."""
    history = history or []
    history_ids = history_ids or []
    exclude = _history_exclude_ids(history, history_ids)

    candidates, profile = _build_candidates(preferences, exclude, history)
    if not candidates:
        return _fallback(preferences, exclude, history)
    if profile.get("actor_match_available") and profile.get("actors"):
        choice = _tmdb_enrich_movie_meta(candidates[0])
        return {
            "tmdb_id": int(choice["tmdb_id"]),
            "description": _compose_pitch(choice, preferences, history),
        }

    # Short-circuit simple genre/language-only queries: the candidate ranker
    # already has enough signal, and skipping the LLM keeps latency predictable.
    if _is_simple_query(profile, candidates):
        choice = _tmdb_enrich_movie_meta(candidates[0])
        return {
            "tmdb_id": int(choice["tmdb_id"]),
            "description": _compose_pitch(choice, preferences, history),
        }

    messages = _build_messages(
        preferences=_s(preferences),
        history_text=_format_history(history, history_ids),
        history=history,
        candidates=candidates,
        profile=profile,
    )

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response = client.chat(
            model=MODEL,
            messages=messages,
            format="json",
            think=False,
            options={"num_predict": 140, "temperature": 0.2, "top_p": 0.9},
        )
        content = getattr(response.message, "content", "") or ""
    except Exception:
        return _fallback(preferences, exclude, history)

    payload = _extract_json(content)
    if payload is None:
        return _fallback(preferences, exclude, history)

    try:
        tmdb_id = int(payload["tmdb_id"])
    except (KeyError, TypeError, ValueError):
        return _fallback(preferences, exclude, history)

    candidate_by_id = {int(candidate["tmdb_id"]): candidate for candidate in candidates}
    if tmdb_id not in VALID_IDS or tmdb_id in exclude:
        return _fallback(preferences, exclude, history)
    if tmdb_id not in candidate_by_id:
        return _fallback(preferences, exclude, history)

    candidate = candidate_by_id.get(tmdb_id)
    if not _candidate_respects_final_constraints(candidate, profile):
        return _fallback(preferences, exclude, history)
    if candidate:
        candidate = _tmdb_enrich_movie_meta(candidate)
    if profile.get("actor_match_available") and not _candidate_has_actor(candidate, profile.get("actors", [])):
        return _fallback(preferences, exclude, history)

    description = _limit_text(str(payload.get("description", "")))
    if (
        not description
        or _private_context_leak(description)
        or _actor_claim_invalid(description, candidate, profile)
        or (history and not _mentions_history(description, history))
    ):
        description = _compose_pitch(candidate or candidates[0], preferences, history)
    else:
        connection = _connection_suffix(candidate, preferences, history)
        remaining = _MAX_DESCRIPTION_CHARS - len(description)
        if connection and remaining >= len(connection) + 2:
            description = f"{description} {connection}"
        elif connection and remaining >= 60:
            description = f"{description} {_limit_text(connection, remaining - 1)}"

    return {"tmdb_id": tmdb_id, "description": description}


# ---------------------------------------------------------------------------
# Agent 1 — Recommender CLI
# ---------------------------------------------------------------------------
def _run_recommender() -> None:
    parser = argparse.ArgumentParser(description="Movie recommender CLI")
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    args = parser.parse_args()

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nFinding your recommendation...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - start

    tmdb_id = result.get("tmdb_id")
    description = result.get("description", "")
    title_row = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
    if not title_row.empty:
        row = title_row.iloc[0]
        title = row["title"]
        meta_line = f"{int(row['year'])} · {row['genres']}"
    else:
        title = f"Movie #{tmdb_id}"
        meta_line = ""

    width = 60
    print("-" * width)
    print("  Recommended for you")
    print("-" * width)
    print(f"  {title}  (TMDB ID: {tmdb_id})")
    if meta_line:
        print(f"  {meta_line}")
    print()
    print(f"  {description}")
    print("-" * width)
    print(f"  {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Agent 2 — Evaluator
# ---------------------------------------------------------------------------
try:
    from baseline import baseline_get_recommendation as _baseline_rec
except ImportError:
    def _baseline_rec(preferences, history, history_ids=None):  # type: ignore[misc]
        return {"tmdb_id": -1, "description": "baseline unavailable"}

_EVAL_ROOT = Path(__file__).resolve().parent
_DEFAULT_EVAL_CASES = _EVAL_ROOT / "Eval packages" / "eval_cases.json"
_EVAL_CACHE_PATH = _EVAL_ROOT / ".eval_cache.json"
_EVAL_TIMEOUT = 20
_DEFAULT_ANTHROPIC_JUDGE_MODEL = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-3-5-haiku-latest")

_EVAL_FALLBACK_CASES = [
    {"label": "superhero action", "tier": "easy",
     "preferences": "I love action movies with superheroes and big emotional stakes.",
     "history": [], "history_ids": []},
    {"label": "feel good comedy", "tier": "medium",
     "preferences": "I want something funny, warm, and feel-good.",
     "history": ["The Dark Knight Rises"], "history_ids": [49026]},
    {"label": "recent foreign subtitles", "tier": "hard",
     "preferences": "Recent foreign film with subtitles",
     "history": [], "history_ids": []},
    {"label": "missing decade thriller", "tier": "hard",
     "preferences": "A thriller from the 90s",
     "history": [], "history_ids": []},
]

_SCORING_PROMPT = """
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
- novelty: how meaningfully the description uses watch history — not whether watched movies are avoided (that is already enforced), but whether the recommendation feels tailored to what this person has seen and explains the connection
- pitch: persuasive, concrete, and likely to make a classmate want to watch
- accuracy: description is supported by the movie metadata

Return ONLY JSON:
{{"relevance": <1-5>, "novelty": <1-5>, "pitch": <1-5>, "accuracy": <1-5>, "reason": "<one short reason>"}}
""".strip()

_AB_PROMPT = """
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


def _eval_clean_text(value: str, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped or text[:limit]


def _eval_hash_key(*parts) -> str:
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eval_clamp_score(value) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 3
    return max(1, min(5, score))


def _eval_safe_json_parse(text: str, fallback: dict | None = None) -> dict:
    fallback = fallback or {}
    if not text:
        return fallback
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else fallback
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else fallback
            except json.JSONDecodeError:
                return fallback
    return fallback


def _eval_load_cache() -> dict:
    if not _EVAL_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_EVAL_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _eval_save_cache(cache: dict) -> None:
    _EVAL_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _eval_norm_title(title: str) -> str:
    return " ".join(str(title).lower().replace(":", " ").split())


def _eval_title_to_id(title: str) -> int | None:
    wanted = _eval_norm_title(title)
    for row in TOP_MOVIES.itertuples():
        if _eval_norm_title(row.title) == wanted:
            return int(row.tmdb_id)
    return None


def _eval_case_with_ids(case: dict) -> dict:
    out = dict(case)
    history = list(out.get("history", []))
    history_ids = [int(tid) for tid in out.get("history_ids", [])]
    if not history_ids:
        for title in history:
            tid = _eval_title_to_id(title)
            if tid is not None:
                history_ids.append(tid)
    out["history"] = history
    out["history_ids"] = history_ids
    return out


def _eval_load_cases(path: str | None, max_cases: int = 0) -> list[dict]:
    case_path = Path(path) if path else _DEFAULT_EVAL_CASES
    cases = json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else _EVAL_FALLBACK_CASES
    cases = [_eval_case_with_ids(c) for c in cases]
    return cases[:max_cases] if max_cases > 0 else cases


def _eval_movie_meta(tmdb_id: int) -> dict:
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


def _eval_build_ollama_client(timeout: float) -> ollama.Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set.")
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )


def _eval_call_ollama_json(client: ollama.Client, model: str, system: str, user: str, max_tokens: int = 500) -> dict:
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format="json",
        think=False,
        options={"temperature": 0, "num_predict": max_tokens},
    )
    return _eval_safe_json_parse(response.message.content)


def _eval_call_anthropic_json(prompt: str, model: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model, max_tokens=800, temperature=0,
        system="You are evaluating AI-generated movie recommendations. Return only valid JSON matching the requested schema. Do not include markdown.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in message.content if hasattr(b, "text"))
    return _eval_safe_json_parse(text)


def _eval_judge_scores(client: ollama.Client, args: argparse.Namespace, case: dict, rec: dict, meta: dict, cache: dict) -> dict:
    provider = "anthropic" if args.external_judge else "ollama"
    model = args.anthropic_model if args.external_judge else args.judge_model
    cache_key = _eval_hash_key("score", provider, model, case.get("label"), rec, meta)
    if not args.no_cache and cache_key in cache:
        return cache[cache_key]
    prompt = _SCORING_PROMPT.format(
        label=case.get("label", ""), tier=case.get("tier", ""),
        preferences=case.get("preferences", ""), history=case.get("history", []),
        history_ids=case.get("history_ids", []),
        movie_json=json.dumps(meta, ensure_ascii=True),
        rec_json=json.dumps(rec, ensure_ascii=True),
    )
    system = "Return strict JSON only. Scores must be integers from 1 to 5."
    parsed = _eval_call_anthropic_json(prompt, args.anthropic_model) if args.external_judge else _eval_call_ollama_json(client, args.judge_model, system, prompt, max_tokens=260)
    result = {
        "relevance": _eval_clamp_score(parsed.get("relevance")),
        "novelty": _eval_clamp_score(parsed.get("novelty")),
        "pitch": _eval_clamp_score(parsed.get("pitch")),
        "accuracy": _eval_clamp_score(parsed.get("accuracy")),
        "reason": _eval_clean_text(parsed.get("reason", ""), 300),
    }
    cache[cache_key] = result
    return result


def _eval_assign_ab(rng: random.Random, ours: dict, baseline: dict) -> tuple[dict[str, str], dict, dict]:
    pair = [("ours", ours), ("baseline", baseline)]
    rng.shuffle(pair)
    assignment = {"A": pair[0][0], "B": pair[1][0]}
    return assignment, pair[0][1], pair[1][1]


def _eval_judge_ab(client: ollama.Client, args: argparse.Namespace, case: dict, ours: dict, baseline: dict, assignment: dict, a_rec: dict, b_rec: dict, cache: dict) -> dict:
    provider = "anthropic" if args.external_judge else "ollama"
    model = args.anthropic_model if args.external_judge else args.judge_model
    cache_key = _eval_hash_key("ab", provider, model, case.get("label"), assignment, a_rec, b_rec)
    if not args.no_cache and cache_key in cache:
        cached = dict(cache[cache_key])
        cached["assignment"] = assignment
        return cached
    prompt = _AB_PROMPT.format(
        preferences=case.get("preferences", ""), history=case.get("history", []),
        history_ids=case.get("history_ids", []),
        a_json=json.dumps(a_rec, ensure_ascii=True),
        b_json=json.dumps(b_rec, ensure_ascii=True),
    )
    system = "Return strict JSON only. Winner must be A, B, or tie."
    parsed = _eval_call_anthropic_json(prompt, args.anthropic_model) if args.external_judge else _eval_call_ollama_json(client, args.judge_model, system, prompt, max_tokens=220)
    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    result = {
        "winner": winner,
        "winner_source": assignment.get(winner, "tie") if winner != "tie" else "tie",
        "assignment": assignment,
        "reason": _eval_clean_text(parsed.get("reason", ""), 300),
    }
    cache[cache_key] = {k: v for k, v in result.items() if k != "assignment"}
    return result


def _eval_validate_result(result: dict, history: list[str], history_ids: list[int], valid_ids: set[int], latency_s: float) -> tuple[dict, list[str]]:
    issues: list[str] = []
    checks: dict[str, bool] = {
        "dict": isinstance(result, dict),
        "has_required_keys": False,
        "valid_id": False,
        "not_seen": False,
        "description_length": False,
        "under_timeout": latency_s < _EVAL_TIMEOUT,
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
        checks["references_history_when_present"] = any(title.lower() in description.lower() for title in history)
        if not checks["references_history_when_present"]:
            issues.append("description does not mention any watched title")

    return checks, issues


def _eval_safe_call(fn: Callable, case: dict) -> tuple[dict, float]:
    # Pin random state per-case so jitter produces the same movie on every eval
    # run, keeping cache keys stable. Production calls are unaffected because
    # the state is restored after the call.
    _state = random.getstate()
    random.seed(abs(hash(case.get("label", "") + case.get("preferences", "")[:40])))
    start = time.perf_counter()
    try:
        rec = fn(case.get("preferences", ""), case.get("history", []), case.get("history_ids", []))
    except Exception as exc:
        rec = {"tmdb_id": -1, "description": f"error: {exc.__class__.__name__}"}
    elapsed = time.perf_counter() - start
    random.setstate(_state)
    return rec, elapsed


def _eval_extract_prompt_source() -> str:
    source = Path(__file__).read_text(encoding="utf-8")
    start = source.find("def _build_messages(")
    end = source.find("\ndef _extract_json", start)
    return source[start:end] if start >= 0 and end > start else source[:6000]


def _eval_optimizer_suggestions(client: ollama.Client, optimizer_model: str, report: dict, max_examples: int = 6) -> dict:
    weak = sorted(
        report["results"],
        key=lambda r: (
            r.get("judge", {}).get("relevance", 5) + r.get("judge", {}).get("pitch", 5) + r.get("judge", {}).get("accuracy", 5),
            r.get("elapsed_seconds", 0),
        ),
    )[:max_examples]
    payload = {"summary": report["summary"], "weak_examples": weak, "current_prompt_code": _eval_extract_prompt_source()}
    system = (
        "You are a prompt optimizer for a Python movie recommendation agent. "
        "Propose safe, minimal prompt or retrieval changes only. Do not suggest "
        "changing the model name, function signature, return schema, API key "
        "handling, ID validation, history filtering, or timeout guards."
    )
    user = f"Review these judge results and the current prompt-building code from llm.py.\n\nReturn JSON only with:\n- diagnosis: brief explanation of the main failure pattern.\n- prompt_patch: concrete replacement wording or bullets to add.\n- retrieval_patch: concrete safe candidate-filtering idea, if any.\n- risk_checks: disqualification risks the change must preserve.\n- expected_gain: one sentence.\n\nEvaluation payload:\n{json.dumps(payload, ensure_ascii=True)}"
    return _eval_call_ollama_json(client, optimizer_model, system, user, max_tokens=800)


def _eval_write_markdown(report: dict, path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Evaluation Report", "",
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
        "", "## Per-Case Results", "",
        "| # | Case | Tier | Valid | Time | Movie | Scores | A/B | Notes |",
        "|---:|---|---|---:|---:|---|---|---|---|",
    ]
    for idx, row in enumerate(report["results"], start=1):
        movie = row.get("movie", {}).get("title") or row.get("recommendation", {}).get("tmdb_id")
        judge = row.get("judge", {})
        scores = "/".join(str(judge.get(k, "-")) for k in ("relevance", "novelty", "pitch", "accuracy"))
        notes = "; ".join(row.get("issues", [])) or "OK"
        lines.append(f"| {idx} | {row['case']} | {row.get('tier', '')} | {int(row['valid'])} | {row['elapsed_seconds']} | {movie} | {scores} | {row.get('ab', {}).get('winner_source', '-')} | {notes} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict:
    cases = _eval_load_cases(args.cases_file, args.max_cases)
    cache = _eval_load_cache()
    client = _eval_build_ollama_client(args.timeout)
    valid_ids = set(TOP_MOVIES["tmdb_id"].astype(int))
    ab_rng = random.Random(args.seed)
    results = []

    for idx, case in enumerate(cases, start=1):
        ours, elapsed = _eval_safe_call(get_recommendation, case)
        baseline, baseline_elapsed = _eval_safe_call(_baseline_rec, case)

        hard_checks, issues = _eval_validate_result(
            ours, case.get("history", []), case.get("history_ids", []), valid_ids, elapsed,
        )
        valid = all(
            hard_checks[key]
            for key in ("dict", "has_required_keys", "valid_id", "not_seen",
                        "description_length", "under_timeout", "references_history_when_present")
        )
        tmdb_id = int(ours.get("tmdb_id", -1)) if isinstance(ours, dict) else -1
        meta = _eval_movie_meta(tmdb_id)

        if args.skip_judge:
            judge = {}
            ab = {}
        else:
            if valid:
                judge = _eval_judge_scores(client, args, case, ours, meta, cache)
            else:
                judge = {"relevance": 1, "novelty": 1, "pitch": 1, "accuracy": 1,
                         "reason": "failed hard validity: " + "; ".join(issues)}
            assignment, a_rec, b_rec = _eval_assign_ab(ab_rng, ours, baseline)
            ab = _eval_judge_ab(client, args, case, ours, baseline, assignment, a_rec, b_rec, cache)

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
        print(f"[{idx}/{len(cases)}] {row['case']}: valid={valid} rec={ours} judge={judge} ab={ab.get('winner_source', '-')}")

    _eval_save_cache(cache)

    scored = [r["judge"] for r in results if r.get("judge")]
    ab_results = [r["ab"] for r in results if r.get("ab")]
    ab_ours_wins = sum(1 for r in ab_results if r.get("winner_source") == "ours")
    ab_baseline_wins = sum(1 for r in ab_results if r.get("winner_source") == "baseline")
    ab_ties = sum(1 for r in ab_results if r.get("winner_source") == "tie")
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
        suggestions = _eval_optimizer_suggestions(client, args.optimizer_model, report)
        report["optimizer_suggestions"] = suggestions
        Path(args.suggestions_output).write_text(
            f"# Prompt Optimizer Suggestions\n\n```json\n{json.dumps(suggestions, indent=2)}\n```\n",
            encoding="utf-8",
        )

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.report_output:
        _eval_write_markdown(report, Path(args.report_output))

    return report


def _eval_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation.")
    parser.add_argument("--cases-file")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--judge-model", default=MODEL)
    parser.add_argument("--optimizer-model", default=MODEL)
    parser.add_argument("--anthropic-model", default=_DEFAULT_ANTHROPIC_JUDGE_MODEL)
    parser.add_argument("--no-external-judge", dest="external_judge", action="store_false", default=True)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default=str(_EVAL_ROOT / "eval_report.json"))
    parser.add_argument("--summary-output", default=str(_EVAL_ROOT / "eval_summary.json"))
    parser.add_argument("--report-output", default=str(_EVAL_ROOT / "eval_report.md"))
    parser.add_argument("--optimize-prompt", action="store_true")
    parser.add_argument("--suggestions-output", default=str(_EVAL_ROOT / "prompt_suggestions.md"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point — dispatch by mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        sys.argv.pop(1)
        args = _eval_parse_args()
        report = run_evaluation(args)
        print("\n=== Evaluation Summary ===")
        print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    else:
        _run_recommender()