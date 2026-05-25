"""
Run this file to check that your get_recommendation() implementation satisfies
all grading requirements before you submit.

Usage:
    python test.py
"""

import ast
import importlib.metadata
import json
import os
import re
import sys
import time

from llm import TOP_MOVIES, get_recommendation

VALID_IDS = set(TOP_MOVIES["tmdb_id"].astype(int))
TIMEOUT_SECONDS = 20

TESTS = [
    {
        "label": "basic recommendation",
        "preferences": "I love action movies with superheroes.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "recommendation with watch history",
        "preferences": "I want something funny and feel-good.",
        "history": ["The Dark Knight Rises"],
        "history_ids": [49026],
    },
]


def check_requirements() -> bool:
    """Verify every non-stdlib import in llm.py is listed in requirements.txt."""
    print("\n--- requirements check ---")

    # Collect all top-level module names imported by llm.py
    with open(os.path.join(os.path.dirname(__file__), "llm.py")) as f:
        tree = ast.parse(f.read())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute import only
                imported.add(node.module.split(".")[0])

    # Parse requirements.txt — normalise names (lowercase, - → _)
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(req_path) as f:
            req_names = {
                re.split(r"[>=<!;\s]", line.strip())[0].lower().replace("-", "_")
                for line in f
                if line.strip() and not line.startswith("#")
            }
    except FileNotFoundError:
        print("  FAIL: requirements.txt not found")
        return False

    # stdlib + builtins (sys.stdlib_module_names available since Python 3.10)
    stdlib = set(sys.stdlib_module_names) | set(sys.builtin_module_names)

    # Local modules — sibling .py files in the submission directory don't belong
    # in requirements.txt (they aren't PyPI packages). Treat them like stdlib.
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    local_modules = {
        os.path.splitext(f)[0]
        for f in os.listdir(submission_dir)
        if f.endswith(".py") and not f.startswith("_")
    }

    # Map import name → distribution name(s) for everything currently installed
    pkg_dist = importlib.metadata.packages_distributions()

    missing = []
    for mod in sorted(imported):
        if mod in stdlib or mod == "__future__" or mod in local_modules:
            continue
        dists = pkg_dist.get(mod, [])
        if not dists:
            missing.append((mod, None))
            continue
        covered = any(
            d.lower().replace("-", "_") in req_names for d in dists
        )
        if not covered:
            missing.append((mod, dists))

    if missing:
        print("  FAIL: imports not covered by requirements.txt:")
        for mod, dists in missing:
            if dists:
                print(f"    '{mod}' is provided by {dists} — add one to requirements.txt")
            else:
                print(f"    '{mod}' is not installed and not in requirements.txt")
        return False

    print(f"  PASS: all imports are stdlib or listed in requirements.txt")
    return True


def run_test(test: dict) -> bool:
    label = test["label"]
    print(f"\n--- {label} ---")
    print(f"  preferences : {test['preferences']}")
    print(f"  history     : {test['history']}")

    history_id_set = set(test["history_ids"])

    start = time.perf_counter()
    try:
        result = get_recommendation(
            test["preferences"], test["history"], test["history_ids"]
        )
    except json.JSONDecodeError as e:
        print(f"  FAIL: LLM returned invalid JSON — {e}")
        return False
    except Exception as e:
        print(f"  FAIL: get_recommendation() raised an exception — {e}")
        return False
    elapsed = time.perf_counter() - start

    # Must return a dict
    if not isinstance(result, dict):
        print(f"  FAIL: return value is {type(result).__name__}, expected dict")
        return False

    # Must contain tmdb_id
    if "tmdb_id" not in result:
        print(f"  FAIL: result is missing 'tmdb_id' key — got {list(result.keys())}")
        return False

    # Must contain description
    if "description" not in result:
        print(f"  FAIL: result is missing 'description' key — got {list(result.keys())}")
        return False

    tmdb_id = int(result["tmdb_id"])

    # tmdb_id must be in the candidate list
    if tmdb_id not in VALID_IDS:
        print(f"  FAIL: tmdb_id {tmdb_id} is not in the candidate list {sorted(VALID_IDS)}")
        return False

    # Must not recommend something already watched
    if tmdb_id in history_id_set:
        print(f"  FAIL: tmdb_id {tmdb_id} is already in the user's watch history")
        return False

    # Must respond within the timeout
    if elapsed > TIMEOUT_SECONDS:
        print(f"  FAIL: took {elapsed:.1f}s (limit is {TIMEOUT_SECONDS}s)")
        return False

    description = str(result.get("description", ""))
    print(f"  PASS ({elapsed:.2f}s)")
    print(f"  tmdb_id    : {tmdb_id}")
    print(f"  description: {description[:120]}{'...' if len(description) > 120 else ''}")
    return True


def main():
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY is not set.")
        print("Run:  OLLAMA_API_KEY=your_key_here python test.py")
        sys.exit(1)

    print(f"Candidate pool: {len(VALID_IDS)} movies (IDs: {sorted(VALID_IDS)})")

    req_ok = check_requirements()
    results = [run_test(t) for t in TESTS]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Requirements: {'OK' if req_ok else 'FAIL'}")
    print(f"Results: {passed}/{total} tests passed")
    if not req_ok or passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
