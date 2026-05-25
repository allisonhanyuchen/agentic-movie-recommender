# Evaluation Report

- Generator: `gemma4:31b-cloud`
- Judge: `claude-haiku-4-5-20251001`
- External judge: `True`
- Cases: `8`
- Validity: `8/8` (`100.0%`)
- Mean relevance: `4.25`
- Mean novelty: `4.875`
- Mean pitch: `3.375`
- Mean accuracy: `4`
- A/B wins/losses/ties: `6/2/0`
- A/B win rate: `75.0%`

## Per-Case Results

| # | Case | Tier | Valid | Time | Movie | Scores | A/B | Notes |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | single_genre_action | easy | 1 | 2.528 | Avengers: Infinity War | 5/5/4/5 | baseline | OK |
| 2 | single_genre_family | easy | 1 | 2.658 | Coco | 5/5/4/5 | ours | OK |
| 3 | single_genre_romcom | medium | 1 | 12.468 | Solo Mio | 4/4/3/4 | ours | OK |
| 4 | combo_scifi_dark | medium | 1 | 3.033 | Logan | 3/5/4/4 | ours | OK |
| 5 | combo_funny_horror | medium | 1 | 2.595 | Ready or Not | 4/5/4/3 | ours | OK |
| 6 | mood_uplifting | medium | 1 | 2.921 | About Time | 5/5/4/5 | baseline | OK |
| 7 | hard_french_grief | hard | 1 | 4.848 | Last Summer | 3/5/2/2 | ours | OK |
| 8 | era_recent_foreign | hard | 1 | 0.247 | My Fault | 5/5/2/4 | ours | OK |
