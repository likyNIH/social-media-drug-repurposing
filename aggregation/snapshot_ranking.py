"""
snapshot_ranking.py

Builds a ranked CSV snapshot from whatever has been written so far to
disease_ranking_progress.jsonl. Safe to run at any time, including while
rank_disease_subreddits.py is still running/appending to the progress file —
this script only reads it.

Usage: python snapshot_ranking.py
"""

import csv
import json
import os
from pathlib import Path

DATA_DIR         = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
PROGRESS_FILE    = DATA_DIR / "disease_ranking_progress.jsonl"
SNAPSHOT_CSV     = DATA_DIR / "disease_subreddit_ranking_snapshot.csv"
MIN_SUBSCRIBERS  = 200


def main():
    if not PROGRESS_FILE.exists():
        print(f"No progress file yet at {PROGRESS_FILE}")
        return

    total = 0
    matches = []
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line may be a partial write if job is mid-flush
            if rec.get("subscribers", 0) >= MIN_SUBSCRIBERS:
                matches.append(rec)

    matches.sort(key=lambda x: (x.get("confidence") != "high", -x["subscribers"]))

    with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "gard_id", "disease_name", "subreddit", "subscribers", "num_posts",
                      "num_comments", "description", "matched_query", "match_type", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(matches, 1):
            writer.writerow({"rank": rank, **r})

    high = [r for r in matches if r.get("confidence") == "high"]
    low = [r for r in matches if r.get("confidence") != "high"]
    print(f"{total} diseases processed so far, {len(matches)} with a matching subreddit "
          f"({len(high)} high-confidence, {len(low)} low-confidence/umbrella)")
    print(f"Snapshot saved to: {SNAPSHOT_CSV}\n")

    print("Top 20 high-confidence matches so far:")
    for r in high[:20]:
        posts = r.get('num_posts', 0) or 0
        print(f"  {r['subscribers']:>7,} subs  {posts:>6,} posts  r/{r['subreddit']:<25} {r['disease_name']}  ({r.get('match_type')})")

    if low:
        print("\nTop 10 low-confidence/umbrella matches (verify manually):")
        for r in low[:10]:
            print(f"  {r['subscribers']:>7,}  r/{r['subreddit']:<25} {r['disease_name']}  (via {r.get('matched_query')!r})")


if __name__ == "__main__":
    main()
