"""
check_scrape_progress.py

Reports progress on scrape_rare_disease_subreddits.py: which communities are
fully scraped, which one is currently in progress and how far along (against
its approximate post count from disease_subreddit_ranking_v2.csv), and which
are still queued. Safe to run at any time, including while the scraper is
still running -- it only reads.

Usage: python check_scrape_progress.py
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import scrape_rare_disease_subreddits as target

DATA_DIR = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
OUT_DIR  = target.OUT_DIR
V2_CSV   = DATA_DIR / "disease_subreddit_ranking_v2.csv"


def load_expected_post_counts() -> dict[str, int]:
    counts = {}
    with open(V2_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["subreddit"]] = int(row["num_posts"] or 0)
    return counts


def main():
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    expected = load_expected_post_counts()

    done, in_progress, queued = [], [], []
    for community in target.COMMUNITIES:
        sub = community["subreddit"]
        final_file = OUT_DIR / f"{sub}_raw.jsonl"
        tmp_file = OUT_DIR / f"{sub}_raw.jsonl.tmp"

        if final_file.exists():
            with open(final_file, encoding="utf-8") as f:
                actual = sum(1 for _ in f)
            done.append((sub, actual))
        elif tmp_file.exists():
            with open(tmp_file, encoding="utf-8") as f:
                actual = sum(1 for _ in f)
            in_progress.append((sub, actual, expected.get(sub, 0)))
        else:
            queued.append(sub)

    total = len(target.COMMUNITIES)
    print(f"{len(done)}/{total} communities fully scraped\n")

    if done:
        print("Done:")
        for sub, n in done:
            print(f"  r/{sub}: {n:,} posts saved")
        print()

    if in_progress:
        print("In progress:")
        for sub, actual, exp in in_progress:
            pct = f"{100 * actual / exp:.0f}%" if exp else "?"
            print(f"  r/{sub}: {actual:,} posts so far (~{exp:,} expected, {pct})")
        print()

    if queued:
        print("Still queued:")
        for sub in queued:
            print(f"  r/{sub}")


if __name__ == "__main__":
    main()
