"""
backfill_missing_comments.py

Re-fetches comments for posts where Reddit reports num_comments > 0 but our
scrape recorded an empty "comments" field -- the signature of a fetch that
failed (network timeout, connection error, etc.) rather than a post that
genuinely has no comments. Confirmed via check: ~900 of 163,984 scraped
posts (0.55%) match this pattern, concentrated in r/Narcolepsy which went
through a crash/resume cycle during the original scrape.

Streams each subreddit's file line by line (some are 450MB+) rather than
loading it into memory, rewrites to a .tmp file, and only replaces the
original via atomic rename once every line has been written -- same
crash-safety pattern used by the original scraper.

A small fraction of matches will be posts where every comment was already
"[deleted]"/"[removed]" by scrape time -- those are correctly still empty
after backfilling (nothing to recover, not a bug) and are reported
separately from ones that stayed empty due to a repeat fetch failure.

Usage: python backfill_missing_comments.py
"""

import json
import time
from pathlib import Path

import scrape_rare_disease_subreddits as target

OUT_DIR = target.OUT_DIR


def needs_backfill(doc: dict) -> bool:
    return (doc.get("num_comments") or 0) > 0 and not doc.get("comments")


def backfill_subreddit(subreddit: str) -> dict:
    path = OUT_DIR / f"{subreddit}_raw.jsonl"
    tmp_path = OUT_DIR / f"{subreddit}_raw.jsonl.backfill.tmp"

    stats = {"total": 0, "candidates": 0, "recovered": 0, "still_empty": 0}

    with open(path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                continue

            stats["total"] += 1

            if needs_backfill(doc):
                stats["candidates"] += 1
                comments_text = target.fetch_comments_for_post(doc["id"])
                time.sleep(0.5)

                if comments_text:
                    doc["comments"] = comments_text
                    title = doc.get("title", "")
                    selftext = doc.get("selftext", "")
                    doc["full_text"] = f"{title} {selftext} {comments_text}".strip()
                    stats["recovered"] += 1
                else:
                    stats["still_empty"] += 1

            fout.write(json.dumps(doc) + "\n")

    tmp_path.rename(path)
    return stats


def main():
    print(f"Backfilling missing comments across {len(target.COMMUNITIES)} communities\n")

    grand_total = {"total": 0, "candidates": 0, "recovered": 0, "still_empty": 0}
    per_subreddit = []

    for i, community in enumerate(target.COMMUNITIES, 1):
        subreddit = community["subreddit"]
        print(f"[{i}/{len(target.COMMUNITIES)}] r/{subreddit}")
        stats = backfill_subreddit(subreddit)
        per_subreddit.append((subreddit, stats))
        for k in grand_total:
            grand_total[k] += stats[k]
        if stats["candidates"]:
            print(f"  {stats['candidates']} candidates -> {stats['recovered']} recovered, "
                  f"{stats['still_empty']} still empty (genuinely deleted/removed or fetch failed again)")
        else:
            print("  no candidates, nothing to do")

    print("\n=== Summary ===")
    print(f"{'subreddit':<24}{'candidates':>12}{'recovered':>12}{'still empty':>14}")
    for sub, stats in per_subreddit:
        print(f"{sub:<24}{stats['candidates']:>12,}{stats['recovered']:>12,}{stats['still_empty']:>14,}")
    print()
    print(f"TOTAL: {grand_total['candidates']:,} candidates, "
          f"{grand_total['recovered']:,} recovered, "
          f"{grand_total['still_empty']:,} still empty "
          f"(out of {grand_total['total']:,} posts scanned)")


if __name__ == "__main__":
    main()
