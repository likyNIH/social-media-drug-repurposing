"""
reclassify_final.py

Re-evaluates every record in disease_ranking_progress.jsonl against the
current classify_match() logic, without re-querying arctic-shift (the raw
subreddit data collected during the run doesn't change over time, only the
relevance judgment does). Every matching-logic fix made during development
happened after the run had already collected some or all of its data, so
this reconciles the whole ~15,323-disease result set against one consistent,
fully-fixed version of the logic.

Leaves disease_ranking_progress.jsonl and disease_subreddit_ranking.csv
untouched as the historical/raw record. Writes:
  disease_ranking_reclassified.jsonl   (every record, re-judged)
  disease_subreddit_ranking_v2.csv     (final ranked matches only)

Usage: python reclassify_final.py
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import rank_disease_subreddits as m

DATA_DIR            = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
PROGRESS_FILE        = DATA_DIR / "disease_ranking_progress.jsonl"
DISEASES_CSV         = DATA_DIR / "all_diseases.csv"
RECLASSIFIED_JSONL   = DATA_DIR / "disease_ranking_reclassified.jsonl"
OUT_CSV_V2           = DATA_DIR / "disease_subreddit_ranking_v2.csv"

FIELDNAMES = ["rank", "gard_id", "disease_name", "subreddit", "subscribers", "num_posts",
              "num_comments", "description", "matched_query", "match_type", "confidence"]


def load_synonyms_by_id() -> dict[str, list[str]]:
    synonyms_by_id = {}
    with open(DISEASES_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            synonyms_by_id[row["gardId"]] = m.parse_synonyms(row.get("synonyms", ""))
    return synonyms_by_id


def main():
    synonyms_by_id = load_synonyms_by_id()

    stats = {"unchanged_match": 0, "unchanged_no_match": 0, "demoted": 0, "eliminated": 0, "promoted": 0}
    reclassified = []

    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            if not rec.get("subreddit"):
                stats["unchanged_no_match"] += 1
                reclassified.append(rec)
                continue

            synonyms = synonyms_by_id.get(rec["gard_id"], [])
            new_mt = m.classify_match(rec["subreddit"], rec["description"], rec["disease_name"], synonyms)

            if new_mt is None:
                stats["eliminated"] += 1
                rec = {
                    **rec,
                    "subreddit": None, "subscribers": 0, "num_posts": 0, "num_comments": 0,
                    "description": "", "matched_query": None, "match_type": None, "confidence": None,
                }
            else:
                new_confidence = "high" if new_mt in m.CONFIDENT_MATCH_TYPES else "low"
                old_confidence = rec.get("confidence")
                if new_confidence == old_confidence:
                    stats["unchanged_match"] += 1
                elif new_confidence == "low":
                    stats["demoted"] += 1
                else:
                    stats["promoted"] += 1
                rec = {**rec, "match_type": new_mt, "confidence": new_confidence}

            reclassified.append(rec)

    with open(RECLASSIFIED_JSONL, "w", encoding="utf-8") as f:
        for rec in reclassified:
            f.write(json.dumps(rec) + "\n")

    matches = [r for r in reclassified if r.get("subreddit")]
    matches.sort(key=lambda x: (x.get("confidence") != "high", -x["subscribers"]))

    with open(OUT_CSV_V2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rank, r in enumerate(matches, 1):
            writer.writerow({"rank": rank, **{k: r.get(k) for k in FIELDNAMES if k != "rank"}})

    high_count = sum(1 for r in matches if r.get("confidence") == "high")
    print(f"Reclassified {len(reclassified)} records")
    print(f"  Had no match before or after: {stats['unchanged_no_match']}")
    print(f"  Had a match, confidence unchanged: {stats['unchanged_match']}")
    print(f"  Demoted high -> low: {stats['demoted']}")
    print(f"  Promoted low -> high: {stats['promoted']}")
    print(f"  Eliminated (was a match, now none): {stats['eliminated']}")
    print()
    print(f"Final: {len(matches)} total matches ({high_count} high-confidence, {len(matches) - high_count} low-confidence)")
    print(f"Ranked CSV saved to: {OUT_CSV_V2}")
    print(f"Full reclassified JSONL saved to: {RECLASSIFIED_JSONL}")


if __name__ == "__main__":
    main()
