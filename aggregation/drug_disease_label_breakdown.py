"""
drug_disease_label_breakdown.py

Companion to aggregate_drug_signals.py's repurposing_candidates.csv, which
groups by normalized drug name alone and collapses label-status counts
across every disease a drug appears in. This script re-groups the same
already-extracted, already-cleaned mentions (drug_signals_master.csv) by
(drug, disease) pair instead, so on/off-label counts can be inspected per
disease rather than summed across all of them. No re-extraction needed --
label_status and disease_name were already captured per mention.

Outputs:
  drug_disease_label_breakdown.csv          one row per (drug, disease) pair
  drug_disease_label_breakdown_compact.csv  same, filtered to pairs with at
    least one explicit off_label mention -- drops the ~96% of pairs that
    are unclear-only noise, for a view that's actually scannable

Usage: python drug_disease_label_breakdown.py
"""

import csv
from collections import defaultdict

import scrape_rare_disease_subreddits as scraper

DRUG_SIGNALS_DIR = scraper.OUT_DIR / "drug_signals"
MASTER_PATH = DRUG_SIGNALS_DIR / "drug_signals_master.csv"
OUT_PATH = DRUG_SIGNALS_DIR / "drug_disease_label_breakdown.csv"
OUT_COMPACT_PATH = DRUG_SIGNALS_DIR / "drug_disease_label_breakdown_compact.csv"

JUNK_NAMES = {"", "unclear", "n/a", "none", "na", "not applicable"}


def main():
    groups = defaultdict(lambda: {
        "mentions": 0, "off_label": 0, "on_label": 0, "unclear_label": 0,
        "positive_outcome": 0, "negative_outcome": 0,
        "posts": set(), "quotes": [],
    })

    with open(MASTER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("normalized_name_clean") or "").strip()
            disease = (row.get("disease_name") or "").strip()
            if name in JUNK_NAMES or not disease:
                continue

            g = groups[(name, disease)]
            g["mentions"] += 1
            g["posts"].add(row.get("post_id"))

            label = row.get("label_status")
            if label == "off_label":
                g["off_label"] += 1
            elif label == "on_label":
                g["on_label"] += 1
            else:
                g["unclear_label"] += 1

            progression = row.get("progression_after_treatment")
            final_outcome = row.get("final_outcome")
            if progression == "helped_or_improved" or final_outcome == "improved":
                g["positive_outcome"] += 1
            elif progression in ("did_not_help", "worsened") or final_outcome == "worsened_or_progressed":
                g["negative_outcome"] += 1

            quote = row.get("quote")
            if quote and len(g["quotes"]) < 3:
                g["quotes"].append(quote)

    rows_out = []
    for (name, disease), g in groups.items():
        rows_out.append({
            "normalized_name": name,
            "disease_name": disease,
            "total_mentions": g["mentions"],
            "distinct_posts": len(g["posts"]),
            "off_label_mentions": g["off_label"],
            "on_label_mentions": g["on_label"],
            "unclear_label_mentions": g["unclear_label"],
            "positive_outcome_mentions": g["positive_outcome"],
            "negative_outcome_mentions": g["negative_outcome"],
            "example_quotes": " | ".join(g["quotes"]),
        })

    def sort_key(r):
        positive_ratio = r["positive_outcome_mentions"] / r["total_mentions"] if r["total_mentions"] else 0
        return (r["normalized_name"], -r["off_label_mentions"], -r["distinct_posts"], -positive_ratio)

    rows_out.sort(key=sort_key)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"{len(rows_out)} (drug, disease) pairs -> {OUT_PATH}")

    compact = [r for r in rows_out if r["off_label_mentions"] > 0]
    with open(OUT_COMPACT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(compact[0].keys()))
        writer.writeheader()
        writer.writerows(compact)
    print(f"{len(compact)} pairs with >=1 off-label mention -> {OUT_COMPACT_PATH}")


if __name__ == "__main__":
    main()
