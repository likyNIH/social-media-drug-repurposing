"""
rebuild_repurposing_candidates.py

Rebuilds the ranked repurposing-candidate list from the CORRECTED data
(drug_signals_master_reclassified.csv -- after both the top-20 curated
reference and the disease-first openFDA reference passes), instead of the
raw LLM extraction (which is what aggregate_drug_signals.py's
repurposing_candidates.csv still reflects). Re-running aggregate_drug_signals.py
itself would rebuild drug_signals_master.csv from the raw per-community
files and lose all the correction work, so this reads the already-corrected
master directly.

Same ranking logic as aggregate_drug_signals.py's build_candidates()
(off-label mentions first, then cross-disease corroboration, distinct
posts, positive-outcome ratio), plus:
  - the same generic/class-term junk filter (reused from aggregate_drug_
    signals.py, derived from the extraction's own drug_or_treatment_class
    field, not hand-typed)
  - a label_status_source breakdown per drug, so it's visible how much of
    a candidate's off-label signal came from raw LLM extraction vs. the
    curated/disease-first correction passes
  - a likely_repurposing_relevant flag: "no" for substances confirmed to
    have zero FDA-approved human indication for anything (fenbendazole,
    psilocybin) -- these are real off-label mentions but not genuine
    repurposing candidates in the pharma sense (no existing safety/
    regulatory approval to build a new-indication case on), so they
    shouldn't compete for rank against drugs like carbamazepine-alternatives
    for trigeminal neuralgia. Not silently dropped -- just flagged, since
    the underlying mentions are still real data.

Outputs:
  repurposing_candidates_reclassified.csv

Usage: python rebuild_repurposing_candidates.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper
from aggregate_drug_signals import JUNK_NAMES, class_level_terms

DRUG_SIGNALS_DIR = scraper.OUT_DIR / "drug_signals"
RECLASSIFIED_PATH = DRUG_SIGNALS_DIR / "drug_signals_master_reclassified.csv"
OUT_PATH = DRUG_SIGNALS_DIR / "repurposing_candidates_reclassified.csv"

# Confirmed via direct openFDA check: no human drug label exists at all for
# these (fenbendazole = veterinary-only; psilocybin = no approved human
# indication, Schedule I). Real off-label mentions, but not genuine
# repurposing candidates -- flagged, not dropped.
NO_HUMAN_APPROVAL = {"fenbendazole", "psilocybin"}


def load_rows() -> list[dict]:
    with open(RECLASSIFIED_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_candidates(rows: list[dict]) -> list[dict]:
    excluded_terms = JUNK_NAMES | class_level_terms(rows)

    groups = defaultdict(lambda: {
        "mentions": 0, "off_label": 0, "on_label": 0, "unclear_label": 0,
        "positive_outcome": 0, "negative_outcome": 0,
        "diseases": set(), "conditions": set(), "posts": set(), "quotes": [],
        "off_label_source_counts": defaultdict(int),
    })

    for row in rows:
        name = row["normalized_name_clean"]
        if name in excluded_terms:
            continue
        g = groups[name]
        g["mentions"] += 1

        label = row["label_status"]
        if label == "off_label":
            g["off_label"] += 1
        elif label == "on_label":
            g["on_label"] += 1
        else:
            g["unclear_label"] += 1

        if label == "off_label":
            g["off_label_source_counts"][row.get("label_status_source", "llm_extraction")] += 1

        progression = row["progression_after_treatment"]
        final_outcome = row["final_outcome"]
        if progression == "helped_or_improved" or final_outcome == "improved":
            g["positive_outcome"] += 1
        elif progression in ("did_not_help", "worsened") or final_outcome == "worsened_or_progressed":
            g["negative_outcome"] += 1

        g["diseases"].add(row["disease_name"])
        if row["condition_treated_clean"]:
            g["conditions"].add(row["condition_treated_clean"])
        g["posts"].add(row["post_id"])
        if row.get("quote") and len(g["quotes"]) < 5:
            g["quotes"].append(row["quote"])

    candidates = []
    for name, g in groups.items():
        candidates.append({
            "normalized_name": name,
            "likely_repurposing_relevant": "no" if name in NO_HUMAN_APPROVAL else "yes",
            "total_mentions": g["mentions"],
            "distinct_posts": len(g["posts"]),
            "distinct_diseases": len(g["diseases"]),
            "diseases": "; ".join(sorted(g["diseases"])),
            "conditions_treated": "; ".join(sorted(g["conditions"])[:15]),
            "off_label_mentions": g["off_label"],
            "on_label_mentions": g["on_label"],
            "unclear_label_mentions": g["unclear_label"],
            "off_label_from_llm_extraction": g["off_label_source_counts"].get("llm_extraction", 0),
            "off_label_from_curated_or_disease_reference": (
                g["off_label"] - g["off_label_source_counts"].get("llm_extraction", 0)
            ),
            "positive_outcome_mentions": g["positive_outcome"],
            "negative_outcome_mentions": g["negative_outcome"],
            "example_quotes": " | ".join(g["quotes"]),
        })

    def sort_key(c):
        positive_ratio = c["positive_outcome_mentions"] / c["total_mentions"] if c["total_mentions"] else 0
        relevance_penalty = 0 if c["likely_repurposing_relevant"] == "yes" else 1
        return (relevance_penalty, -c["off_label_mentions"], -c["distinct_diseases"], -c["distinct_posts"], -positive_ratio)

    candidates.sort(key=sort_key)
    return candidates


def main():
    print(f"Loading corrected data from {RECLASSIFIED_PATH}...")
    rows = load_rows()
    print(f"{len(rows):,} total drug mentions loaded\n")

    print("Rebuilding repurposing candidate rankings from corrected labels...")
    candidates = build_candidates(rows)

    fieldnames = list(candidates[0].keys())
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"{len(candidates):,} distinct normalized drug names -> {OUT_PATH}\n")

    off_label_candidates = [c for c in candidates if c["off_label_mentions"] > 0]
    multi_disease = [c for c in candidates if c["distinct_diseases"] > 1]
    flagged = [c for c in candidates if c["likely_repurposing_relevant"] == "no"]
    print(f"{len(off_label_candidates):,} drugs have >=1 off-label mention")
    print(f"{len(multi_disease):,} drugs corroborated across multiple diseases")
    print(f"{len(flagged):,} drugs flagged as not repurposing-relevant (no human FDA approval at all)")

    print("\nTop 25 candidates:")
    print(f"{'drug':<32}{'off-label':>10}{'diseases':>10}{'posts':>8}{'positive':>10}")
    for c in candidates[:25]:
        print(f"{c['normalized_name']:<32}{c['off_label_mentions']:>10}{c['distinct_diseases']:>10}"
              f"{c['distinct_posts']:>8}{c['positive_outcome_mentions']:>10}")


if __name__ == "__main__":
    main()
