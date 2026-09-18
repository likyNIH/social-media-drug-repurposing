"""
aggregate_drug_signals.py

Final step: combines all 20 communities' per-disease drug_signals.csv
files into one master table, cleans known data-quality issues found
during the extraction run monitoring (label_status/progression_after_
treatment/mention_type occasionally containing free-text or
cross-contaminated enum values from the wrong field -- e.g. "hopeful",
or "research_or_trial_mention" appearing in label_status instead of
mention_type), and builds a ranked repurposing-candidate list.

Grouped by normalized drug name (not drug+condition): condition_treated
is free text ("fatigue" vs "tiredness" vs "excessive sleepiness" for the
same thing), so grouping strictly on it would fragment the same real
signal across near-duplicate phrasings. Distinct diseases and conditions
mentioned are still preserved per drug for review, just not used as a
hard grouping key.

Ranking prioritizes, in order:
  1. Explicit off-label mentions (the strongest direct textual signal)
  2. Distinct diseases corroborating the same drug (a drug showing up
     across multiple different rare disease communities, none of which
     are its primary approved indication, is itself a repurposing signal)
  3. Distinct posts (independent patient corroboration within a disease)
  4. Positive reported outcome ratio

This does NOT cross-reference an external drug-indication database (e.g.
DrugBank/RxNorm) -- "off-label" here means the TEXT explicitly said so or
described an atypical use, not a computed mismatch against approved
indications. This is a prioritized list for human review, not a
validated repurposing determination.

Outputs:
  drug_signals_master.csv       every cleaned mention, one row each
  repurposing_candidates.csv    ranked, one row per normalized drug name

Usage: python aggregate_drug_signals.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper

DRUG_SIGNALS_DIR = scraper.OUT_DIR / "drug_signals"
OUT_MASTER = DRUG_SIGNALS_DIR / "drug_signals_master.csv"
OUT_CANDIDATES = DRUG_SIGNALS_DIR / "repurposing_candidates.csv"

VALID_LABEL_STATUS = {"on_label", "off_label", "unclear"}
VALID_PROGRESSION = {"helped_or_improved", "did_not_help", "worsened",
                      "stable_or_no_change", "mixed", "unclear", "not_reported"}
VALID_FINAL_OUTCOME = {"improved", "worsened_or_progressed", "stable",
                        "relapsed_after_initial_improvement", "mixed", "unclear", "not_reported"}
VALID_MENTION_TYPE = {"patient_reports_taking", "commenter_suggests", "asking_about",
                       "doctor_recommended_or_prescribed", "research_or_trial_mention",
                       "incidental", "unclear"}

JUNK_NAMES = {
    "", "unclear", "n/a", "none", "na", "not applicable",
    "meds", "medicine", "medicines", "drug", "drugs", "otc", "vitamins", "steroids", "supplements",
}


def class_level_terms(rows: list[dict]) -> set[str]:
    """normalized_name values that are themselves generic class/category terms,
    per the extraction's own drug_or_treatment_class field -- e.g. normalized_name
    "antibiotics" tagged with class "antibiotic". Derived from the data, not a
    hand-typed guess list (mirrors build_curation_targets.py's filter)."""
    terms = set()
    for row in rows:
        v = (row.get("drug_or_treatment_class") or "").strip().lower()
        if v and v != "unclear" and not v.startswith("["):
            terms.add(v)
    return terms


def clean_enum(value: str, valid_set: set) -> str:
    v = (value or "").strip()
    return v if v in valid_set else "unclear"


def load_and_clean_all() -> list[dict]:
    rows = []
    for community in scraper.COMMUNITIES:
        sub = community["subreddit"]
        path = DRUG_SIGNALS_DIR / f"{sub}_drug_signals.csv"
        if not path.exists():
            print(f"  Warning: no drug_signals.csv for r/{sub}, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["label_status"] = clean_enum(row.get("label_status"), VALID_LABEL_STATUS)
                row["progression_after_treatment"] = clean_enum(row.get("progression_after_treatment"), VALID_PROGRESSION)
                row["final_outcome"] = clean_enum(row.get("final_outcome"), VALID_FINAL_OUTCOME)
                row["mention_type"] = clean_enum(row.get("mention_type"), VALID_MENTION_TYPE)
                row["normalized_name_clean"] = (row.get("normalized_name") or "").strip().lower()
                row["condition_treated_clean"] = (row.get("condition_treated") or "").strip().lower()
                rows.append(row)
    return rows


def write_master(rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_MASTER, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_candidates(rows: list[dict]) -> list[dict]:
    excluded_terms = JUNK_NAMES | class_level_terms(rows)

    groups = defaultdict(lambda: {
        "mentions": 0, "off_label": 0, "on_label": 0, "unclear_label": 0,
        "positive_outcome": 0, "negative_outcome": 0,
        "diseases": set(), "conditions": set(), "posts": set(), "quotes": [],
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
            "total_mentions": g["mentions"],
            "distinct_posts": len(g["posts"]),
            "distinct_diseases": len(g["diseases"]),
            "diseases": "; ".join(sorted(g["diseases"])),
            "conditions_treated": "; ".join(sorted(g["conditions"])[:15]),
            "off_label_mentions": g["off_label"],
            "on_label_mentions": g["on_label"],
            "unclear_label_mentions": g["unclear_label"],
            "positive_outcome_mentions": g["positive_outcome"],
            "negative_outcome_mentions": g["negative_outcome"],
            "example_quotes": " | ".join(g["quotes"]),
        })

    def sort_key(c):
        positive_ratio = c["positive_outcome_mentions"] / c["total_mentions"] if c["total_mentions"] else 0
        return (-c["off_label_mentions"], -c["distinct_diseases"], -c["distinct_posts"], -positive_ratio)

    candidates.sort(key=sort_key)
    return candidates


def write_candidates(candidates: list[dict]):
    if not candidates:
        return
    fieldnames = list(candidates[0].keys())
    with open(OUT_CANDIDATES, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def main():
    print("Loading and cleaning all 20 communities' drug signal CSVs...")
    rows = load_and_clean_all()
    print(f"{len(rows):,} total drug mentions loaded\n")

    write_master(rows)
    print(f"Master table saved: {OUT_MASTER}")

    print("Building repurposing candidate rankings...")
    candidates = build_candidates(rows)
    write_candidates(candidates)
    print(f"{len(candidates):,} distinct normalized drug names -> {OUT_CANDIDATES}\n")

    off_label_candidates = [c for c in candidates if c["off_label_mentions"] > 0]
    multi_disease = [c for c in candidates if c["distinct_diseases"] > 1]
    print(f"{len(off_label_candidates):,} drugs have >=1 explicit off-label mention")
    print(f"{len(multi_disease):,} drugs corroborated across multiple diseases")

    print("\nTop 20 candidates:")
    print(f"{'drug':<28}{'off-label':>10}{'diseases':>10}{'posts':>8}{'positive':>10}")
    for c in candidates[:20]:
        print(f"{c['normalized_name']:<28}{c['off_label_mentions']:>10}{c['distinct_diseases']:>10}"
              f"{c['distinct_posts']:>8}{c['positive_outcome_mentions']:>10}")


if __name__ == "__main__":
    main()
