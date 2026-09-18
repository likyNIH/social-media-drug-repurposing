"""
build_curation_targets.py

Produces a per-disease "curation target list" -- the drugs worth manually
researching approved-indication status for, ranked by how much mention
volume they actually account for. Companion to drug_disease_label_breakdown.py's
output: re-reads drug_disease_label_breakdown.csv (already grouped by
(drug, disease) pair) and, per disease, keeps the top N drugs by
total_mentions plus how much of that disease's total mention volume they
cover cumulatively -- so curation effort goes where it actually moves the
aggregate numbers, not down the long tail of one-off mentions.

Outputs:
  curation_targets.csv       top N rows per disease, ready to fill in three
    blank columns: "entry_type" (prescription_drug / otc_supplement /
    procedure_or_device / other_not_a_drug -- the class-term filter below
    catches obvious cases like "stimulants" or "cpap", but plenty of real
    entries, e.g. dietary supplements or testing services, still need a
    human call), "approved_for_disease" (only meaningful when entry_type is
    prescription_drug -- yes/no/unsure), and "source" (citation, e.g.
    Orphanet/DailyMed/NORD link)
  curation_targets_summary.csv   one row per disease: how many distinct
    drugs mentioned, how many needed to cover 80% of mention volume

Filters out generic/class-level entries before ranking: any normalized_name
that exactly matches one of the extraction's own drug_or_treatment_class
values (drug_signals_master.csv) is treated as non-specific -- e.g.
normalized_name "stimulants" tagged with class "stimulant", or normalized_name
"cpap" tagged with class "cpap". These can't be checked against an approved-
indication source (there's no single answer for "is 'stimulants' approved"),
so they're not useful curation targets. This is derived from the extraction's
own output, not a hand-typed guess list.

Usage: python build_curation_targets.py [TOP_N]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper

DRUG_SIGNALS_DIR = scraper.OUT_DIR / "drug_signals"
BREAKDOWN_PATH = DRUG_SIGNALS_DIR / "drug_disease_label_breakdown.csv"
MASTER_PATH = DRUG_SIGNALS_DIR / "drug_signals_master.csv"
OUT_TARGETS = DRUG_SIGNALS_DIR / "curation_targets.csv"
OUT_SUMMARY = DRUG_SIGNALS_DIR / "curation_targets_summary.csv"
OUT_EXCLUDED = DRUG_SIGNALS_DIR / "curation_excluded_terms.csv"

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
COVERAGE_THRESHOLD = 0.8


def load_class_level_terms() -> set[str]:
    """normalized_name values that are themselves generic class/category
    terms, per the extraction's own drug_or_treatment_class field."""
    terms = set()
    with open(MASTER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("drug_or_treatment_class") or "").strip().lower()
            if v and v != "unclear" and not v.startswith("["):
                terms.add(v)
    return terms


def main():
    class_terms = load_class_level_terms()

    by_disease = defaultdict(list)
    excluded = []
    with open(BREAKDOWN_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["total_mentions"] = int(row["total_mentions"])
            row["unclear_label_mentions"] = int(row["unclear_label_mentions"])
            if row["normalized_name"].strip().lower() in class_terms:
                excluded.append(row)
                continue
            by_disease[row["disease_name"]].append(row)

    excluded.sort(key=lambda r: -r["total_mentions"])
    with open(OUT_EXCLUDED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(excluded[0].keys()))
        writer.writeheader()
        writer.writerows(excluded)
    print(f"{len(excluded)} (drug, disease) pairs excluded as class-level terms -> {OUT_EXCLUDED}")

    target_rows = []
    summary_rows = []

    for disease in sorted(by_disease):
        drugs = sorted(by_disease[disease], key=lambda r: -r["total_mentions"])
        disease_total = sum(r["total_mentions"] for r in drugs)

        running = 0
        n_for_80pct = 0
        for r in drugs:
            running += r["total_mentions"]
            n_for_80pct += 1
            if running >= COVERAGE_THRESHOLD * disease_total:
                break

        summary_rows.append({
            "disease_name": disease,
            "distinct_drugs_mentioned": len(drugs),
            "total_mentions": disease_total,
            "drugs_to_cover_80pct_of_mentions": n_for_80pct,
        })

        cumulative = 0
        for rank, r in enumerate(drugs[:TOP_N], 1):
            cumulative += r["total_mentions"]
            target_rows.append({
                "disease_name": disease,
                "rank": rank,
                "normalized_name": r["normalized_name"],
                "total_mentions": r["total_mentions"],
                "distinct_posts": r["distinct_posts"],
                "pct_of_disease_mentions": round(100 * r["total_mentions"] / disease_total, 1),
                "cumulative_pct": round(100 * cumulative / disease_total, 1),
                "current_unclear_mentions": r["unclear_label_mentions"],
                "entry_type": "",             # fill in: prescription_drug / otc_supplement /
                                               # procedure_or_device / other_not_a_drug
                "approved_for_disease": "",   # fill in ONLY if entry_type is prescription_drug:
                                               # yes / no / unsure
                "source": "",                 # fill in: Orphanet/DailyMed/NORD link or citation
            })

    with open(OUT_TARGETS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()))
        writer.writeheader()
        writer.writerows(target_rows)
    print(f"{len(target_rows)} curation-target rows across {len(by_disease)} diseases -> {OUT_TARGETS}")

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Per-disease summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
