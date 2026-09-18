"""
apply_curated_labels.py

Applies the manually curated approved-indication reference (curation_targets.csv
-- entry_type / approved_for_disease / source, filled in for the top drugs per
disease per build_curation_targets.py) back onto drug_signals_master.csv to
correct label_status using real approved-indication data.

Only acts on (drug, disease) pairs where entry_type is "prescription_drug" and
approved_for_disease is a confident "yes" or "no" -- "unsure" entries and
non-drug entry_types are left alone, since those don't give a clean answer.

Two kinds of edits, kept distinguishable via label_status_source:

  "unclear" mentions are RESOLVED using the curated verdict:
    approved_for_disease == "yes"  ->  unclear becomes on_label
    approved_for_disease == "no"   ->  unclear becomes off_label
    source -> "curated_reference"

  "on_label" / "off_label" mentions are CORRECTED if they conflict with the
  curated verdict (e.g. the LLM called an FDA-approved use "off_label" because
  the post's casual framing read as unauthorized/recreational use rather than
  a disease-approval fact -- see amphetamine/dextroamphetamine and
  methylphenidate for narcolepsy, both correctly curated "yes" but still
  showing up as off_label mentions from the raw extraction):
    approved_for_disease == "yes" and label_status == "off_label" -> on_label
    approved_for_disease == "no"  and label_status == "on_label"  -> off_label
    source -> "curated_correction"
  Mentions that already agree with the curated verdict are left untouched
  (still "llm_extraction" -- the curated data confirms them, no edit needed).

  "unsure" entries, and any (drug, disease) pair outside the top-N curation
  list, are left exactly as the LLM extracted them.

Writes a SEPARATE file (does not modify drug_signals_master.csv), with one
added column, label_status_source, so every row's provenance is traceable:
  "llm_extraction"     untouched -- no curation entry, "unsure", or already
                        agreed with the curated verdict
  "curated_reference"  unclear -> on_label/off_label, resolved from curation
  "curated_correction" on_label/off_label -> the opposite, corrected because
                        it conflicted with the curated verdict

Outputs:
  drug_signals_master_reclassified.csv

Usage: python apply_curated_labels.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper

DRUG_SIGNALS_DIR = scraper.OUT_DIR / "drug_signals"
MASTER_PATH = DRUG_SIGNALS_DIR / "drug_signals_master.csv"
CURATION_PATH = DRUG_SIGNALS_DIR / "curation_targets.csv"
OUT_PATH = DRUG_SIGNALS_DIR / "drug_signals_master_reclassified.csv"


def load_curation() -> dict:
    lookup = {}
    with open(CURATION_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["entry_type"] != "prescription_drug":
                continue
            if row["approved_for_disease"] not in ("yes", "no"):
                continue
            key = (row["disease_name"], row["normalized_name"].strip().lower())
            lookup[key] = row["approved_for_disease"]
    return lookup


def main():
    curation = load_curation()
    print(f"{len(curation)} (drug, disease) pairs with a confident curated answer")

    before = Counter()
    after = Counter()
    resolved = 0
    corrected = 0

    rows = []
    with open(MASTER_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["label_status_source"]
        for row in reader:
            before[row["label_status"]] += 1
            row["label_status_source"] = "llm_extraction"

            key = (row["disease_name"], (row.get("normalized_name_clean") or "").strip().lower())
            verdict = curation.get(key)

            if verdict is not None and row["label_status"] == "unclear":
                row["label_status"] = "on_label" if verdict == "yes" else "off_label"
                row["label_status_source"] = "curated_reference"
                resolved += 1
            elif verdict == "yes" and row["label_status"] == "off_label":
                row["label_status"] = "on_label"
                row["label_status_source"] = "curated_correction"
                corrected += 1
            elif verdict == "no" and row["label_status"] == "on_label":
                row["label_status"] = "off_label"
                row["label_status_source"] = "curated_correction"
                corrected += 1

            after[row["label_status"]] += 1
            rows.append(row)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{resolved:,} unclear mentions resolved, {corrected:,} on/off_label mentions corrected\n")
    print(f"{'label_status':<15}{'before':>10}{'after':>10}")
    for status in ("on_label", "off_label", "unclear"):
        print(f"{status:<15}{before[status]:>10,}{after[status]:>10,}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
