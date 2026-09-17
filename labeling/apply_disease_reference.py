"""
apply_disease_reference.py

Applies the disease-first approved-drug reference (disease_approved_drugs.csv
-- built by searching each disease's name directly against openFDA's
indications_and_usage text, rather than checking mentioned drugs one at a
time) on top of the existing curated_labels reclassification. Reads
drug_signals_master_reclassified.csv (already corrected once via
curation_targets.csv / apply_curated_labels.py) and applies a second pass.

Three-way verdict instead of a strict yes/no:
  "yes"         -> unconditional approval. Resolves unclear -> on_label,
                   and corrects a conflicting off_label call -> on_label
                   (or on_label -> off_label, if verdict is "no").
  "no"          -> unconditional non-approval (either confirmed absent from
                   the disease's approved list, or explicitly excluded on
                   the drug's own label, e.g. deferoxamine/hemochromatosis).
                   Same correction logic in the opposite direction.
  "conditional" -> approved only for a named subset (genotype, disease
                   subtype, age range, or as part of a combination regimen
                   -- see condition_qualifier). Only resolves "unclear"
                   mentions, and only when the mention's own condition_treated
                   text contains one of that drug's qualifier keywords (below)
                   -- never overrides an existing on_label/off_label call,
                   since we can't confirm the subtype match with confidence
                   otherwise.

Writes back to the SAME reclassified file (this is a second correction pass
on top of the first, not a competing one), adding "disease_reference" as a
new label_status_source value alongside the existing "llm_extraction" /
"curated_reference" / "curated_correction".

Usage: python apply_disease_reference.py
"""

import csv
from collections import Counter, defaultdict

import scrape_rare_disease_subreddits as scraper

DRUG_SIGNALS_DIR = scraper.OUT_DIR / "drug_signals"
RECLASSIFIED_PATH = DRUG_SIGNALS_DIR / "drug_signals_master_reclassified.csv"
REFERENCE_PATH = DRUG_SIGNALS_DIR / "disease_approved_drugs.csv"

# Keyword sets for conditional entries -- only touches "unclear" mentions
# whose condition_treated text names the qualifying subset. Entries with no
# keyword list here are left conditional-but-unresolved (too risky to guess).
CONDITIONAL_KEYWORDS = {
    ("cystic fibrosis", "ivacaftor"): ["g551d", "gating mutation", "responsive mutation"],
    ("cystic fibrosis", "elexacaftor/tezacaftor/ivacaftor"): ["f508del"],
    ("cystic fibrosis", "lumacaftor/ivacaftor"): ["f508del"],
    ("cystic fibrosis", "tezacaftor/ivacaftor"): ["f508del"],
    ("mastocytosis", "imatinib"): ["aggressive systemic mastocytosis", "asm"],
    ("mastocytosis", "midostaurin"): ["aggressive systemic mastocytosis", "asm", "mast cell leukemia", "sm-ahn"],
    ("mastocytosis", "cimetidine"): ["hypersecretory", "zollinger"],
    ("mastocytosis", "omeprazole"): ["hypersecretory", "zollinger"],
    ("mastocytosis", "ranitidine"): ["hypersecretory", "zollinger"],
    ("scleroderma", "tocilizumab"): ["ild", "interstitial lung"],
    ("scleroderma", "nintedanib"): ["ild", "interstitial lung"],
    ("adult glioblastoma", "bevacizumab"): ["recurrent"],
    ("myasthenia gravis", "eculizumab"): ["achr", "acetylcholine receptor"],
    ("myasthenia gravis", "rozanolixizumab"): ["achr", "musk", "acetylcholine receptor"],
    ("myasthenia gravis", "inebilizumab"): ["achr", "musk", "acetylcholine receptor"],
    ("myasthenia gravis", "zilucoplan"): ["achr", "acetylcholine receptor"],
    ("myasthenia gravis", "ravulizumab"): ["achr", "acetylcholine receptor"],
    ("thalassemia", "luspatercept"): ["transfusion"],
    ("thalassemia", "betibeglogene autotemcel"): ["transfusion"],
    ("thalassemia", "exagamglogene autotemcel"): ["transfusion"],
    ("thalassemia", "deferasirox"): ["non-transfusion", "ntdt"],
    ("lip and oral cavity carcinoma", "docetaxel"): ["head and neck", "scchn"],
    ("lip and oral cavity carcinoma", "methotrexate"): ["head and neck", "scchn"],
    ("lip and oral cavity carcinoma", "nivolumab"): ["head and neck", "scchn", "recurrent", "metastatic"],
    ("lip and oral cavity carcinoma", "cetuximab"): ["head and neck", "scchn"],
    ("proximal spinal muscular atrophy", "onasemnogene abeparvovec"): ["smn1"],
    ("hemophilia", "desmopressin"): ["mild"],
    ("hemophilia", "coagulation factor viia"): ["inhibitor"],
    ("hemophilia", "tranexamic acid"): ["short-term", "short term"],
    ("thalassemia", "thiotepa"): ["transplant", "hsct", "conditioning"],
}


def load_reference():
    """Returns dict[(disease, alias_lowercase)] -> (verdict, disease, canonical_name)."""
    lookup = {}
    with open(REFERENCE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disease = row["disease_name"]
            canonical = row["normalized_name"].strip().lower()
            verdict = row["approved_for_disease"]
            aliases = [canonical] + [
                b.strip().lower() for b in row["brand_names"].split(";") if b.strip()
            ]
            for alias in aliases:
                lookup[(disease, alias)] = (verdict, canonical)
    return lookup


def main():
    reference = load_reference()
    print(f"{len(reference)} (disease, alias) lookup keys from disease_approved_drugs.csv")

    before = Counter()
    after = Counter()
    by_source = Counter()
    resolved = 0
    corrected = 0

    rows = []
    with open(RECLASSIFIED_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            before[row["label_status"]] += 1

            disease = row["disease_name"]
            name = (row.get("normalized_name_clean") or "").strip().lower()
            hit = reference.get((disease, name))

            if hit:
                verdict, canonical = hit
                status = row["label_status"]

                if verdict in ("yes", "no"):
                    target = "on_label" if verdict == "yes" else "off_label"
                    if status == "unclear":
                        row["label_status"] = target
                        row["label_status_source"] = "disease_reference"
                        resolved += 1
                    elif status != target:
                        row["label_status"] = target
                        row["label_status_source"] = "disease_reference"
                        corrected += 1

                elif verdict == "conditional" and status == "unclear":
                    keywords = CONDITIONAL_KEYWORDS.get((disease, canonical), [])
                    cond_text = (row.get("condition_treated") or "").lower()
                    if keywords and any(kw in cond_text for kw in keywords):
                        row["label_status"] = "on_label"
                        row["label_status_source"] = "disease_reference"
                        resolved += 1

            after[row["label_status"]] += 1
            by_source[row["label_status_source"]] += 1
            rows.append(row)

    with open(RECLASSIFIED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{resolved:,} additional unclear mentions resolved, {corrected:,} on/off_label mentions corrected\n")
    print(f"{'label_status':<15}{'before':>10}{'after':>10}")
    for status in ("on_label", "off_label", "unclear"):
        print(f"{status:<15}{before[status]:>10,}{after[status]:>10,}")
    print()
    print("label_status_source breakdown after this pass:")
    for source, n in by_source.most_common():
        print(f"  {source:<20}{n:>10,}")
    print(f"\n-> {RECLASSIFIED_PATH}")


if __name__ == "__main__":
    main()
