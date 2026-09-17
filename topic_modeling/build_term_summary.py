"""
build_term_summary.py

Recreates sm_term_summary.md's term-inventory methodology against
scraped_docs/scleroderma_raw.jsonl (the full-history, 4,598-post scrape),
rather than the original 219-post test-case scrape it was originally built
from. Reuses the same category/term-group structure as the original file
(a curated scleroderma vocabulary), but rediscovers actual surface-form
examples and hit/mention counts fresh from this dataset instead of reusing
the original's examples.

For most term-groups, a "term" is a fixed set of alias strings matched
case-insensitively with word boundaries (deduped so a single occurrence in
text isn't double-counted against multiple aliases in the same group). Two
groups (diagnosis timing phrases, medication dose mentions) are inherently
free-text patterns rather than a fixed alias list, so they're matched via a
looser regex and reported using the actual matched spans as examples.

"Record hits" = number of posts containing >=1 match for a term-group.
"Mentions" = total match count across all posts (a post mentioning a term
3 times counts once toward record hits, 3 times toward mentions).

Usage: python build_term_summary.py
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/ncats/users/liky/social_media_data")
INPUT_JSONL = DATA_DIR / "scraped_docs" / "scleroderma_raw.jsonl"
OUT_MD = DATA_DIR / "sm_term_summary_v2.md"

MAX_EXAMPLES = 10

# category -> [(group_key, group_label, [aliases])]
TERM_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "diseases_conditions": [
        ("systemic sclerosis scleroderma", ["scleroderma", "ssc", "systemic sclerosis"]),
        ("raynaud phenomenon", ["raynaud", "raynauds", "raynaud's"]),
        ("autoimmune disease", ["autoimmune", "autoimmune disease", "autoimmune diseases"]),
        ("rheumatoid arthritis", ["ra", "rheumatoid arthritis"]),
        ("lupus", ["lupus", "sle"]),
        ("morphea localized scleroderma", ["morphea", "localized scleroderma", "localised scleroderma"]),
        ("myositis", ["myositis", "polymyositis", "dermatomyositis"]),
        ("anxiety", ["anxiety", "panic"]),
        ("gastroesophageal reflux disease", ["gerd", "reflux", "acid reflux"]),
        ("interstitial lung disease", ["ild", "interstitial lung disease", "pulmonary fibrosis", "lung fibrosis"]),
        ("crest syndrome", ["crest", "crest syndrome"]),
        ("limited systemic sclerosis", ["limited systemic sclerosis", "limited systemic", "limited scleroderma", "lcssc"]),
    ],
    "care_context": [
        ("diagnosis diagnosed", ["diagnosis", "diagnosed", "diagnose", "dx"]),
        ("rheumatologist rheumatology", ["rheumatologist", "rheumatology", "rheum", "rheumy"]),
        ("doctor physician", ["doctor", "physician", "specialist", "dr"]),
        ("clinical trial research", ["clinical trial", "trial", "research"]),
    ],
    "symptoms_manifestations": [
        ("pain", ["pain", "painful", "hurts"]),
        ("fatigue", ["fatigue", "tired", "exhausted"]),
        ("joint pain arthritis symptoms", ["joint pain", "arthritis", "arthralgia"]),
        ("swelling edema", ["swelling", "swollen", "edema"]),
        ("gi symptoms", ["gi", "gastrointestinal", "stomach issues", "gut issues"]),
        ("skin tightening thickening", ["skin tightening", "skin thickening", "tight skin", "skin tightness", "thickened skin"]),
        ("esophageal symptoms", ["esophagus", "esophageal", "oesophageal"]),
        ("itching pruritus", ["itchy", "itching", "itch"]),
        ("digital ulcers", ["digital ulcers", "finger ulcers"]),
        ("calcinosis", ["calcinosis", "calcium deposits"]),
        ("shortness of breath", ["shortness of breath", "trouble breathing", "breathless", "difficulty breathing", "sob"]),
        ("cough", ["cough", "coughing"]),
    ],
    "progression_course": [
        ("improvement better", ["better", "helped", "helps", "working", "worked", "improvement", "improved"]),
        ("getting worse progression", ["worse", "progression", "getting worse", "worsening", "progressive"]),
        ("flare flaring", ["flare", "flares", "flaring", "flare up"]),
        ("side effects adverse effects", ["side effects", "side effect", "reaction", "reactions"]),
        ("no benefit not working", ["did not help", "didn't help", "failed", "not working", "not helping"]),
        ("stable not progressing", ["stable", "no progression"]),
    ],
    "medications_treatments": [
        ("prednisone corticosteroids", ["prednisone", "steroids", "steroid", "corticosteroid"]),
        ("methotrexate", ["methotrexate", "mtx"]),
        ("mycophenolate mofetil", ["cellcept", "mycophenolate", "mycophenolate mofetil", "mmf"]),
        ("stem cell transplant", ["stem cell transplant", "hsct", "bone marrow transplant"]),
        ("hydroxychloroquine", ["hydroxychloroquine", "plaquenil"]),
        ("rituximab", ["rituximab", "rituxan"]),
        ("intravenous immunoglobulin", ["ivig"]),
        ("proton pump inhibitor", ["ppi", "ppis", "pantoprazole", "omeprazole", "prilosec", "esomeprazole"]),
        ("antibiotic", ["antibiotics", "antibiotic"]),
        ("nsaid anti inflammatory", ["nsaids", "ibuprofen", "advil", "aleve"]),
        ("tocilizumab", ["actemra", "tocilizumab"]),
    ],
    "labs_biomarkers_tests": [
        ("antinuclear antibody", ["ana", "antinuclear antibody"]),
        ("blood work labs", ["labs", "bloodwork", "lab work", "blood work"]),
        ("anti scl 70 topoisomerase i antibody", ["scl-70", "scl70", "anti-scl-70"]),
        ("anti centromere antibody", ["centromere", "anti-centromere", "anticentromere", "aca"]),
        ("rna polymerase iii antibody", ["rna polymerase iii", "rnap iii"]),
        ("erythrocyte sedimentation rate", ["esr", "sed rate", "sedimentation rate"]),
        ("c reactive protein", ["crp"]),
        ("dlco", ["dlco"]),
        ("creatine kinase", ["ck", "cpk"]),
        ("forced vital capacity", ["fvc"]),
    ],
    "treatment_response": [
        ("positive response", ["better", "helped", "helps", "relief", "working", "worked", "improved", "benefit"]),
        ("mixed uncertain response", ["not sure", "hard to tell", "mixed"]),
        ("negative response", ["did not help", "didn't help", "failed", "not working", "not helping", "made it worse", "made worse"]),
    ],
    "procedures_monitoring": [
        ("ct imaging", ["ct", "hrct"]),
        ("pulmonary function test", ["pft", "pfts", "pulmonary function test", "pulmonary function tests"]),
        ("echocardiogram", ["echocardiogram", "echo"]),
        ("endoscopy", ["endoscopy"]),
        ("manometry", ["manometry", "esophageal manometry"]),
    ],
    "adverse_effects": [
        ("side effect mention", ["side effects", "side effect", "reaction", "reactions", "intolerance", "intolerant"]),
    ],
}

# free-text pattern groups: category -> (group_key, compiled regex)
PATTERN_GROUPS: dict[str, tuple[str, re.Pattern]] = {
    "progression_course": (
        "diagnosis timing",
        re.compile(r"diagnosed\b(?:(?!\.).){0,45}?\b(?:ago|year|years|month|months|decade|since|in\s+\d{4})\b", re.IGNORECASE),
    ),
    "medications_treatments": (
        "dose dosing mention",
        re.compile(r"\b\d{1,4}\s?mg\b", re.IGNORECASE),
    ),
}


def build_alias_regex(aliases: list[str]) -> re.Pattern:
    escaped = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def load_texts(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts.append(doc.get("full_text") or "")
    return texts


def main():
    texts = load_texts(INPUT_JSONL)
    print(f"Loaded {len(texts)} documents from {INPUT_JSONL}")

    # category -> group_key -> {"records": int, "mentions": int, "examples": Counter}
    results: dict[str, dict[str, dict]] = {}
    compiled: dict[str, dict[str, re.Pattern]] = {}

    for category, groups in TERM_GROUPS.items():
        results[category] = {}
        compiled[category] = {}
        for group_key, aliases in groups:
            compiled[category][group_key] = build_alias_regex(aliases)
            results[category][group_key] = {"records": 0, "mentions": 0, "examples": Counter()}

    for category, (group_key, pattern) in PATTERN_GROUPS.items():
        results.setdefault(category, {})
        compiled.setdefault(category, {})
        compiled[category][group_key] = pattern
        results[category][group_key] = {"records": 0, "mentions": 0, "examples": Counter()}

    for text in texts:
        if not text:
            continue
        for category, group_patterns in compiled.items():
            for group_key, pattern in group_patterns.items():
                matches = list(pattern.finditer(text))
                if not matches:
                    continue
                stats = results[category][group_key]
                stats["records"] += 1
                stats["mentions"] += len(matches)
                for m in matches:
                    stats["examples"][m.group(0)] += 1

    # ---- render markdown ----
    lines = []
    lines.append("# Scleroderma Drug Repurposing Term Summary (v2 -- full-history dataset)")
    lines.append("")
    lines.append(f"Recreated using the same category/term-group structure as the original "
                 f"summary, applied to `scraped_docs/scleroderma_raw.jsonl` "
                 f"({len(texts):,} documents) instead of the original 219-post test scrape. "
                 f"Example surface forms below were rediscovered from this dataset, not "
                 f"copied from the original.")
    lines.append("")

    category_totals = []
    for category, groups in results.items():
        rec_total = sum(g["records"] for g in groups.values())
        men_total = sum(g["mentions"] for g in groups.values())
        category_totals.append((category, rec_total, men_total))
    category_totals.sort(key=lambda x: -x[1])

    lines.append("## Top Categories")
    lines.append("")
    for category, rec_total, men_total in category_totals:
        lines.append(f"- {category}: {rec_total:,} record hits; {men_total:,} mentions")
    lines.append("")

    lines.append("## Top Terms By Category")
    lines.append("")
    for category, _, _ in category_totals:
        lines.append(f"### {category}")
        lines.append("")
        groups = results[category]
        sorted_groups = sorted(groups.items(), key=lambda kv: -kv[1]["records"])
        for group_key, stats in sorted_groups:
            if stats["records"] == 0:
                continue
            top_examples = [ex for ex, _ in stats["examples"].most_common(MAX_EXAMPLES)]
            examples_str = " | ".join(top_examples)
            lines.append(f"- {group_key}: {stats['records']:,} records; {stats['mentions']:,} mentions; "
                         f"examples `{examples_str}`")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
