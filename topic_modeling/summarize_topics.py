"""
summarize_topics.py

Scans topics.json across all 20 communities' topic models and classifies
each topic as:
  - clinical: enough overlap with a broad, disease-agnostic medical/
    treatment keyword list to plausibly be worth including in the LLM
    extraction step (Step 8)
  - generic_chatter: mostly common English stopwords -- support/daily-life
    noise, not usefully specific to anything
  - non_english_cluster: mostly common Spanish/Portuguese/French stopwords
    -- Top2Vec clustering by language rather than by subject, confirmed to
    happen in glioblastoma and Microtia's topic models
  - other: doesn't clearly fall into the above -- needs a human look

This is a coarse triage, not a substitute for actually reading the word
clouds -- it exists to help prioritize which of the ~915 topics across all
20 communities are worth a first look, per the manual Step 7 labeling task.

Outputs:
  topic_summary_full.md     every topic, every community, with its label
  (also prints a condensed per-community overview to stdout)

Usage: python summarize_topics.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper

DATA_DIR = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
TOPIC_DIR = scraper.OUT_DIR / "topic_models"
OUT_MD = DATA_DIR / "topic_summary_full.md"

CLINICAL_WORDS = {
    "pain", "painful", "fatigue", "tired", "exhausted", "doctor", "specialist",
    "diagnosis", "diagnosed", "treatment", "treated", "medication", "medicine",
    "meds", "dose", "dosage", "mg", "prescription", "prescribed", "symptom",
    "symptoms", "surgery", "surgical", "therapy", "hospital", "clinic",
    "biopsy", "infusion", "injection", "side", "effect", "effects", "flare",
    "chronic", "condition", "disease", "syndrome", "test", "tests", "scan",
    "blood", "labs", "steroid", "steroids", "immune", "immunosuppressant",
    "antibody", "antibodies", "rheumatologist", "neurologist", "cardiologist",
    "trial", "study", "research", "insurance", "appointment", "nurse",
    "swelling", "rash", "nausea", "infection", "inflammation", "relief",
}

ENGLISH_STOPWORDS = {
    "the", "is", "to", "not", "that", "you", "it", "this", "of", "as", "so",
    "for", "are", "have", "be", "way", "tell", "someone", "people", "very",
    "person", "things", "with", "and", "but", "just", "us", "really",
    "think", "time", "hard", "everything", "others", "say", "should",
    "there", "one", "am", "no", "up", "within", "due",
}

NON_ENGLISH_STOPWORDS = {
    "que", "se", "da", "nao", "como", "pero", "si", "es", "poco", "de",
    "pasando", "sentir", "apenas", "possivel", "vez", "tomando", "senti",
    "cama", "puede", "primeira", "para", "com", "uma", "isso", "esta",
    "sofrimento", "ano",
}


def classify_topic(words: list[str]) -> str:
    top = [w.lower() for w in words[:15]]
    clinical_hits = sum(1 for w in top if w in CLINICAL_WORDS)
    english_stop_hits = sum(1 for w in top if w in ENGLISH_STOPWORDS)
    non_english_hits = sum(1 for w in top if w in NON_ENGLISH_STOPWORDS)

    if non_english_hits >= 3:
        return "non_english_cluster"
    if clinical_hits >= 3:
        return "clinical"
    if english_stop_hits >= 6:
        return "generic_chatter"
    return "other"


def main():
    lines = ["# Topic Summary (all communities)\n"]
    overview = []

    for community in scraper.COMMUNITIES:
        sub = community["subreddit"]
        topics_path = TOPIC_DIR / sub / "topics.json"
        if not topics_path.exists():
            continue
        with open(topics_path, encoding="utf-8") as f:
            topics = json.load(f)

        counts = {"clinical": 0, "generic_chatter": 0, "non_english_cluster": 0, "other": 0}
        lines.append(f"## r/{sub} ({community['disease_name']}) -- {len(topics)} topics\n")
        for t in topics:
            label = classify_topic(t["top_words"])
            counts[label] += 1
            lines.append(f"- [{label}] Topic {t['topic_id']}: {', '.join(t['top_words'][:12])}")
        lines.append("")

        overview.append((sub, community["disease_name"], len(topics), counts))

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'subreddit':<24}{'total':>7}{'clinical':>10}{'chatter':>10}{'non-eng':>9}{'other':>8}")
    for sub, disease, total, counts in overview:
        print(f"{sub:<24}{total:>7}{counts['clinical']:>10}{counts['generic_chatter']:>10}"
              f"{counts['non_english_cluster']:>9}{counts['other']:>8}")

    print(f"\nFull per-topic breakdown written to {OUT_MD}")


if __name__ == "__main__":
    main()
