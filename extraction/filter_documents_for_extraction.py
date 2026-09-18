"""
filter_documents_for_extraction.py

The step between topic labeling (Step 7) and LLM drug-signal extraction
(Step 8): selects only documents whose topic was labeled
"clinical_treatment" (topic_labels.json + doc_topics.json), and writes them
to a filtered JSONL per community. This is what should actually get fed to
drug_repurposing_prompt.py, not the full scraped corpus -- per Karas et al.
(2022), most Reddit content in these communities isn't clinically relevant,
so running extraction unfiltered would waste the majority of LLM compute on
daily-life/support chatter.

Note: scleroderma, autoimmunehepatitis, glioblastoma, and Microtia originally
had too-coarse a Top2Vec topic split (2 topics each) to filter meaningfully.
Re-running with a lower min_cluster_size (see build_topic_models.py's
process_subreddit) fixed all four -- this is no longer an active issue, kept
here as a note in case a similar coarse split shows up in a future disease.

Outputs:
  scraped_docs/filtered_for_extraction/<subreddit>_filtered.jsonl
  (each original post record, plus topic_id and topic_label for traceability)

Usage: python filter_documents_for_extraction.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper

TOPIC_DIR = scraper.OUT_DIR / "topic_models"
OUT_DIR = scraper.OUT_DIR / "filtered_for_extraction"
OUT_DIR.mkdir(exist_ok=True)

INCLUDE_CATEGORIES = {"clinical_treatment"}

# previously had a too-coarse topic split (see module docstring) -- now
# fixed, kept as an empty set so the summary's flagging logic is a no-op
# rather than needing to be removed outright.
KNOWN_COARSE: set[str] = set()


def filter_subreddit(subreddit: str) -> dict:
    labels_path = TOPIC_DIR / subreddit / "topic_labels.json"
    doc_topics_path = TOPIC_DIR / subreddit / "doc_topics.json"
    raw_path = scraper.OUT_DIR / f"{subreddit}_raw.jsonl"

    if not labels_path.exists() or not doc_topics_path.exists():
        return {"total": 0, "kept": 0, "status": "missing topic model/labels"}

    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    with open(doc_topics_path, encoding="utf-8") as f:
        doc_topics = json.load(f)

    topic_label_by_id = {t["topic_id"]: t["label"] for t in labels}
    keep_topic_ids = {t["topic_id"] for t in labels if t["category"] in INCLUDE_CATEGORIES}
    topic_by_doc_id = {dt["id"]: dt["topic_id"] for dt in doc_topics}

    out_path = OUT_DIR / f"{subreddit}_filtered.jsonl"
    total = 0
    kept = 0
    with open(raw_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            # documents whose full_text was empty or an exact duplicate of
            # another post were never assigned a topic during Top2Vec
            # training (build_topic_models.py dedupes on text), so they
            # correctly have no entry here and get skipped.
            topic_id = topic_by_doc_id.get(doc.get("id"))
            if topic_id is None or topic_id not in keep_topic_ids:
                continue
            doc["topic_id"] = topic_id
            doc["topic_label"] = topic_label_by_id.get(topic_id, "")
            fout.write(json.dumps(doc) + "\n")
            kept += 1

    if kept == 0 and out_path.exists():
        out_path.unlink()  # don't leave an empty file lying around

    return {"total": total, "kept": kept, "status": "ok"}


def main():
    print(f"Filtering documents for extraction across {len(scraper.COMMUNITIES)} communities")
    print(f"Including categories: {sorted(INCLUDE_CATEGORIES)}\n")

    grand_total, grand_kept = 0, 0
    for community in scraper.COMMUNITIES:
        sub = community["subreddit"]
        result = filter_subreddit(sub)
        grand_total += result["total"]
        grand_kept += result["kept"]
        pct = f"{100 * result['kept'] / result['total']:.1f}%" if result["total"] else "-"
        flag = "  <- coarse topic split, see module docstring" if sub in KNOWN_COARSE else ""
        print(f"  r/{sub:<24} {result['kept']:>7,} / {result['total']:<7,} kept ({pct})  [{result['status']}]{flag}")

    print(f"\nTOTAL: {grand_kept:,} / {grand_total:,} documents kept ({100 * grand_kept / grand_total:.1f}%)")
    print(f"Filtered files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
