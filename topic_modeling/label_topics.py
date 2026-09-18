"""
label_topics.py

LLM-assisted pre-labeling for Step 7 (manual topic labeling), across all
20 communities' Top2Vec topics (~915 total). Uses the same gemma3-27b /
vllm infrastructure as scleroderma_drug_extraction.py, and
topic_labeling_prompt.create_topic_labeling_prompt() for the prompt itself.

For each topic: pulls its top keywords plus a few real example documents
(via doc_topics.json), asks the model for a draft label + relevance
category (clinical_treatment / support_community / off_topic /
data_artifact / unclear), and saves the result.

This is a DRAFT pass, not a final labeling -- review category assignments
before using them to select documents for Step 8 extraction. Confirmed in
this project that automated methods can miscategorize niche clinical
vocabulary (a keyword heuristic completely missed a drug-trial topic
because "takeda, centessa, alkermes" didn't match any generic medical
term list); spot-check a sample of "clinical_treatment" and "unclear"
labels particularly, since those are the ones that matter most for
filtering and are most likely to need a second look.

Run on the GPU partition with the test-llama conda env (has vllm 0.11.2):
  sbatch submit_label_topics.sh

Outputs, per community:
  scraped_docs/topic_models/<subreddit>/topic_labels.json
"""

import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping"))
import scrape_rare_disease_subreddits as scraper
from topic_labeling_prompt import create_topic_labeling_prompt

TOPIC_DIR = scraper.OUT_DIR / "topic_models"

MODEL_PATH = "/vast/projects/ncats-llms/gemma3-27b/"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 4096   # topics are keywords + short excerpts, far shorter
                       # than full posts -- no need for the 16384 used by
                       # scleroderma_drug_extraction.py
MAX_OUTPUT_TOKENS = 300

BATCH_SIZE = 20
EXAMPLES_PER_TOPIC = 3
SNIPPET_MAX_CHARS = 500


def load_doc_lookup(subreddit: str) -> dict[str, str]:
    """id -> full_text, for pulling example excerpts per topic."""
    lookup = {}
    with open(scraper.OUT_DIR / f"{subreddit}_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            lookup[d["id"]] = d.get("full_text") or ""
    return lookup


def build_topic_jobs(subreddit: str, disease_name: str) -> list[dict]:
    """One job per topic: keywords + example snippets, ready to prompt."""
    topics_path = TOPIC_DIR / subreddit / "topics.json"
    doc_topics_path = TOPIC_DIR / subreddit / "doc_topics.json"
    if not topics_path.exists() or not doc_topics_path.exists():
        return []

    with open(topics_path, encoding="utf-8") as f:
        topics = json.load(f)
    with open(doc_topics_path, encoding="utf-8") as f:
        doc_topics = json.load(f)

    doc_lookup = load_doc_lookup(subreddit)

    ids_by_topic: dict[int, list[str]] = {}
    for dt in doc_topics:
        ids_by_topic.setdefault(dt["topic_id"], []).append(dt["id"])

    jobs = []
    for t in topics:
        ids = ids_by_topic.get(t["topic_id"], [])[:EXAMPLES_PER_TOPIC]
        snippets = [doc_lookup.get(i, "")[:SNIPPET_MAX_CHARS] for i in ids]
        jobs.append({
            "subreddit": subreddit,
            "disease_name": disease_name,
            "topic_id": t["topic_id"],
            "top_words": t["top_words"],
            "snippets": [s for s in snippets if s],
        })
    return jobs


def extract_json_from_text(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end]), None
    except json.JSONDecodeError:
        pass
    return None, text


def main():
    print("Building topic labeling jobs across all communities...")
    all_jobs = []
    for community in scraper.COMMUNITIES:
        sub = community["subreddit"]
        out_path = TOPIC_DIR / sub / "topic_labels.json"
        if out_path.exists():
            print(f"  r/{sub}: already labeled, skipping")
            continue
        jobs = build_topic_jobs(sub, community["disease_name"])
        print(f"  r/{sub}: {len(jobs)} topics queued")
        all_jobs.extend(jobs)

    if not all_jobs:
        print("Nothing to do.")
        return

    print(f"\n{len(all_jobs)} total topics to label")
    print(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    results_by_subreddit: dict[str, list[dict]] = {}

    for i in range(0, len(all_jobs), BATCH_SIZE):
        batch = all_jobs[i:i + BATCH_SIZE]
        conversations = [
            [{"role": "user", "content": create_topic_labeling_prompt(
                job["disease_name"], job["top_words"], job["snippets"])}]
            for job in batch
        ]
        outputs = llm.chat(conversations, sampling_params)

        for job, output in zip(batch, outputs):
            response_text = output.outputs[0].text.strip() if output.outputs else ""
            parsed, raw_on_error = extract_json_from_text(response_text)

            result = {
                "topic_id": job["topic_id"],
                "top_words": job["top_words"],
                "label": (parsed or {}).get("label", ""),
                "category": (parsed or {}).get("category", "unclear"),
                "reasoning": (parsed or {}).get("reasoning", ""),
                "parse_error": raw_on_error is not None,
            }
            results_by_subreddit.setdefault(job["subreddit"], []).append(result)

        print(f"Processed {min(i + BATCH_SIZE, len(all_jobs))}/{len(all_jobs)} topics")

    for sub, results in results_by_subreddit.items():
        out_path = TOPIC_DIR / sub / "topic_labels.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        n_errors = sum(1 for r in results if r["parse_error"])
        print(f"r/{sub}: {len(results)} topics labeled ({n_errors} parse errors) -> {out_path}")


if __name__ == "__main__":
    main()
