"""
run_drug_extraction.py

Step 8 (LLM-Based Drug Signal Extraction), generalized from
scleroderma_drug_extraction.py across all 20 scraped rare disease
communities, using drug_repurposing_prompt.py (the generalized,
disease-parameterized prompt) instead of the scleroderma-only one, and
reading from scraped_docs/filtered_for_extraction/ (only clinical_treatment-
topic documents) instead of the full raw corpus -- 133,028 of 163,984
scraped documents (81.1%), per filter_documents_for_extraction.py.

This is a much bigger job than the original scleroderma test case (4,598
documents): 133,028 documents across 20 communities, with full post text
that can run up to ~12,000 tokens at the long tail. Unlike the topic
labeling pass (short prompts, finished in one job easily), this is sized
more like the original scraping job -- likely needs multiple chained SLURM
segments, so results are checkpointed per-post (JSONL, flushed
immediately, .tmp-then-rename per community) rather than held in memory
until a whole community finishes, matching the resumability pattern
scrape_rare_disease_subreddits.py needed for the same reason.

Run on the GPU partition with the test-llama conda env (has vllm 0.11.2):
  sbatch submit_drug_extraction.sh

Outputs, per community:
  scraped_docs/drug_signals/<subreddit>_drug_signals.jsonl   full per-post results
  scraped_docs/drug_signals/<subreddit>_drug_signals.csv     one row per drug mention
"""

import csv
import json
from pathlib import Path

from vllm import LLM, SamplingParams

import scrape_rare_disease_subreddits as scraper
from drug_repurposing_prompt import create_drug_repurposing_prompt

FILTERED_DIR = scraper.OUT_DIR / "filtered_for_extraction"
OUT_DIR = scraper.OUT_DIR / "drug_signals"
OUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "/vast/projects/ncats-llms/gemma3-27b/"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.90

# The generalized prompt's fixed template is ~2,700 tokens (vs. the
# scleroderma-only prompt's shorter one) -- bumped max_model_len and the
# output budget up from the original script's 16384/2048 to keep headroom
# at the long tail (full_text p100 ~12,000 tokens per
# scleroderma_drug_extraction.py's own measurement), given the richer
# schema (medical_terms + more drug_mention fields) also needs more room
# to generate into.
MAX_MODEL_LEN = 20480
MAX_OUTPUT_TOKENS = 3072
MAX_INPUT_CHARS = (MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - 700) * 4

BATCH_SIZE = 10


def load_filtered_posts(subreddit: str) -> list[dict]:
    path = FILTERED_DIR / f"{subreddit}_filtered.jsonl"
    if not path.exists():
        return []
    posts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return posts


def extract_json_object(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end]), None
    except json.JSONDecodeError:
        pass
    return None, text


def process_community(llm: LLM, sampling_params: SamplingParams, subreddit: str, disease_name: str) -> str:
    out_path = OUT_DIR / f"{subreddit}_drug_signals.jsonl"
    tmp_path = OUT_DIR / f"{subreddit}_drug_signals.jsonl.tmp"

    if out_path.exists():
        return "already done"

    posts = load_filtered_posts(subreddit)
    if not posts:
        return "no filtered documents for this community"

    # resume from a partial .tmp file left behind by a crash/time-limit kill
    already_done_ids = set()
    if tmp_path.exists():
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    already_done_ids.add(json.loads(line)["post_id"])
                except Exception:
                    pass
        if already_done_ids:
            print(f"  Resuming: {len(already_done_ids)} posts already saved in {tmp_path.name}")

    remaining = [p for p in posts if p.get("id") not in already_done_ids]
    print(f"  {len(posts)} filtered posts, {len(remaining)} remaining")

    with open(tmp_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]
            conversations = []
            for post in batch:
                text = post.get("full_text") or f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                if len(text) > MAX_INPUT_CHARS:
                    text = text[:MAX_INPUT_CHARS]
                conversations.append([{"role": "user", "content": create_drug_repurposing_prompt(disease_name, text)}])

            outputs = llm.chat(conversations, sampling_params)

            for post, output in zip(batch, outputs):
                response_text = output.outputs[0].text.strip() if output.outputs else ""
                parsed, raw_on_error = extract_json_object(response_text)

                result = {
                    "post_id": post.get("id"),
                    "title": post.get("title"),
                    "score": post.get("score"),
                    "num_comments": post.get("num_comments"),
                    "created_utc": post.get("created_utc"),
                    "topic_id": post.get("topic_id"),
                    "topic_label": post.get("topic_label"),
                    "drug_mentions": (parsed or {}).get("drug_mentions", []),
                    "medical_terms": (parsed or {}).get("medical_terms", {}),
                    "parse_error": raw_on_error is not None,
                }
                fout.write(json.dumps(result) + "\n")
                fout.flush()

            print(f"  Processed {min(i + BATCH_SIZE, len(remaining))}/{len(remaining)} remaining posts")

    tmp_path.rename(out_path)
    return "ok"


def write_csv(subreddit: str, disease_name: str):
    jsonl_path = OUT_DIR / f"{subreddit}_drug_signals.jsonl"
    csv_path = OUT_DIR / f"{subreddit}_drug_signals.csv"
    if not jsonl_path.exists():
        return 0

    fieldnames = [
        "subreddit", "disease_name", "post_id", "title", "score", "num_comments",
        "created_utc", "topic_id", "topic_label", "drug_mention", "normalized_name",
        "drug_or_treatment_class", "condition_treated", "label_status", "mention_type",
        "progression_after_treatment", "final_outcome", "adverse_effects",
        "dose_or_schedule", "quote",
    ]
    n_rows = 0
    with open(jsonl_path, encoding="utf-8") as fin, open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for line in fin:
            rec = json.loads(line)
            for mention in rec.get("drug_mentions", []):
                if not isinstance(mention, dict):
                    continue
                writer.writerow({
                    "subreddit": subreddit,
                    "disease_name": disease_name,
                    "post_id": rec["post_id"],
                    "title": rec["title"],
                    "score": rec["score"],
                    "num_comments": rec["num_comments"],
                    "created_utc": rec["created_utc"],
                    "topic_id": rec["topic_id"],
                    "topic_label": rec["topic_label"],
                    "drug_mention": mention.get("drug_mention", ""),
                    "normalized_name": mention.get("normalized_name", ""),
                    "drug_or_treatment_class": mention.get("drug_or_treatment_class", ""),
                    "condition_treated": mention.get("condition_treated", ""),
                    "label_status": mention.get("label_status", ""),
                    "mention_type": mention.get("mention_type", ""),
                    "progression_after_treatment": mention.get("progression_after_treatment", ""),
                    "final_outcome": mention.get("final_outcome", ""),
                    "adverse_effects": "; ".join(mention.get("adverse_effects") or []),
                    "dose_or_schedule": mention.get("dose_or_schedule", ""),
                    "quote": mention.get("quote", ""),
                })
                n_rows += 1
    return n_rows


def main():
    print(f"{len(scraper.COMMUNITIES)} communities queued for drug signal extraction\n")

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

    for i, community in enumerate(scraper.COMMUNITIES, 1):
        sub = community["subreddit"]
        print(f"[{i}/{len(scraper.COMMUNITIES)}] r/{sub} ({community['disease_name']})")
        status = process_community(llm, sampling_params, sub, community["disease_name"])
        print(f"  {status}")
        if status == "ok":
            n_rows = write_csv(sub, community["disease_name"])
            print(f"  {n_rows} drug mentions -> {OUT_DIR / f'{sub}_drug_signals.csv'}")


if __name__ == "__main__":
    main()
