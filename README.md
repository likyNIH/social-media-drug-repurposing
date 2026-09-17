# Rare Disease Drug Repurposing via Reddit Topic Modeling

Identifies drug repurposing signals in rare disease patient communities on
Reddit by combining topic modeling (Top2Vec) with LLM-based structured
extraction. Methodology adapted from:

> Karas, Qu, Xu & Zhu (2022). "Experiments with LDA and Top2Vec for embedded
> topic discovery on social media data: A case study of cystic fibrosis."
> *Frontiers in Artificial Intelligence*, 5:948313.

## Data

This repository is **code only**. The scraped Reddit data, LLM extraction
outputs, and topic models are not included: some individual files exceed
GitHub's size limits, and the underlying data (patient posts, verbatim
quotes, comments) is not appropriate to redistribute in bulk. Every script
below reads from and writes to `scraped_docs/`, which you regenerate
locally by running the pipeline in order.

Two small, non-sensitive reference tables (openFDA-derived approved-drug
lookups, no patient data) are included under `reference_data/` for anyone
re-running the correction steps (5b/5c below).

## Pipeline

1. **Rank communities** — `rank_disease_subreddits.py`
   Ranks all ~6,000 GARD rare diseases by Reddit community size via the
   arctic-shift API. Output feeds the manually-curated shortlist in
   `scrape_rare_disease_subreddits.py`'s `COMMUNITIES` list.

2. **Scrape** — `scrape_rare_disease_subreddits.py`
   Pulls full post + comment history per community from the arctic-shift
   public API (no auth required) into `scraped_docs/<subreddit>_raw.jsonl`.
   Resumable at the subreddit level.
   `backfill_missing_comments.py` re-fetches posts where a failed comment
   fetch was silently recorded as "no comments."

3. **Topic modeling** — `build_topic_models.py`
   Runs Top2Vec (doc2vec embedding) across all scraped communities —
   Karas et al. found Top2Vec/doc2vec outperformed LDA on this kind of
   data and needs no manual K selection. Also generates word clouds per
   topic.

4. **Topic labeling** — `label_topics.py` + `topic_labeling_prompt.py`
   LLM-assisted labeling of each topic as clinical/non-clinical, using the
   same vLLM + Gemma3-27b infrastructure as the extraction step.

5. **Filter for extraction** — `filter_documents_for_extraction.py`
   Keeps only documents whose topic was labeled clinically relevant,
   written to `scraped_docs/filtered_for_extraction/`.

6. **Drug signal extraction** — `run_drug_extraction.py` +
   `drug_repurposing_prompt.py`
   LLM extraction of structured drug mentions (name, normalized name,
   label status, mention type, outcome, supporting quote) per post, across
   all communities.

7. **Aggregation** — `aggregate_drug_signals.py`
   Combines all per-disease outputs into `drug_signals_master.csv`,
   cleaning known extraction-quality issues (cross-contaminated enum
   fields, etc.).

8. **Two-pass label correction:**
   - **5b.** `build_curation_targets.py` + `apply_curated_labels.py` —
     manually curated approved-indication lookup for each disease's
     top drugs by mention volume, applied back onto the master table.
   - **5c.** `apply_disease_reference.py` — a second, disease-first pass
     using `reference_data/disease_approved_drugs.csv` (built from
     openFDA `indications_and_usage` text), applied on top of pass 5b's
     output (`drug_signals_master_reclassified.csv`).

9. **Repurposing candidates** — `rebuild_repurposing_candidates.py`
   Ranks drug repurposing candidates from the fully corrected data.

## Setup

```bash
pip install -r requirements.txt
```

Run the pipeline stages in the order above; each stage's script docstring
documents its exact inputs/outputs and any known data-quality caveats.

## Reference

Karas, C., Qu, Y., Xu, Q., & Zhu, X. (2022). Experiments with LDA and
Top2Vec for embedded topic discovery on social media data: A case study of
cystic fibrosis. *Frontiers in Artificial Intelligence*, 5, 948313.
https://doi.org/10.3389/frai.2022.948313
