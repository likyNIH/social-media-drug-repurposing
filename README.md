# Rare Disease Drug Repurposing via Reddit Topic Modeling

Identifies drug repurposing signals in rare disease patient communities on
Reddit by combining topic modeling (Top2Vec) with LLM-based structured
extraction (Gemma3-27B) across 20 rare disease subreddits. Methodology
adapted from:

> Karas, Qu, Xu & Zhu (2022). "Experiments with LDA and Top2Vec for embedded
> topic discovery on social media data: A case study of cystic fibrosis."
> *Frontiers in Artificial Intelligence*, 5:948313.

## Data

The raw scraped Reddit data (individual posts/comments, verbatim patient
quotes) and the full per-community LLM extraction outputs are **not**
included: some files exceed GitHub's size limits, and that underlying data
is not appropriate to redistribute in bulk. Every script below reads from
and writes to `scraped_docs/`, which you regenerate locally by running the
pipeline in order.

The `data/` folder contains only aggregate, non-quote-level outputs safe to
share:

- `all_diseases.csv` — the full GARD rare disease reference list
  (gardId/gardName/synonyms), public NCATS data, no patient content.
- `sm_term_summary.md`, `sm_term_summary_test_219posts.md` — term-frequency
  summaries (category, term group, record/mention counts, and short
  matched surface forms like `SSc | systemic sclerosis`) for the
  scleroderma community, at two different scrape sizes. No verbatim posts.
- `topic_summary_full.md` — Top2Vec topic word clusters (top keywords per
  topic) across all scraped communities. Aggregate word lists only, no
  verbatim posts.

Note: `disease_approved_drugs.csv`, the openFDA-derived approved-drug
lookup table used by `apply_disease_reference.py` (step 5c below), is
maintained outside this repo and is not currently checked in here.

## Pipeline

Scripts are organized by pipeline stage. Paths below are relative to the
repo root.

1. **Rank communities** — `scraping/rank_disease_subreddits.py`
   Ranks all ~6,000 GARD rare diseases by Reddit community size via the
   arctic-shift API. Output feeds the manually-curated shortlist in
   `scraping/scrape_rare_disease_subreddits.py`'s `COMMUNITIES` list.

2. **Scrape** — `scraping/scrape_rare_disease_subreddits.py`
   Pulls full post + comment history per community from the arctic-shift
   public API (no auth required) into `scraped_docs/<subreddit>_raw.jsonl`.
   Resumable at the subreddit level.
   `scraping/backfill_missing_comments.py` re-fetches posts where a failed
   comment fetch was silently recorded as "no comments."

3. **Topic modeling** — `topic_modeling/build_topic_models.py`
   Runs Top2Vec (doc2vec embedding) across all scraped communities —
   Karas et al. found Top2Vec/doc2vec outperformed LDA on this kind of
   data and needs no manual K selection. Also generates word clouds per
   topic.

4. **Topic labeling** — `topic_modeling/label_topics.py` +
   `topic_modeling/topic_labeling_prompt.py`
   LLM-assisted labeling of each topic as clinical/non-clinical, using the
   same vLLM + Gemma3-27b infrastructure as the extraction step.

5. **Filter for extraction** — `extraction/filter_documents_for_extraction.py`
   Keeps only documents whose topic was labeled clinically relevant,
   written to `scraped_docs/filtered_for_extraction/`.

6. **Drug signal extraction** — `extraction/run_drug_extraction.py` +
   `extraction/drug_repurposing_prompt.py`
   LLM extraction of structured drug mentions (name, normalized name,
   label status, mention type, outcome, supporting quote) per post, across
   all communities.

7. **Aggregation** — `aggregation/aggregate_drug_signals.py`
   Combines all per-disease outputs into `drug_signals_master.csv`,
   cleaning known extraction-quality issues (cross-contaminated enum
   fields, etc.).

8. **Two-pass label correction:**
   - **5b.** `labeling/build_curation_targets.py` +
     `labeling/apply_curated_labels.py` — manually curated
     approved-indication lookup for each disease's top drugs by mention
     volume, applied back onto the master table.
   - **5c.** `labeling/apply_disease_reference.py` — a second,
     disease-first pass using `disease_approved_drugs.csv` (built from
     openFDA `indications_and_usage` text; see note in Data above),
     applied on top of pass 5b's output
     (`drug_signals_master_reclassified.csv`).

9. **Repurposing candidates** — `aggregation/rebuild_repurposing_candidates.py`
   Ranks drug repurposing candidates from the fully corrected data.

## Setup

```bash
pip install -r requirements.txt
```

All pipeline data (scraped posts, extraction outputs, intermediate files)
reads from and writes to a single data directory, controlled by the
`SM_DATA_DIR` environment variable:

```bash
export SM_DATA_DIR=/path/to/your/data/dir   # defaults to ./social_media_data
```

Run the pipeline stages in the order above; each stage's script docstring
documents its exact inputs/outputs and any known data-quality caveats.

## Reference

Karas, C., Qu, Y., Xu, Q., & Zhu, X. (2022). Experiments with LDA and
Top2Vec for embedded topic discovery on social media data: A case study of
cystic fibrosis. *Frontiers in Artificial Intelligence*, 5, 948313.
https://doi.org/10.3389/frai.2022.948313
