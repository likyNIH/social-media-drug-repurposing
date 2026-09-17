"""
build_topic_models.py

Generalizes scleroderma_topic_model.py + scleroderma_wordclouds.py across
all 20 scraped rare disease communities, using Top2Vec with doc2vec
embedding only (no LDA). Per Karas et al. (2022) -- the paper this whole
pipeline is based on -- Top2Vec/doc2vec produced higher coherence (0.672 vs
~0.495) and more interpretable topics than LDA, and unlike LDA, needs no
heavy preprocessing (no stopword removal, lemmatization, or a hand-tuned
topic count), which makes it far cheaper to run across 20 diseases than
replicating the full LDA prep pipeline 20 times.

This produces the inputs needed for Step 7 (manual topic labeling) --
keyword lists and word clouds per topic per disease. It deliberately stops
there: identifying which topics are clinically meaningful (vs. "daily
life"/support chatter) requires a human to look at the word clouds, per
the paper's own methodology. Once you've labeled the medication/symptom
topics for a disease, use each document's dominant-topic assignment
(saved in <subreddit>_doc_topics.json) to select which posts to feed to
the LLM extraction step (drug_repurposing_prompt.py), rather than running
extraction over every scraped post.

Skips diseases whose corpus is too small for Top2Vec to find any topics
(logged, not a fatal error -- expected for the smallest communities).

Outputs per subreddit, under scraped_docs/topic_models/<subreddit>/:
  topics.json          top keywords + scores per topic
  topics.csv           same, flat CSV
  doc_topics.json      each document's id + assigned topic_id
  wordclouds/topic_NN.png, all_topics.png

Usage: python build_topic_models.py
"""

import json
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from top2vec import Top2Vec

import scrape_rare_disease_subreddits as scraper

DATA_DIR = scraper.OUT_DIR  # scraped_docs/
OUT_ROOT = DATA_DIR / "topic_models"
OUT_ROOT.mkdir(exist_ok=True)

TOP_N_WORDS = 20
WC_WIDTH, WC_HEIGHT, WC_BG = 800, 400, "white"


def load_raw_docs(subreddit: str) -> tuple[list[str], list[str]]:
    """Load full_text per post, deduped, alongside post ids (paper: dedup
    to avoid bias). full_text is already URL/email/bracket-cleaned and
    contraction-expanded by the scraper's clean_text()."""
    path = DATA_DIR / f"{subreddit}_raw.jsonl"
    seen = set()
    doc_ids, docs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (d.get("full_text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            doc_ids.append(d.get("id"))
            docs.append(text)
    return doc_ids, docs


def make_wordcloud(freq: dict) -> WordCloud:
    return WordCloud(width=WC_WIDTH, height=WC_HEIGHT, background_color=WC_BG,
                      collocations=False).generate_from_frequencies(freq)


def save_grid(images_and_titles, ncols, out_path, suptitle):
    n = len(images_and_titles)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, (img, title) in zip(axes, images_and_titles):
        ax.imshow(img, interpolation="bilinear")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_subreddit(subreddit: str, disease_name: str, min_cluster_size: int | None = None,
                       force: bool = False) -> str:
    out_dir = OUT_ROOT / subreddit
    if (out_dir / "topics.json").exists() and not force:
        return "already done"

    doc_ids, docs = load_raw_docs(subreddit)
    if len(docs) < 50:
        return f"skipped -- only {len(docs)} unique documents, too small for Top2Vec"

    # min_cluster_size is HDBSCAN's actual lever for topic granularity
    # (Top2Vec's own default is 15). Some communities collapsed to just 2
    # topics at the default even at a comparable corpus size to others that
    # split into 15-25+ -- lowering this forces finer splits for those.
    hdbscan_args = {"min_cluster_size": min_cluster_size} if min_cluster_size else None

    try:
        model = Top2Vec(documents=docs, embedding_model="doc2vec",
                         speed="learn", workers=4, min_count=5,
                         hdbscan_args=hdbscan_args)
    except Exception as e:
        return f"failed -- Top2Vec error: {e}"

    num_topics = model.get_num_topics()
    if num_topics == 0:
        return "skipped -- Top2Vec found 0 topics"

    out_dir.mkdir(parents=True, exist_ok=True)
    wc_dir = out_dir / "wordclouds"
    wc_dir.mkdir(exist_ok=True)

    topic_words, word_scores, topic_ids = model.get_topics()
    topics = []
    grid = []
    for words, scores, topic_id in zip(topic_words, word_scores, topic_ids):
        top_words = words[:TOP_N_WORDS].tolist()
        top_scores = scores[:TOP_N_WORDS].tolist()
        topics.append({
            "topic_id": int(topic_id),
            "keywords": [{"word": w, "score": round(float(s), 6)} for w, s in zip(top_words, top_scores)],
            "top_words": top_words,
        })
        freq = {w: float(s) for w, s in zip(words[:50].tolist(), scores[:50].tolist())}
        wc = make_wordcloud(freq)
        wc.to_file(str(wc_dir / f"topic_{topic_id:02d}.png"))
        grid.append((wc.to_image(), f"Topic {topic_id}"))

    save_grid(grid, ncols=3, out_path=wc_dir / "all_topics.png",
              suptitle=f"Top2Vec Topics -- r/{subreddit} ({disease_name})")

    with open(out_dir / "topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)
    with open(out_dir / "topics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic_id", "rank", "word", "score"])
        for t in topics:
            for rank, kw in enumerate(t["keywords"], 1):
                writer.writerow([t["topic_id"], rank, kw["word"], kw["score"]])

    # per-document topic assignment, needed later to select which posts
    # to feed to LLM extraction once topics are manually labeled
    doc_topic_ids, doc_dists = model.get_documents_topics(doc_ids=list(range(len(docs))))[:2]
    doc_topics = [{"id": doc_ids[i], "topic_id": int(doc_topic_ids[i])} for i in range(len(docs))]
    with open(out_dir / "doc_topics.json", "w", encoding="utf-8") as f:
        json.dump(doc_topics, f, indent=2)

    return f"ok -- {num_topics} topics, {len(docs)} documents"


def main():
    print(f"{len(scraper.COMMUNITIES)} communities queued for topic modeling\n")
    for i, community in enumerate(scraper.COMMUNITIES, 1):
        sub = community["subreddit"]
        print(f"[{i}/{len(scraper.COMMUNITIES)}] r/{sub} ({community['disease_name']})...", flush=True)
        result = process_subreddit(sub, community["disease_name"])
        print(f"  {result}")


if __name__ == "__main__":
    main()
