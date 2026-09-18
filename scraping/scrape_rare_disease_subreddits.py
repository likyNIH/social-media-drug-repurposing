"""
scrape_rare_disease_subreddits.py

Scrapes full post + comment history from each community in COMMUNITIES using
the arctic-shift API, one JSONL file per subreddit. Generalizes
scrape_scleroderma.py to the shortlist of genuinely disease-specific, rare
(non-umbrella) communities selected from disease_subreddit_ranking_v2.csv.

Resumable at the subreddit level: a community already saved to
scraped_docs/<subreddit>_raw.jsonl is skipped, so an interrupted run can
just be restarted.

Runtime note: unlike the original scleroderma test case (~219 posts), several
of these communities have tens of thousands of posts (Narcolepsy: ~44k,
visualsnow: ~38k) and each post needs its own comments/search pagination
call, so this will run for many hours across the full list -- plan to run
it via sbatch (see submit_rank_disease_subs.sh for the chain-resubmission
pattern used for the ranking script) rather than in an interactive shell.

Usage: python scrape_rare_disease_subreddits.py
"""

import os
import re
import requests
import json
import time
from pathlib import Path

# ------------------------------------------------------------------
# Text cleaning: URL/email/bracket removal + contraction expansion
# (Steps A and B of the planned preprocessing pipeline, applied here so
# scraped output is already clean.)
# ------------------------------------------------------------------

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(https?://[^\)]*\)")
URL_RE           = re.compile(r"(?:https?://|www\.)\S+")
EMAIL_RE         = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
BRACKETED_RE     = re.compile(r"\[[^\]]*\]")
WHITESPACE_RE    = re.compile(r"[ \t]+")

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "isn't": "is not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not", "shouldn't": "should not",
    "wasn't": "was not", "weren't": "were not", "won't": "will not", "wouldn't": "would not",
    "couldn't've": "could not have", "shouldn't've": "should not have", "wouldn't've": "would not have",
    "could've": "could have", "should've": "should have", "would've": "would have",
    "might've": "might have", "must've": "must have",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "it'll": "it will", "we'll": "we will", "they'll": "they will", "that'll": "that will",
    "there'll": "there will", "who'll": "who will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "it'd": "it would", "we'd": "we would", "they'd": "they would", "that'd": "that would",
    "there'd": "there would", "who'd": "who would",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "there's": "there is", "here's": "here is", "what's": "what is", "who's": "who is",
    "let's": "let us", "y'all": "you all", "'tis": "it is", "'twas": "it was",
}
# match longest keys first so e.g. "couldn't've" wins over "couldn't"
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in CONTRACTIONS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _expand_match(m: re.Match) -> str:
    # .get() with the original text as fallback, not a direct dict lookup:
    # confirmed in production that some Unicode characters (e.g. Turkish
    # capital "I" with a dot, U+0130) case-fold via .lower() into a string
    # that doesn't match any plain-ASCII dictionary key (it becomes "i" plus
    # a separate combining dot-above character, not plain "i"). A crash here
    # kills a multi-day scraping job over one stray character in one comment
    # out of hundreds of thousands -- leaving the text unexpanded is a far
    # better failure mode than an unhandled exception.
    matched = m.group(0)
    expansion = CONTRACTIONS.get(matched.lower())
    if expansion is None:
        return matched
    return expansion.capitalize() if matched[0].isupper() else expansion


def expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(_expand_match, text)


def clean_text(text: str) -> str:
    """
    URL/email/bracket removal + contraction expansion, applied to raw post
    titles, selftext, and individual comment bodies before they're stored.
    """
    if not text:
        return text
    text = text.replace("’", "'").replace("‘", "'")  # curly -> straight apostrophe
    text = MARKDOWN_LINK_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = BRACKETED_RE.sub("", text)
    text = expand_contractions(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


BASE_URL = "https://arctic-shift.photon-reddit.com"
DATA_DIR = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
OUT_DIR  = DATA_DIR / "scraped_docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EARLIEST_DATE = "2005-01-01"  # before Reddit existed -- fetches full history

# Genuinely disease-specific (non-umbrella), rare communities selected from
# disease_subreddit_ranking_v2.csv after removing false positives, umbrella/
# broad-category matches, and non-rare conditions (tuberculosis, preeclampsia,
# anal fistula, myocarditis, avascular necrosis, thyroid cancer, IgA
# nephropathy). r/Hemochromatosis is borderline rarity -- kept here pending
# a final call, remove the entry below if you decide to drop it.
COMMUNITIES = [
    {"gard_id": "GARD:0022460", "disease_name": "narcolepsy",                              "subreddit": "Narcolepsy"},
    {"gard_id": "GARD:0012062", "disease_name": "visual snow syndrome",                    "subreddit": "visualsnow"},
    {"gard_id": "GARD:0006233", "disease_name": "cystic fibrosis",                         "subreddit": "CysticFibrosis"},
    {"gard_id": "GARD:0007805", "disease_name": "trigeminal neuralgia",                    "subreddit": "TrigeminalNeuralgia"},
    {"gard_id": "GARD:0024434", "disease_name": "hereditary hemochromatosis",              "subreddit": "Hemochromatosis"},
    {"gard_id": "GARD:0008737", "disease_name": "idiopathic hypersomnia",                  "subreddit": "idiopathichypersomnia"},
    {"gard_id": "GARD:0007122", "disease_name": "myasthenia gravis",                       "subreddit": "MyastheniaGravis"},
    {"gard_id": "GARD:0025213", "disease_name": "adult glioblastoma",                      "subreddit": "glioblastoma"},
    {"gard_id": "GARD:0010418", "disease_name": "hemophilia",                              "subreddit": "Hemophilia"},
    {"gard_id": "GARD:0018705", "disease_name": "scleroderma",                             "subreddit": "scleroderma"},
    {"gard_id": "GARD:0007756", "disease_name": "thalassemia",                             "subreddit": "thalassemia"},
    {"gard_id": "GARD:0006987", "disease_name": "mastocytosis",                            "subreddit": "mastcelldisease"},
    {"gard_id": "GARD:0007607", "disease_name": "sarcoidosis",                             "subreddit": "sarcoidosis"},
    {"gard_id": "GARD:0005871", "disease_name": "autoimmune hepatitis",                    "subreddit": "autoimmunehepatitis"},
    {"gard_id": "GARD:0004531", "disease_name": "proximal spinal muscular atrophy",        "subreddit": "spinalmuscularatrophy"},
    {"gard_id": "GARD:0005725", "disease_name": "acromegaly",                              "subreddit": "acromegaly"},
    {"gard_id": "GARD:0009342", "disease_name": "lip and oral cavity carcinoma",           "subreddit": "oralcancer"},
    {"gard_id": "GARD:0011971", "disease_name": "renal nutcracker syndrome",               "subreddit": "NutcrackerSyndrome"},
    {"gard_id": "GARD:0027044", "disease_name": "long QT syndrome",                        "subreddit": "LongQTSyndrome"},
    {"gard_id": "GARD:0000431", "disease_name": "microtia",                                "subreddit": "Microtia"},
]


REQUEST_RETRIES = 3          # confirmed by testing: arctic-shift occasionally
REQUEST_RETRY_BACKOFF_S = 5  # returns an empty batch transiently, not just at
                             # genuine end-of-data -- an unretried empty batch
                             # in fetch_all_posts would truncate a subreddit's
                             # entire remaining history silently. Also used to
                             # retry actual request exceptions (timeouts,
                             # connection errors) in fetch_comments_for_post --
                             # confirmed in production that ~0.5% of posts
                             # were permanently losing their comments to
                             # transient network errors that were caught but
                             # never retried.


def fetch_all_posts(subreddit: str) -> list[dict]:
    posts = []
    after = EARLIEST_DATE

    while True:
        batch = None
        for attempt in range(REQUEST_RETRIES):
            resp = requests.get(
                f"{BASE_URL}/api/posts/search",
                params={
                    "subreddit": subreddit,
                    "after":     after,
                    "limit":     "auto",
                    "sort":      "asc",
                    "fields":    "id,title,selftext,created_utc,score,num_comments",
                },
                timeout=15,
            )
            batch = resp.json().get("data") or []
            if batch:
                break
            if attempt < REQUEST_RETRIES - 1:
                print(f"  Empty batch (attempt {attempt+1}/{REQUEST_RETRIES}), retrying...")
                time.sleep(REQUEST_RETRY_BACKOFF_S)

        if not batch:
            break

        posts.extend(batch)
        after = batch[-1]["created_utc"]
        print(f"  Fetched {len(posts)} posts so far (last: {after})")
        time.sleep(1)

    return posts


def fetch_comments_for_post(post_id: str) -> str:
    """
    Fetch all comments for a post using comments/search with pagination.
    comments/tree does not support the 'fields' param and returns 400.
    """
    all_bodies = []
    after = None
    while True:
        params = {
            "link_id": post_id,
            "limit":   "auto",  # up to 1000 per call
            "sort":    "asc",
        }
        if after:
            params["after"] = after

        # Retries cover both a transiently empty (but successful) response
        # AND an actual request exception (timeout, connection reset, bad
        # JSON) -- previously an exception on any page gave up on the whole
        # post immediately with no retry, which was confirmed to silently
        # and permanently lose comments on ~0.5% of posts to nothing more
        # than a slow 15s response.
        batch = None
        for attempt in range(REQUEST_RETRIES):
            try:
                resp = requests.get(
                    f"{BASE_URL}/api/comments/search",
                    params=params,
                    timeout=15
                )
                batch = resp.json().get("data") or []
            except Exception as e:
                if attempt < REQUEST_RETRIES - 1:
                    time.sleep(REQUEST_RETRY_BACKOFF_S)
                    continue
                print(f"    Warning: could not fetch comments for {post_id}: {e}")
                return " ".join(all_bodies)
            if batch:
                break
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(REQUEST_RETRY_BACKOFF_S)

        if not batch:
            break

        for c in batch:
            body = c.get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                all_bodies.append(clean_text(body))

        # paginate using created_utc of last item
        if len(batch) < 100:
            break
        after = batch[-1].get("created_utc")
        time.sleep(0.3)

    return " ".join(all_bodies)


def scrape_subreddit(subreddit: str, out_dir: Path):
    print(f"Fetching posts from r/{subreddit}...")
    posts = fetch_all_posts(subreddit)
    print(f"Total posts: {len(posts)}")

    # Write to a .tmp file and only rename to the final path once every post
    # has been written. main()'s resumability check treats the final path's
    # existence as "fully scraped" -- writing straight to it would mean a
    # subreddit killed mid-scrape (very possible: some of these have tens of
    # thousands of posts) leaves behind a partial file that looks complete
    # and gets silently skipped forever on resume.
    out_file = out_dir / f"{subreddit}_raw.jsonl"
    tmp_file = out_dir / f"{subreddit}_raw.jsonl.tmp"

    # Resume from a partial .tmp file left behind by a crash or kill instead
    # of re-fetching comments for posts already saved -- confirmed necessary
    # in production: a crash partway through Narcolepsy's ~44,700 posts would
    # otherwise have thrown away tens of thousands of already-fetched posts.
    already_done_ids = set()
    if tmp_file.exists():
        with open(tmp_file, encoding="utf-8") as f:
            for line in f:
                try:
                    already_done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass  # last line may be a partial write from a hard kill
        if already_done_ids:
            print(f"  Resuming: {len(already_done_ids)} posts already saved in {tmp_file.name}")

    with open(tmp_file, "a") as f:
        for i, post in enumerate(posts):
            if post["id"] in already_done_ids:
                continue
            if i % 100 == 0:
                print(f"  Fetching comments for post {i+1}/{len(posts)}...")
            comments_text = fetch_comments_for_post(post["id"])
            time.sleep(0.5)

            title = clean_text(post.get("title", ""))
            selftext = clean_text(post.get("selftext", ""))

            doc = {
                "id":           post["id"],
                "title":        title,
                "selftext":     selftext,
                "comments":     comments_text,
                "full_text":    f"{title} {selftext} {comments_text}".strip(),
                "created_utc":  post.get("created_utc"),
                "score":        post.get("score"),
                "num_comments": post.get("num_comments"),
            }
            f.write(json.dumps(doc) + "\n")
            f.flush()

    tmp_file.rename(out_file)
    print(f"Done. Saved to {out_file}")


def main():
    print(f"{len(COMMUNITIES)} communities queued for scraping\n")

    for i, community in enumerate(COMMUNITIES, 1):
        subreddit = community["subreddit"]
        out_file = OUT_DIR / f"{subreddit}_raw.jsonl"

        if out_file.exists():
            print(f"[{i}/{len(COMMUNITIES)}] r/{subreddit} already scraped, skipping ({out_file})")
            continue

        print(f"[{i}/{len(COMMUNITIES)}] {community['disease_name']} -> r/{subreddit}")
        scrape_subreddit(subreddit, OUT_DIR)
        print()


if __name__ == "__main__":
    main()
