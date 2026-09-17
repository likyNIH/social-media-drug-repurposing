import requests
import json
import time
from pathlib import Path

BASE_URL  = "https://arctic-shift.photon-reddit.com"
SUBREDDIT = "scleroderma"
OUT_DIR   = Path("H:\Documents\scraped_docs")
OUT_DIR.mkdir(exist_ok=True)


def fetch_all_posts(subreddit: str) -> list[dict]:
    posts = []
    after = "2026-04-01"

    while True:
        resp  = requests.get(f"{BASE_URL}/api/posts/search", params={
            "subreddit": subreddit,
            "after":     after,
            "limit":     "auto",
            "sort":      "asc",
            "fields":    "id,title,selftext,created_utc,score,num_comments"
        })
        batch = resp.json().get("data") or []
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
    try:
        after = None
        while True:
            params = {
                "link_id": post_id,
                "limit":   "auto",  # up to 1000 per call
                "sort":    "asc",
            }
            if after:
                params["after"] = after

            resp = requests.get(
                f"{BASE_URL}/api/comments/search",
                params=params,
                timeout=15
            )
            batch = resp.json().get("data") or []
            if not batch:
                break

            for c in batch:
                body = c.get("body", "")
                if body and body not in ("[deleted]", "[removed]"):
                    all_bodies.append(body)

            # paginate using created_utc of last item
            if len(batch) < 100:
                break
            after = batch[-1].get("created_utc")
            time.sleep(0.3)

    except Exception as e:
        print(f"    Warning: could not fetch comments for {post_id}: {e}")

    return " ".join(all_bodies)


def scrape_subreddit(subreddit: str):
    print(f"Fetching posts from r/{subreddit}...")
    posts = fetch_all_posts(subreddit)
    print(f"Total posts: {len(posts)}")

    out_file = OUT_DIR / f"{subreddit}_raw.jsonl"
    with open(out_file, "w") as f:
        for i, post in enumerate(posts):
            if i % 100 == 0:
                print(f"  Fetching comments for post {i+1}/{len(posts)}...")
            comments_text = fetch_comments_for_post(post["id"])
            time.sleep(0.5)

            doc = {
                "id":           post["id"],
                "title":        post.get("title", ""),
                "selftext":     post.get("selftext", ""),
                "comments":     comments_text,
                "full_text":    f"{post.get('title','')} {post.get('selftext','')} {comments_text}".strip(),
                "created_utc":  post.get("created_utc"),
                "score":        post.get("score"),
                "num_comments": post.get("num_comments"),
            }
            f.write(json.dumps(doc) + "\n")

    print(f"Done. Saved to {out_file}")


scrape_subreddit(SUBREDDIT)