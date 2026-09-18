"""
rank_disease_subreddits.py

Searches for subreddits matching each disease in the GARD rare disease list,
ranks them by subscriber count, and saves results to CSV.

Uses the arctic-shift API (https://arctic-shift.photon-reddit.com) instead of
Reddit's own search, since Reddit's public subreddits/search.json endpoint
returns 403 from this network (blocked as datacenter/bot traffic).

arctic-shift's /api/subreddits/search only matches on a *prefix* of the
subreddit's name (not full-text search across title/description), so a query
like "postural orthostatic tachycardia syndrome" will never find r/POTS.
To work around this, each disease is searched using several candidate
prefixes instead of one:
  1. Acronym-like synonyms (e.g. "POTS", "EDS") — most likely to literally
     be the subreddit's name.
  2. The full disease name / synonym with spaces and punctuation stripped
     (e.g. "ehlersdanlossyndrome").
  3. Individual significant words from the name/synonyms (generic terms like
     "syndrome" or "disease" are excluded — they're too common to be a useful
     subreddit-name prefix and would mostly waste API calls).
Candidates are tried in that priority order and stop early once a
strong (high-subscriber) relevant match is found, to limit API calls for
diseases that already have an obvious community.

Strategy per disease:
  1. Try candidate prefixes in priority order (see above), up to MAX_CANDIDATES.
  2. Take the highest-subscriber result that passes a relevance check
     (subreddit name or description must contain a word from the disease
     name or any synonym).
  3. Skip communities below MIN_SUBSCRIBERS.
  4. Stop trying further candidates once a match >= EARLY_EXIT_SUBSCRIBERS
     is found.

Resumable: progress is appended to a JSONL file after each disease so the
script can be interrupted and restarted without repeating work.

Outputs:
  /ncats/users/liky/social_media_data/disease_ranking_progress.jsonl  (intermediate, one line per disease)
  /ncats/users/liky/social_media_data/disease_subreddit_ranking.csv   (final ranked list, diseases with subs only)
"""

import csv
import json
import os
import time
import re
from pathlib import Path

import requests

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

DATA_DIR       = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
DISEASES_CSV   = DATA_DIR / "all_diseases.csv"
PROGRESS_FILE  = DATA_DIR / "disease_ranking_progress.jsonl"
OUT_CSV        = DATA_DIR / "disease_subreddit_ranking.csv"

ARCTIC_BASE_URL = "https://arctic-shift.photon-reddit.com"

MIN_SUBSCRIBERS       = 200    # ignore communities smaller than this
SEARCH_LIMIT          = 5      # top N results per candidate query
DELAY_S               = 0.3    # seconds between requests
MAX_CANDIDATES        = 6      # max candidate prefixes to try per disease
EARLY_EXIT_SUBSCRIBERS = 5000  # stop trying more candidates once matched this big

GENERIC_WORDS = {
    "syndrome", "syndromes", "disease", "diseases", "disorder", "disorders",
    "deficiency", "deficiencies", "congenital", "hereditary", "familial",
    "idiopathic", "primary", "secondary", "type", "types", "associated",
    "related", "condition", "conditions", "complex", "spectrum", "group",
    # Clinical-qualifier boilerplate: GARD subtype names routinely append
    # phrases like "severe form", "late onset", "classic variant" — these
    # describe severity/timing, not the disease itself, and are common
    # enough (831 "onset", 330 "form", 265 "variant" occurrences across
    # all_diseases.csv) to be a real source of coincidental corroboration
    # (e.g. "form" from "severe perinatal form" matching r/PickAnAndroidForMe).
    "onset", "form", "forms", "variant", "variants",
}

# GARD's "synonyms" column sometimes contains a full clinical description
# rather than a clean alternate name (e.g. "congenital ablepharon, absent
# eyelashes/eyebrows, macrostomia, ... and other systemic anomalies"). Plain
# English filler words from text like that carry no disease-specific signal
# but are common enough to coincidentally appear in unrelated subreddits'
# descriptions, so they're excluded the same way GENERIC_WORDS excludes
# disease terminology.
ENGLISH_STOPWORDS = {
    "other", "others", "with", "without", "from", "into", "onto", "about",
    "this", "that", "these", "those", "have", "having", "were", "been",
    "some", "such", "than", "then", "also", "very", "much", "many", "most",
    "more", "less", "least", "each", "every", "both", "either", "neither",
    "when", "where", "while", "which", "what", "whom", "whose", "there",
    "their", "they", "them", "your", "yours", "over", "under", "between",
    "among", "during", "before", "after", "again", "further", "once",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def parse_synonyms(raw: str) -> list[str]:
    """Parse the pipe-separated synonyms column into a list."""
    return [s.strip() for s in raw.split("|") if s.strip()]


def slugify(text: str) -> str:
    """Lowercase and strip everything but letters/digits, for prefix matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def significant_words(text: str) -> list[str]:
    """Words longer than 3 chars, excluding generic disease terminology and English filler words."""
    words = re.split(r"\W+", text.lower())
    return [w for w in words if len(w) > 3 and w not in GENERIC_WORDS and w not in ENGLISH_STOPWORDS]


def shares_root(a: str, b: str) -> bool:
    """
    Rough check for whether two words are just different grammatical forms
    of the same root (e.g. "nail"/"nails", "absent"/"absence") rather than
    genuinely distinct words. Not a real stemmer — just a shared-prefix
    heuristic, deliberately conservative so it doesn't merge unrelated short
    words that happen to start the same way (e.g. "type"/"tyre").
    """
    if a == b:
        return True
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix_len += 1
    shortest = min(len(a), len(b))
    return prefix_len >= 4 and prefix_len >= shortest - 2


def build_search_candidates(disease_name: str, synonyms: list[str]) -> list[str]:
    """
    Build an ordered list of subreddit-name-prefix candidates to try for a
    disease. Order matters: acronym-like synonyms are tried first since
    they're most likely to literally be the subreddit's name, followed by
    full name/synonym slugs, then individual significant words.
    """
    acronyms, slugs, words = [], [], []

    def add_words(text: str):
        for w in significant_words(text):
            if w not in words:
                words.append(w)

    for syn in synonyms:
        raw = syn.strip()
        if not raw:
            continue
        if raw.isupper() and 2 <= len(raw) <= 6:
            slug = slugify(raw)
            if slug and slug not in acronyms:
                acronyms.append(slug)
        else:
            slug = slugify(raw)
            if slug and len(slug) > 3 and slug not in slugs:
                slugs.append(slug)
        add_words(raw)

    name_slug = slugify(disease_name)
    if name_slug and len(name_slug) > 3 and name_slug not in slugs:
        slugs.append(name_slug)
    add_words(disease_name)

    return (acronyms + slugs + words)[:MAX_CANDIDATES]


COMPOUND_MIN_LEN = 8  # full-name/synonym slugs at least this long are specific enough
                       # to count as a match on their own (see is_relevant)


def classify_match(sub_name: str, sub_desc: str, disease_name: str, synonyms: list[str]) -> str | None:
    """
    Heuristic: is this subreddit plausibly about this disease, and how
    confident is that guess? Returns a match-type string, or None if there's
    no plausible connection at all.

    A single generic word coincidentally collides with unrelated subreddits
    surprisingly often — "marie" (from "Charcot-Marie-Tooth") matches inside
    "Marietta", "als" matches inside "alsace", "lateral" matches inside
    "LaterAlligator". No single length threshold fixes this, since real
    compound subreddit names (e.g. "charcotmarietooth") are built the same
    way real coincidences are (concatenated words with no delimiter).

    The distinguishing signal is corroboration: a coincidence produces one
    hit, a genuine disease-specific match usually produces several (the real
    subreddit's name or description echoes multiple distinct disease-related
    terms together). So a partial/substring match only counts once >=2
    distinct terms agree; a single term is only trusted on its own when it's
    unambiguous — an acronym, a long full-name/synonym slug, or the disease
    phrase spelled out in the description.

    A single generic word *exactly* naming the subreddit (e.g. "dwarfism",
    "skeletal") is also allowed through, but flagged as the low-confidence
    "single_word_exact" type — this is structurally identical to genuine
    coincidences like "mosaic" or "nuclear" (both are just one common word
    from a synonym matching a subreddit's full name), and the two can't be
    told apart algorithmically. It's surfaced rather than silently dropped
    so a human can sort real umbrella categories from coincidences during
    the pipeline's existing manual review step, rather than losing real
    umbrella-community matches entirely.
    """
    sub_name = sub_name.lower()
    sub_desc = sub_desc.lower()

    words, acronyms, compounds, phrases = [], [], [], [disease_name]

    def add_word(w: str) -> None:
        # Skip words that are just a different grammatical form of one
        # already collected (plural/singular like "nail"/"nails", or
        # adjective/noun alternations like "absent"/"absence") — otherwise
        # they'd count as 2 independent corroborating terms below when
        # they're really one piece of evidence. This isn't a real stemmer,
        # just a shared-prefix heuristic; it's deliberately conservative
        # (requires a long shared prefix) so it doesn't merge genuinely
        # distinct short words like "type"/"tyre".
        for existing in words:
            if shares_root(w, existing):
                return
        words.append(w)

    for w in significant_words(disease_name):
        add_word(w)
    for syn in synonyms:
        for w in significant_words(syn):
            add_word(w)
        slug = slugify(syn)
        if slug and len(slug) <= 3:
            acronyms.append(slug)
        phrases.append(syn)

    for text in [disease_name] + synonyms:
        slug = slugify(text)
        if len(slug) >= COMPOUND_MIN_LEN:
            compounds.append(slug)

    # 1. acronym or full compound name/synonym equals the subreddit exactly — unambiguous
    if sub_name in set(acronyms) | set(compounds):
        return "acronym_or_compound_exact"

    # 2. a full disease-name/synonym slug is specific enough to substring-match alone
    for c in compounds:
        if c in sub_name:
            return "compound_substring"

    # 3. corroborated partial match: >=2 distinct significant words both
    #    appear in the (concatenated, delimiter-free) subreddit name
    if len({w for w in words if w in sub_name}) >= 2:
        return "corroborated_name"

    # 4. an acronym mentioned as a standalone word in the description —
    #    sparse community bios often only say "support group for ALS" once
    for a in acronyms:
        if re.search(rf"\b{re.escape(a)}\b", sub_desc):
            return "acronym_in_description"

    # 5. the disease's full name or a synonym written out in the description
    #    (word-boundary, not substring — a short phrase like "ALS" must not
    #    match merely because it's a substring of an unrelated word like
    #    "alsace" appearing in the text).
    #
    #    Restricted to phrases of >=2 words: a genuine multi-word phrase
    #    appearing verbatim in an unrelated description is extremely
    #    unlikely by chance, so it's safe to trust alone. A single-word
    #    "phrase" is just a short synonym in disguise, and GARD synonym
    #    lists routinely include bare single-word synonyms (often the
    #    disease's own acronym, e.g. "TRAPS", "PALE", "SEGA") that
    #    frequently collide with unrelated but very common English words —
    #    "TRAPS" (tumor necrosis factor receptor-associated periodic
    #    syndrome) matched r/traps, "PALE" matched r/palegirls, purely
    #    because those communities' own descriptions happened to use the
    #    same ordinary word. A single-word synonym still gets a chance via
    #    rule 4 (acronym_in_description) or the corroboration rules below.
    for p in phrases:
        p_clean = p.strip().lower()
        if p_clean and len(p_clean.split()) >= 2 and re.search(rf"\b{re.escape(p_clean)}\b", sub_desc):
            return "phrase_in_description"

    # 6. corroborated partial match: >=2 distinct significant words both
    #    appear (as whole words) in the description
    if len({w for w in words if re.search(rf"\b{re.escape(w)}\b", sub_desc)}) >= 2:
        return "corroborated_description"

    # 7. low confidence: a single generic word exactly names the subreddit.
    #    Could be a real umbrella category (r/dwarfism) or a pure coincidence
    #    (r/Mosaic, r/nuclear) — surfaced for manual review either way.
    if sub_name in set(words):
        return "single_word_exact"

    return None


CONFIDENT_MATCH_TYPES = {
    "acronym_or_compound_exact", "compound_substring", "corroborated_name",
    "acronym_in_description", "phrase_in_description", "corroborated_description",
}


def search_subreddits(query: str) -> list[dict]:
    """Call arctic-shift's subreddit search (prefix match). Returns list of subreddit data dicts."""
    try:
        resp = requests.get(
            f"{ARCTIC_BASE_URL}/api/subreddits/search",
            params={"subreddit_prefix": query, "limit": SEARCH_LIMIT},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("data") or []
    except Exception as e:
        print(f"    Request error for {query!r}: {e}")
        return []


def load_progress() -> set[str]:
    """Return set of gard_ids already written to the progress file."""
    if not PROGRESS_FILE.exists():
        return set()
    seen = set()
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["gard_id"])
            except Exception:
                pass
    return seen


def load_diseases() -> list[dict]:
    with open(DISEASES_CSV, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    diseases = load_diseases()
    print(f"Loaded {len(diseases)} diseases from {DISEASES_CSV}")

    seen = load_progress()
    remaining = len(diseases) - len(seen)
    print(f"Progress: {len(seen)} done, {remaining} remaining\n")

    progress_fh = open(PROGRESS_FILE, "a", encoding="utf-8")

    try:
        for i, row in enumerate(diseases):
            gard_id      = row["gardId"]
            disease_name = row["gardName"].strip()

            if gard_id in seen:
                continue

            synonyms = parse_synonyms(row.get("synonyms", ""))
            candidates = build_search_candidates(disease_name, synonyms)

            print(f"[{i+1}/{len(diseases)}] {disease_name}  (candidates: {candidates})")

            # Confident matches (corroborated / acronym / compound) always win over
            # a low-confidence single-word "umbrella" match (see classify_match),
            # even if the umbrella match has more subscribers — e.g. a small but
            # disease-specific community should outrank a large but coincidental
            # single-word collision like r/Mosaic or r/nuclear.
            best, best_umbrella = None, None
            for query in candidates:
                results = search_subreddits(query)
                time.sleep(DELAY_S)

                for sub in results:
                    subs_count = sub.get("subscribers", 0)
                    if subs_count < MIN_SUBSCRIBERS:
                        continue
                    sub_name = sub.get("display_name", "")
                    sub_desc = sub.get("public_description", "")
                    meta = sub.get("_meta") or {}

                    match_type = classify_match(sub_name, sub_desc, disease_name, synonyms)
                    if match_type is None:
                        continue

                    candidate_record = {
                        "gard_id":       gard_id,
                        "disease_name":  disease_name,
                        "subreddit":     sub_name,
                        "subscribers":   subs_count,
                        "num_posts":     meta.get("num_posts") or 0,
                        "num_comments":  meta.get("num_comments") or 0,
                        "description":   sub_desc[:200].replace("\n", " "),
                        "matched_query": query,
                        "match_type":    match_type,
                        "confidence":    "high" if match_type in CONFIDENT_MATCH_TYPES else "low",
                    }

                    if match_type in CONFIDENT_MATCH_TYPES:
                        if best is None or subs_count > best["subscribers"]:
                            best = candidate_record
                    else:
                        if best_umbrella is None or subs_count > best_umbrella["subscribers"]:
                            best_umbrella = candidate_record

                if best and best["subscribers"] >= EARLY_EXIT_SUBSCRIBERS:
                    break

            winner = best or best_umbrella

            if winner:
                print(f"  -> r/{winner['subreddit']} ({winner['subscribers']:,} subscribers, "
                      f"{winner['num_posts']:,} posts, via {winner['matched_query']!r}, {winner['match_type']})")
                record = winner
            else:
                print("  -> no match")
                record = {
                    "gard_id":       gard_id,
                    "disease_name":  disease_name,
                    "subreddit":     None,
                    "subscribers":   0,
                    "num_posts":     0,
                    "num_comments":  0,
                    "description":   "",
                    "matched_query": None,
                    "match_type":    None,
                    "confidence":    None,
                }

            progress_fh.write(json.dumps(record) + "\n")
            progress_fh.flush()

    finally:
        progress_fh.close()

    # ------------------------------------------------------------------
    # Compile final ranked output from progress file
    # ------------------------------------------------------------------

    matches = []
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("subscribers", 0) >= MIN_SUBSCRIBERS:
                    matches.append(rec)
            except Exception:
                pass

    matches.sort(key=lambda x: x["subscribers"], reverse=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "gard_id", "disease_name", "subreddit", "subscribers", "num_posts",
                      "num_comments", "description", "matched_query", "match_type", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(matches, 1):
            writer.writerow({"rank": rank, **r})

    print(f"\nDone. {len(matches)} communities found.")
    print(f"Results saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
