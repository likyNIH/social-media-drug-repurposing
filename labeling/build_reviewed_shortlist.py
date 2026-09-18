"""
build_reviewed_shortlist.py

Applies a manual audit of "disease ranking shortlist.csv" (156 rows,
confidence=="high" matches deduped by subreddit): removes confirmed false
positives (proper-noun/brand/hobby-community collisions unrelated to the
disease) and flags matches that are topically adjacent but medically
distinct from the community's actual focus, for a second look before
scraping.

Leaves "disease ranking shortlist.csv" untouched. Writes:
  disease ranking shortlist_reviewed.csv

Usage: python build_reviewed_shortlist.py
"""

import csv
import os
from pathlib import Path

DATA_DIR    = Path(os.environ.get(
    "SM_DATA_DIR", str(Path(__file__).resolve().parent.parent / "social_media_data")))
IN_CSV      = DATA_DIR / "disease ranking shortlist.csv"
OUT_CSV     = DATA_DIR / "disease ranking shortlist_reviewed.csv"

# Confirmed false positives: coincidental word overlap with an unrelated
# proper noun, brand, or hobby community -- not the disease at all.
REMOVE = {
    "GARD:0003485": "r/BuenosAires is the Argentine capital, not the eponym",
    "GARD:0013712": "r/alphaandbetausers is a product beta-testing community",
    "GARD:0000353": "r/SkullAndBonesGame is a Ubisoft video game",
    "GARD:0007305": "r/CerebroDigital is a Spanish-language STEM education community",
    "GARD:0020616": "r/Purebarre is barre fitness classes",
    "GARD:0020331": "r/IsolatedVocals is a music/vocal-isolation community",
    "GARD:0018254": "r/ClearwaterFl is Clearwater, Florida",
    "GARD:0022321": "r/MiddleEastHistory is Middle Eastern history/geography",
    "GARD:0001196": "r/CerebellarHypoplasia here is about pets/animals with CH, not humans",
    "GARD:0021771": "r/Canal_Central is a Brazilian YouTube channel fan community",
    "GARD:0004619": "r/Hypophosphatasia is a different disease (bone mineralization vs RBC disorder)",
    "GARD:0026253": "r/stalker_ANOMALY is unrelated/unclear, likely game or stalking-themed",
    "GARD:0005966": "r/MediterraneanFever is for genetic FMF, not bacterial brucellosis",
    "GARD:0020590": "r/CAIhadastroke is an AI-fail meme community (\"CAIHA\" acronym collision)",
    "GARD:0018516": "r/AbnormalBehavior is a shock-content/viral-video community",
    "GARD:0004085": "r/prurigonodularis is a different skin disease (itchy nodules vs genetic poikiloderma)",
    "GARD:0018760": "r/Sanchez_houdaa is a fan community for a person named Sanchez Houda",
}

# Real medical connection, but the community's actual focus is a different
# or broader condition than the matched disease -- worth a manual check
# before treating it as disease-specific.
VERIFY = {
    "GARD:0016960": "hereditary motor neuropathy != traumatic spinal cord injury",
    "GARD:0009578": "acute obstetric emergency != chronic lifestyle NAFLD",
    "GARD:0002438": "unclear real connection to hiatal hernia",
    "GARD:0020123": "general stem-cell-research community, not disease-specific",
    "GARD:0025071": "community is for the athletic-trainer profession, not patients",
    "GARD:0001496": "different valve pathology (stenosis vs prolapse)",
    "GARD:0010401": "different retinal disease, different genetics (RP vs macular degeneration)",
    "GARD:0016685": "tagline \"for Redditors with extra large hearts\" reads like a pun sub",
    "GARD:0011004": "possibly a different specific pigmentary diagnosis",
}


def main():
    with open(IN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["review_note"]
    kept = []
    removed_count = 0

    for row in rows:
        gid = row["gard_id"]
        if gid in REMOVE:
            removed_count += 1
            continue
        row["review_note"] = f"VERIFY: {VERIFY[gid]}" if gid in VERIFY else ""
        kept.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    verify_count = sum(1 for r in kept if r["review_note"])
    print(f"Read {len(rows)} rows from {IN_CSV.name}")
    print(f"Removed {removed_count} confirmed false positives")
    print(f"Flagged {verify_count} for manual verification (kept, marked in review_note)")
    print(f"{len(kept)} rows written to {OUT_CSV.name}")


if __name__ == "__main__":
    main()
