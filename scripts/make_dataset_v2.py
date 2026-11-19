"""Build dataset v2 and write it to disk.

Run from the project root:
    python -m scripts.make_dataset_v2 data/v2/build                  # real DistilBERT token counts
    python -m scripts.make_dataset_v2 data/v2/build --count words    # quick check without the model
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from scripts.build_v2 import build_dataset, write_dataset
from scripts.documents import document_problems, load_documents
from scripts.public_pools import load_alpaca, load_dolly

ROOT = Path(__file__).resolve().parent.parent
BANDS = ["1-32", "32-64", "64-128", "128-256", "256-512", "512+"]


def token_counter(kind):
    """Content tokens (no [CLS] or [SEP]); the counts add up piece by piece, as embed needs."""
    if kind == "words":
        return lambda text: len(text.split())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", choices=["distilbert", "words"], default="distilbert")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    docs, skipped = load_documents(ROOT / "data" / "v2" / "docs_test.csv")
    problems = [(d["id"], p) for d in docs for p in document_problems(d)]
    if problems or skipped:
        sys.exit(f"documents need fixing first: problems={problems} skipped={skipped}")
    frames = json.load(open(ROOT / "data" / "v2" / "frames.json"))
    rows = load_dolly(ROOT / "data" / "external" / "dolly15k.jsonl") + load_alpaca(ROOT / "data" / "external" / "alpaca.json")
    with open(ROOT / "data" / "final" / "dataset_train.csv", newline="") as f:
        v1 = list(csv.DictReader(f))

    out, report = build_dataset(frames, rows, token_counter(a.count), random.Random(a.seed), docs=docs, v1_rows=v1)
    write_dataset(out, report, a.out)

    print(f"wrote {len(out)} rows to {a.out}")
    for split, c in report["counts"].items():
        print(f"  {split:<11} attacks {c['attack']:>5}   benign {c['benign']:>5}")
    print("visibility:", report["visibility"])
    print("overlaps:  ", report["overlaps"])
    print("unverified long attacks:", report["unverified_long_attacks"])
    by_band = {}
    for r in out:
        by_band[(r["split"], r["band"], r["label"])] = by_band.get((r["split"], r["band"], r["label"]), 0) + 1
    print("attacks per band (benign is equal):")
    for split in report["counts"]:
        print(f"  {split:<11}", {b: by_band.get((split, b, 1), 0) for b in BANDS})


if __name__ == "__main__":
    main()
