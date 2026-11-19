"""Do attack/clean pairs stay the same length, and payloads stay inside the window, under another tokenizer?

The dataset was built and checked with the DistilBERT tokenizer. Run this before training a model with a
different tokenizer (it downloads the tokenizers, so run it where the models are available):
    python -m scripts.tokenizer_audit data/v2/final distilbert-base-uncased roberta-base \
        microsoft/deberta-v3-base answerdotai/ModernBERT-base
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.length_audit import _auc

SPLITS = ("train", "validation", "test")


def payload_span(attack, clean):
    """Character span in `attack` of the words that differ from `clean`; words both share count as context."""
    found = list(re.finditer(r"\S+", attack))
    a, c = [m.group() for m in found], clean.split()
    head = 0
    while head < min(len(a), len(c)) and a[head] == c[head]:
        head += 1
    tail = 0
    while tail < min(len(a), len(c)) - head and a[-1 - tail] == c[-1 - tail]:
        tail += 1
    if head + tail >= len(a):
        raise ValueError("the attack and the clean twin do not differ in any word")
    return found[head].start(), found[len(a) - tail - 1].end()


def audit_pairs(rows, count_tokens, window):
    """Per split: how the token counts of attack and clean twins differ, and which payloads end inside `window`."""
    members = defaultdict(dict)
    for r in rows:
        if r["pair_id"]:
            members[r["pair_id"]][r["label"]] = r
    counts = defaultdict(lambda: {"attack": [], "clean": []})
    seen = defaultdict(lambda: {"pairs": 0, "hidden": 0, "dataset_hidden": 0, "disagree": 0})
    for pid, m in sorted(members.items()):
        if set(m) != {0, 1}:
            raise ValueError(f"pair {pid} needs one attack and one clean row")
        attack, clean = m[1], m[0]
        split = attack["split"]
        counts[split]["attack"].append(count_tokens(attack["prompt"]))
        counts[split]["clean"].append(count_tokens(clean["prompt"]))
        _, end = payload_span(attack["prompt"], clean["prompt"])
        visible = count_tokens(attack["prompt"][:end]) <= window
        cell = seen[split]
        cell["pairs"] += 1
        cell["hidden"] += not visible
        cell["dataset_hidden"] += attack["visible"] is False
        cell["disagree"] += attack["visible"] != visible
    length = {}
    for split, c in counts.items():
        diffs = [a - k for a, k in zip(c["attack"], c["clean"])]
        length[split] = {"mean_diff": round(sum(diffs) / len(diffs), 3),
                         "share_unequal": round(sum(d != 0 for d in diffs) / len(diffs), 3),
                         "auc": _auc(c["attack"], c["clean"])}
    return {"pairs": {s: v["pairs"] for s, v in seen.items()}, "length": length, "visibility": dict(seen)}


def load_pairs(folder):
    """The paired rows of a built dataset, with `visible` as True/False/None."""
    rows = []
    for split in SPLITS:
        with open(Path(folder) / f"dataset_v2_{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["pair_id"]:
                    rows.append({"pair_id": r["pair_id"], "split": split, "label": int(r["label"]), "prompt": r["prompt"],
                                 "visible": {"True": True, "False": False}.get(r["visible"])})
    return rows


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("results/tokenizer_audit.json"))
    a = ap.parse_args()
    rows, report = load_pairs(a.folder), {}
    for name in a.models:
        tok = AutoTokenizer.from_pretrained(name)
        window = a.max_length - tok.num_special_tokens_to_add(pair=False)
        count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
        report[name] = {"window": window, "cased": tok.tokenize("Ignore") != tok.tokenize("ignore"), **audit_pairs(rows, count, window)}
        r = report[name]
        print(f"\n{name}  (content window {window}, {'cased' if r['cased'] else 'uncased'})")
        for split in SPLITS:
            L, V = r["length"][split], r["visibility"][split]
            print(f"  {split:<11} pairs {V['pairs']:>4}  attack-clean tokens: mean {L['mean_diff']:>6}  unequal {L['share_unequal']:.2f}  length AUC {L['auc']}"
                  f"  | hidden {V['hidden']:>3} (dataset flag {V['dataset_hidden']:>3}, disagree {V['disagree']:>3})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
