"""Length sweep for the input classifier.

Two questions, answered on the v1 classifier before we change anything:
  1. Length shortcut: how often is purely benign text of a given length blocked?
  2. Blind spot: how often is a known attack still caught when it sits at the start or at the
     end of benign padding of a given total length?

Run from the project root:
    python -m scripts.length_sweep run data/sweeps/length_sweep.csv
    python -m scripts.length_sweep summary data/sweeps/length_sweep.csv

Everything is rebuilt from fixed seeds, so the CSV stores no prompt text.
"""
import argparse
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "data" / "baselines" / "v1_decisions.csv"
LENGTHS = [16, 32, 64, 96, 128, 192, 256, 512]     # total tokens, [CLS] and [SEP] included
N_BENIGN = 100                                      # benign samples per length
GERMAN = set("ist ein eine und der die das wie nicht sie ich was wer wo warum sind wird werden "
             "mit auf den dem des zu von im zum zur bitte antworte".split())


def is_english(text: str) -> bool:
    """Rough filter: ASCII only and no common German word. The test split mixes in German prompts."""
    if not text.isascii():
        return False
    return not (set(re.findall(r"[a-z]+", text.lower())) & GERMAN)


def end_punct(text: str) -> str:
    text = text.strip()
    return text if text[-1] in ".!?" else text + "."


def load_pools():
    """Payloads: test-split attacks the sanitizer lets through and the classifier catches on its own.
    Padding: test-split safe prompts. Both single-line English, so they join cleanly."""
    df = pd.read_csv(DECISIONS)
    df = df[df["split"] == "test"]
    ok = ~df["prompt"].str.contains("\n") & df["prompt"].map(is_english)
    payloads = df[(df["label"] == 1) & ~df["san_blocked"].astype(bool)
                  & (df["clf_prob"] >= 0.99) & (df["words"] <= 25) & ok]
    padding = df[(df["label"] == 0) & ok]
    return payloads, padding


def build_padding(texts, counts, budget, rng):
    """Greedy fill with whole prompts in random order until the token budget is used up."""
    order = list(range(len(texts)))
    rng.shuffle(order)
    used, chosen = 0, []
    for i in order:
        if used + counts[i] <= budget:
            chosen.append(texts[i])
            used += counts[i]
    return " ".join(chosen), used


def run(out_path: Path) -> None:
    from three_layer_firewall.firewall import Firewall      # heavy import only when running

    fw = Firewall()

    def n_tok(text):
        return len(fw.tokenizer(text, add_special_tokens=False)["input_ids"])

    payloads, padding = load_pools()
    print(f"payloads: {len(payloads)}   padding prompts: {len(padding)}")
    pad_texts = [end_punct(p) for p in padding["prompt"]]
    pad_counts = [n_tok(t) for t in pad_texts]       # token counts add up: pieces are joined by spaces
    pay_texts = [end_punct(p) for p in payloads["prompt"]]
    pay_counts = [n_tok(t) for t in pay_texts]

    rows = []

    def scan(kind, target, payload_row, text, expected):
        r = fw.scan_input(text)
        d = r["details"]
        prob = d["clf_attack_prob"]
        rows.append({
            "kind": kind, "target_tokens": target, "payload_row": payload_row,
            "expected_tokens": expected, "tokens": d["prompt_tokens"], "truncated": d["truncated"],
            "san_blocked": d["san_blocked"], "clf_prob": prob,
            "clf_hit": prob is not None and prob > fw.ATTACK_THRESHOLD,
            "blocked": r["blocked"],
        })

    for target in LENGTHS:
        for i in range(N_BENIGN):
            rng = random.Random(f"benign|{target}|{i}")
            text, used = build_padding(pad_texts, pad_counts, target - 2, rng)
            scan("benign", target, -1, text, used + 2)
        for prow, ptext, pcount in zip(payloads["row"], pay_texts, pay_counts):
            budget = target - 2 - pcount
            if budget < 1:                       # payload does not fit with any padding
                continue
            rng = random.Random(f"attack|{target}|{prow}")
            pad, used = build_padding(pad_texts, pad_counts, budget, rng)
            if used == 0:
                continue
            scan("attack_start", target, prow, ptext + " " + pad, pcount + used + 2)
            scan("attack_end", target, prow, pad + " " + ptext, pcount + used + 2)
        print(f"  length {target} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(rows)} rows)")
    summary(out_path)


def summary(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"\ntoken-count mismatches (must be 0): {(df['tokens'] != df['expected_tokens']).sum()}")
    print(f"sanitizer blocks (these rows have no classifier score): {int(df['san_blocked'].sum())}")
    for col, title in (("clf_hit", "classifier says attack"), ("blocked", "firewall blocks")):
        print(f"\n{title} (rate per kind and length)")
        print(df.pivot_table(index="target_tokens", columns="kind", values=col, aggfunc="mean").round(3))
    print("\nrows per cell")
    print(df.pivot_table(index="target_tokens", columns="kind", values="blocked", aggfunc="size"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("csv", type=Path)
    a = ap.parse_args()
    run(a.out) if a.cmd == "run" else summary(a.csv)