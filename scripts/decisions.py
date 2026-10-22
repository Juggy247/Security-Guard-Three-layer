"""Record the firewall's per-prompt decisions on the project datasets, or compare two recordings.

Run from the project root:
    python -m scripts.decisions record data/baselines/v1_decisions.csv
    python -m scripts.decisions compare data/baselines/v1_decisions.csv data/baselines/now.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {
    "train": ROOT / "data" / "final" / "dataset_train.csv",
    "test": ROOT / "data" / "final" / "dataset_test.csv",
    "generalization": ROOT / "data" / "final" / "generalization_test.csv",
}
THRESHOLD = 0.7   # same as firewall.py
BATCH = 32


def record(out_path: Path) -> None:
    # heavy imports only when recording, so 'compare' stays fast
    import torch
    from three_layer_firewall.firewall import Firewall
    from three_layer_firewall.sanitizer import PromptSanitizer

    fw = Firewall()
    sanitizer = PromptSanitizer()
    rows = []
    for split, path in SPLITS.items():
        df = pd.read_csv(path)[["prompt", "label"]].dropna().reset_index(drop=True)
        probs = []
        for i in range(0, len(df), BATCH):
            batch = df["prompt"].iloc[i:i + BATCH].tolist()
            enc = fw.tokenizer(batch, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=128)
            with torch.no_grad():
                logits = fw.model(**enc).logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
        for i, (prompt, label) in enumerate(zip(df["prompt"], df["label"])):
            san = sanitizer.scan(prompt)
            rows.append({
                "split": split,
                "row": i,
                "label": int(label),
                "words": len(prompt.split()),
                "prompt": prompt,
                "san_blocked": bool(san["blocked"]),
                "san_reason": san["reason"],
                "clf_prob": probs[i],
                "firewall_blocked": bool(san["blocked"]) or probs[i] > THRESHOLD,
            })
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} decisions to {out_path}")
    summary(out)


def summary(df: pd.DataFrame) -> None:
    for split, g in df.groupby("split"):
        attacks, safe = g[g["label"] == 1], g[g["label"] == 0]
        print(f"\n[{split}] n={len(g)}  (attacks={len(attacks)}, safe={len(safe)})")
        print(f"  sanitizer blocks:   {int(g['san_blocked'].sum())}")
        if len(attacks):
            print(f"  recall (attacks blocked): {attacks['firewall_blocked'].mean():.3f}")
        if len(safe):
            print(f"  false-positive rate:      {safe['firewall_blocked'].mean():.3f}")

    print("\n[by length, all splits]  words | safe n (FPR) | attack n (recall)")
    buckets = pd.cut(df["words"], bins=[-1, 20, 60, 10**9], labels=["<=20", "21-60", ">60"])
    for name, g in df.groupby(buckets, observed=True):
        safe, attacks = g[g["label"] == 0], g[g["label"] == 1]
        fpr = f"{safe['firewall_blocked'].mean():.3f}" if len(safe) else "n/a"
        rec = f"{attacks['firewall_blocked'].mean():.3f}" if len(attacks) else "n/a"
        print(f"  {name:>6} | {len(safe):5d} ({fpr}) | {len(attacks):5d} ({rec})")


def compare(old_path: Path, new_path: Path, show: int = 15) -> None:
    old, new = pd.read_csv(old_path), pd.read_csv(new_path)
    m = old.merge(new, on=["split", "row"], suffixes=("_old", "_new"))
    if not (len(m) == len(old) == len(new)):
        raise SystemExit("The two recordings do not cover the same prompts.")
    for col in ["san_blocked", "firewall_blocked"]:
        newly_blocked = m[(~m[f"{col}_old"]) & (m[f"{col}_new"])]
        newly_allowed = m[(m[f"{col}_old"]) & (~m[f"{col}_new"])]
        print(f"\n{col}: {len(newly_blocked)} newly blocked, {len(newly_allowed)} newly allowed")
        for name, part in [("newly blocked", newly_blocked), ("newly allowed", newly_allowed)]:
            for _, r in part.head(show).iterrows():
                print(f"  [{name}] ({r['split']}, label={r['label_old']}) {r['prompt_old'][:90]!r}")
            if len(part) > show:
                print(f"  ... and {len(part) - show} more {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out", type=Path)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("old", type=Path)
    cmp_.add_argument("new", type=Path)
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.out)
    else:
        compare(args.old, args.new)


if __name__ == "__main__":
    main()