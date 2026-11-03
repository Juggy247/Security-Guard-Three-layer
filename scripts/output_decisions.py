"""Record the output filter's per-response decisions on the output evaluation set, or compare two recordings.

Run from the project root:
    python -m scripts.output_decisions record data/baselines/output_baseline.csv
    python -m scripts.output_decisions compare data/baselines/output_baseline.csv data/baselines/now_output.csv

Results cover the 'dev' split only. Add --heldout to include the held-out split (do that once, at the end).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "output_eval"


def record(out_path: Path, heldout: bool) -> None:
    # heavy import only when recording, so 'compare' stays fast
    from three_layer_firewall.output_filter import OutputFilter

    prompts = json.loads((EVAL_DIR / "system_prompts.json").read_text(encoding="utf-8"))
    data = pd.read_csv(EVAL_DIR / "responses.csv", keep_default_na=False)
    if not heldout:
        data = data[data["split"] == "dev"]
    flt = OutputFilter(system_prompt=prompts["basic"])
    rows = []
    for _, r in data.iterrows():
        t0 = time.perf_counter()
        res = flt.check(r["response"], system_prompt=prompts[r["system_prompt_id"]])
        rows.append({
            "id": r["id"],
            "category": r["category"],
            "system_prompt_id": r["system_prompt_id"],
            "split": r["split"],
            "should_block": int(r["should_block"]),
            "words": len(r["response"].split()),
            "response": r["response"],
            "blocked": bool(res["blocked"]),
            "rule_id": res.get("rule_id"),
            "similarity": res.get("similarity"),
            "ms": (time.perf_counter() - t0) * 1000,
        })
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} decisions to {out_path}")
    summary(out, heldout)


def view(df: pd.DataFrame, heldout: bool) -> pd.DataFrame:
    return df if heldout else df[df["split"] == "dev"]


def summary(df: pd.DataFrame, heldout: bool = False) -> None:
    df = view(df, heldout)
    block, ok = df[df["should_block"] == 1], df[df["should_block"] == 0]
    tp, fp = int(block["blocked"].sum()), int(ok["blocked"].sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    print(f"\n[{'dev + heldout' if heldout else 'dev only'}] n={len(df)} "
          f"(should block={len(block)}, should pass={len(ok)})")
    print(f"  recall (should-block rows blocked): {block['blocked'].mean():.3f}")
    print(f"  false-positive rate:                {ok['blocked'].mean():.3f}")
    print(f"  precision:                          {precision:.3f}")

    print("\n[by category]  blocked / n   (for benign categories, lower is better)")
    for cat, g in df.groupby("category"):
        kind = "should block" if g["should_block"].iloc[0] == 1 else "should pass "
        print(f"  {cat:26} {int(g['blocked'].sum()):3d} / {len(g):3d}   {kind}")

    print("\n[rules that fired]  should-block rows | should-pass rows")
    for rule, g in df[df["blocked"]].groupby("rule_id"):
        print(f"  {rule:50} {int((g['should_block'] == 1).sum()):3d} | {int((g['should_block'] == 0).sum()):3d}")


def compare(old_path: Path, new_path: Path, heldout: bool, show: int = 25) -> None:
    old, new = pd.read_csv(old_path), pd.read_csv(new_path)
    if not heldout:
        old, new = old[old["split"] == "dev"], new[new["split"] == "dev"]
    m = old.merge(new, on="id", suffixes=("_old", "_new"))
    if not (len(m) == len(old) == len(new)):
        raise SystemExit("The two recordings do not cover the same responses.")
    newly_blocked = m[(~m["blocked_old"]) & (m["blocked_new"])]
    newly_allowed = m[(m["blocked_old"]) & (~m["blocked_new"])]
    for name, part in [("newly blocked", newly_blocked), ("newly allowed", newly_allowed)]:
        print(f"\n{len(part)} {name}")
        for _, r in part.head(show).iterrows():
            # blocking a should-block row, or allowing a should-pass row, is good
            good = (r["should_block_old"] == 1) == (name == "newly blocked")
            print(f"  [{'GOOD' if good else 'BAD '}] {r['category_old']:24} {r['response_old'][:70]!r}")
        if len(part) > show:
            print(f"  ... and {len(part) - show} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out", type=Path)
    rec.add_argument("--heldout", action="store_true")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("old", type=Path)
    cmp_.add_argument("new", type=Path)
    cmp_.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.out, args.heldout)
    else:
        compare(args.old, args.new, args.heldout)


if __name__ == "__main__":
    main()