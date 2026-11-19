"""Benign pool rows and attack/clean pairs for dataset v2, split by the id of the source row."""
import random
from collections import defaultdict

from scripts.public_pools import is_tricky, to_prompt
from scripts.dataset_v2 import SIDES, SPLIT_SIDE, available, expand, long_pair


def _source(row_id):
    return row_id.split("-")[0]


def pool_rows(rows, splits, count_tokens, rng, exclude_ids=()):
    """One benign row per Dolly or Alpaca row, in the split its id was assigned to."""
    out = []
    for row in rows:
        if row["id"] in exclude_ids or not row["instruction"]:
            continue
        prompt = to_prompt(row, rng)
        out.append({
            "prompt": prompt, "label": 0, "attack_type": "safe_tricky" if is_tricky(prompt) else "safe",
            "source": _source(row["id"]), "split": splits[row["id"]], "frame_id": row["id"], "pair_id": None,
            "tokens": count_tokens(prompt), "position": None, "payload_start": None,
        })
    return out


def pair_rows(rows, splits, payloads, budgets, positions, per_split, count_tokens, rng, min_fill=0.8, fillers=None):
    """An attack prompt (payload inside the passage) and its clean twin for each usable passage.

    With fillers ({split: {token count: [sentences]}}) the clean twin gets a filler of the payload's length in its
    place, so both have the same token count; only payloads that have such a filler in the split are used."""
    sizes = {p: count_tokens(p) for side in payloads.values() for p in side} if fillers is not None else {}
    candidates = sorted((r for r in rows if r["context"]), key=lambda r: r["id"])
    rng.shuffle(candidates)
    made = defaultdict(int)
    out = []
    for row in candidates:
        split = splits[row["id"]]
        if made[split] >= per_split:
            continue
        options, bank = payloads[SPLIT_SIDE[split]], None
        if fillers is not None:
            bank = fillers.get(split, {})
            options = [p for p in options if bank.get(sizes[p])]
            if not options:
                continue
        payload = rng.choice(options)
        position = rng.choice(positions)
        order = list(budgets)
        rng.shuffle(order)
        for budget in order:
            try:
                pair = long_pair(row, payload, position, budget, count_tokens, rng, bank)
            except ValueError:
                continue
            if pair["tokens"] < min_fill * budget:
                continue
            made[split] += 1
            pid = f"{row['id']}|{budget}|{position}"
            common = {"source": _source(row["id"]), "split": split, "frame_id": row["id"], "pair_id": pid}
            out.append({**common, "prompt": pair["attack"], "label": 1, "attack_type": "embedded",
                        "tokens": pair["tokens"], "position": position, "payload_start": pair["payload_start"],
                        "payload_tokens": pair["payload_tokens"]})
            out.append({**common, "prompt": pair["clean"], "label": 0, "attack_type": "safe",
                        "tokens": count_tokens(pair["clean"]), "position": None, "payload_start": None,
                        "payload_tokens": None})
            break
    return out

def document_pairs(docs, tasks, payloads, budgets, positions, count_tokens, rng):
    """Attack/clean pairs from the authored documents: every document at every budget and position."""
    rows = [{"id": f"doc-{d['id']}", "instruction": rng.choice(tasks), "context": d["text"]} for d in docs]
    meta = {f"doc-{d['id']}": {"doc_id": d["id"], **{k: d[k] for k in ("genre", "hard_negative", "form", "provenance")}}
            for d in docs}
    splits = {r["id"]: "test" for r in rows}
    out = []
    for budget in budgets:
        for position in positions:
            for r in pair_rows(rows, splits, payloads, [budget], [position], len(rows), count_tokens, rng):
                out.append({**r, **meta[r["frame_id"]]})
    return out

def payload_bank(frames, per_family, rng, families=None):
    """Attack sentences for the embedded pairs: up to per_family distinct texts per attack family, for each side."""
    names = families or list(frames["attack"])
    bank = {}
    for side in SIDES:
        texts = []
        for name in names:
            family = frames["attack"][name]
            k = min(per_family, available(family, side))
            if k:
                texts += [item["text"] for item in expand(family, side, k, rng)]
        bank[side] = texts
    return bank