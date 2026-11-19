"""Assemble dataset v2 from the frames, the public pools, the authored documents and the v1 rows."""
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.balance import balance, band_of
from scripts.dataset_v2 import frame_rows
from scripts.integrity import assign_splits, drop_overlaps
from scripts.pair_check import check_pairs
from scripts.sources import document_pairs, filler_bank, pair_rows, payload_bank, pool_rows
from scripts.visibility import mark_visibility

SPECIAL_TOKENS = 2          # [CLS] and [SEP]
COLUMNS = ["prompt", "label", "attack_type", "source", "split", "frame_id", "pair_id", "band", "tokens", "content_tokens",
           "position", "payload_start", "payload_tokens", "visible", "doc_id", "genre", "hard_negative", "form", "provenance"]
TASKS = ["Summarize the following text.", "List the main points of this text.", "What is the tone of this text?",
         "Rewrite this text in simpler words.", "Who is this text written for?", "Give this text a short title."]
DEFAULTS = {
    "per_family": {"train": 60, "validation": 15, "test": 30},       # frames: at most this many texts per family
    "payloads_per_family": 20,
    "pool_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
    "v1_fractions": {"train": 0.85, "validation": 0.15},
    "split_seed": 42,
    "budgets": [64, 128, 256, 512],                                   # content tokens of the longest pair prompt
    "positions": [0.0, 0.25, 0.5, 0.75, 1.0],
    "pair_cells": {"train": 40, "validation": 6, "test": 8},         # pairs per (budget, position) cell and split
    "window": 126,                                                     # content tokens the classifier can see
    "overlap_threshold": 0.7,
    "balance_cap": 400,                                                # per class, split and band
    "hard_negative_share": 0.5,                                        # at most this share of kept benign rows are hard negatives
    "filler_per_size": 50,                                             # harmless fillers kept per split and token count
}


def _pairs_from_passages(rows, splits, payloads, cfg, count_tokens, rng, fillers):
    """Attack/clean pairs from passages, each passage used once; returns (pair rows, ids used)."""
    used, out = set(), []
    for budget in cfg["budgets"]:
        for position in cfg["positions"]:
            for split, cap in cfg["pair_cells"].items():
                free = [r for r in rows if r["id"] not in used and splits[r["id"]] == split]
                new = pair_rows(free, splits, payloads, [budget], [position], cap, count_tokens, rng, fillers=fillers)
                used |= {r["frame_id"] for r in new}
                out += new
    return out, used


def build_dataset(frames, rows, count_tokens, rng, docs=None, v1_rows=None, tasks=None, config=None):
    """Return (dataset rows, report). `rows` are Dolly/Alpaca rows; count_tokens must count content tokens additively."""
    cfg = {**DEFAULTS, **(config or {})}
    per = cfg["per_family"]
    train = frame_rows(frames, "train", per["train"], rng)
    validation = frame_rows(frames, "validation", per["validation"], rng, exclude={r["prompt"] for r in train})
    plain = train + validation + frame_rows(frames, "test", per["test"], rng)
    if v1_rows:
        side = assign_splits([r["prompt"] for r in v1_rows], cfg["v1_fractions"], cfg["split_seed"])
        plain += [{"prompt": r["prompt"], "label": int(r["label"]), "attack_type": r["attack_type"], "source": "v1",
                   "split": side[r["prompt"]], "frame_id": "v1", "pair_id": None} for r in v1_rows]

    splits = assign_splits([r["id"] for r in rows], cfg["pool_fractions"], cfg["split_seed"])
    payloads = payload_bank(frames, cfg["payloads_per_family"], rng)
    sizes = {count_tokens(p) for side in payloads.values() for p in side}
    fillers = filler_bank(rows, splits, sizes, count_tokens, rng, cfg["filler_per_size"])
    if docs and not fillers.get("test"):
        raise ValueError("no harmless filler sentence for the test split")
    pairs, used = _pairs_from_passages(rows, splits, payloads, cfg, count_tokens, rng, fillers)
    if docs:
        pairs += document_pairs(docs, tasks or TASKS, payloads, cfg["budgets"], cfg["positions"], count_tokens, rng,
                                fillers=fillers["test"])
    pairs, visibility = mark_visibility(pairs, cfg["window"])
    plain += pool_rows(rows, splits, count_tokens, rng, exclude_ids=used)

    for r in plain:                                    # content tokens first, then add [CLS] and [SEP]
        r["content_tokens"] = r["tokens"] if "tokens" in r else count_tokens(r["prompt"])
    for r in pairs:
        r["content_tokens"] = r["tokens"]
    everything = [{**dict.fromkeys(COLUMNS), **r, "tokens": r["content_tokens"] + SPECIAL_TOKENS} for r in plain + pairs]
    kept, overlaps = drop_overlaps(everything, cfg["overlap_threshold"])

    singles = [r for r in kept if r["pair_id"] is None]
    paired = [r for r in kept if r["pair_id"] is not None]
    hard_negative = lambda r: r["label"] == 0 and (r["source"] in ("frames", "v1") or r["attack_type"] == "safe_tricky")
    chosen, balance_report = balance(singles, cfg["balance_cap"], rng, prefer=hard_negative, prefer_share=cfg["hard_negative_share"])
    band = {r["pair_id"]: band_of(r["tokens"]) for r in paired if r["label"] == 1}
    out = chosen + [{**r, "band": band[r["pair_id"]]} for r in paired]
    rng.shuffle(out)

    counts = {s: {"attack": sum(r["label"] == 1 for r in out if r["split"] == s),
                  "benign": sum(r["label"] == 0 for r in out if r["split"] == s)} for s in sorted({r["split"] for r in out})}
    report = {
        "balance": {f"{s}|{b}": v for (s, b), v in balance_report.items()},
        "visibility": visibility,
        "overlaps": overlaps,
        "counts": counts,
        "sources": {f"{s}|{src}|{lab}": n for (s, src, lab), n in sorted(Counter((r["split"], r["source"], r["label"]) for r in out).items())},
        "pair_check": check_pairs(out, count_tokens, [p for side in payloads.values() for p in side], cfg["window"]),
        "unverified_long_attacks": sum(r["label"] == 1 and r["pair_id"] is None and r["content_tokens"] > cfg["window"] for r in out),
    }
    return out, report


def write_dataset(rows, report, out_dir):
    """Write dataset_v2_{train,validation,test}.csv and report.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        with open(out / f"dataset_v2_{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                if r["split"] == split:
                    writer.writerow({k: "" if r.get(k) is None else r[k] for k in COLUMNS})
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
