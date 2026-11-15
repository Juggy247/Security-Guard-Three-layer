"""Keep attacks and benign prompts equal in number inside every split and length band."""
from collections import defaultdict

EDGES = [1, 32, 64, 128, 256, 512]          # token counts, [CLS] and [SEP] included


def band_of(tokens):
    for lo, hi in zip(EDGES, EDGES[1:]):
        if lo <= tokens < hi:
            return f"{lo}-{hi}"
    if tokens >= EDGES[-1]:
        return f"{EDGES[-1]}+"
    raise ValueError(f"token count {tokens} is below the first band")


def balance(rows, cap, rng):
    """Return (selected, report). Every (split, band) keeps n attacks and n benign rows,
    n = min(attacks, benign, cap). The report shows what was available and what was kept."""
    groups = defaultdict(lambda: {1: [], 0: []})
    for r in rows:
        groups[(r["split"], band_of(r["tokens"]))][r["label"]].append(r)
    selected, report = [], {}
    for key in sorted(groups):
        attacks, benign = groups[key][1], groups[key][0]
        n = min(len(attacks), len(benign), cap)
        report[key] = {"attack": len(attacks), "benign": len(benign), "kept": n}
        for pool in (attacks, benign):
            for r in rng.sample(pool, n):
                selected.append({**r, "band": key[1]})
    return selected, report