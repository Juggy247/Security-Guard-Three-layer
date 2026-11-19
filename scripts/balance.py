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


def _pick(pool, n, rng, prefer, share):
    """n rows from pool; rows that `prefer` flags make up at most `share` of them unless the others run out."""
    if prefer is None:
        return rng.sample(pool, n)
    preferred = [r for r in pool if prefer(r)]
    others = [r for r in pool if not prefer(r)]
    k = max(min(len(preferred), int(share * n)), n - len(others))
    return rng.sample(preferred, k) + rng.sample(others, n - k)


def balance(rows, cap, rng, prefer=None, prefer_share=0.5):
    """Return (selected, report). Every (split, band) keeps n attacks and n benign rows,
    n = min(attacks, benign, cap). The report shows what was available and what was kept.
    prefer(row) marks rows (such as hard negatives) that should not be drowned out by plentiful ones."""
    groups = defaultdict(lambda: {1: [], 0: []})
    for r in rows:
        groups[(r["split"], band_of(r["tokens"]))][r["label"]].append(r)
    selected, report = [], {}
    for key in sorted(groups):
        attacks, benign = groups[key][1], groups[key][0]
        n = min(len(attacks), len(benign), cap)
        report[key] = {"attack": len(attacks), "benign": len(benign), "kept": n}
        for pool in (attacks, benign):
            for r in _pick(pool, n, rng, prefer, prefer_share):
                selected.append({**r, "band": key[1]})
    return selected, report