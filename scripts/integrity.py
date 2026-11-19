"""Checks that keep dataset v2 honest: stable split assignment and duplicate detection."""
import hashlib
import re
from collections import defaultdict

SHINGLE = 3          # words per shingle for the near-duplicate check
ORDER = ("test", "validation", "train")          # earlier splits are protected from later ones

def assign_splits(ids, fractions, seed):
    """Give every id one split, decided only by (seed, id), so adding data never moves old ids."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    out = {}
    for id_ in ids:
        digest = hashlib.sha256(f"{seed}|{id_}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        edge = 0.0
        for name, frac in fractions.items():
            edge += frac
            if u < edge:
                out[id_] = name
                break
        else:
            out[id_] = name          # rounding at the very top of the range
    return out


def _words(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _shingles(words):
    if len(words) <= SHINGLE:
        return {" ".join(words)}
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def find_overlaps(texts_a, texts_b, threshold=0.7):
    """(i, j, kind) for each pair where a[i] and b[j] are the same text ("exact") or share most
    of their 3-word shingles ("near", Jaccard >= threshold)."""
    words_a = [_words(t) for t in texts_a]
    sh_a = [_shingles(w) for w in words_a]
    index = defaultdict(set)
    for i, shingles in enumerate(sh_a):
        for s in shingles:
            index[s].add(i)
    hits = []
    for j, text in enumerate(texts_b):
        words_b = _words(text)
        sh_b = _shingles(words_b)
        candidates = set()
        for s in sh_b:
            candidates |= index.get(s, set())
        for i in sorted(candidates):
            if words_a[i] == words_b:
                hits.append((i, j, "exact"))
            elif len(sh_a[i] & sh_b) / len(sh_a[i] | sh_b) >= threshold:
                hits.append((i, j, "near"))
    return hits

def drop_overlaps(rows, threshold=0.7, order=ORDER):
    """Remove rows that copy (exactly or nearly) a row of a more protected split. Returns (rows, report).

    Test is never touched; validation loses rows that overlap test; train loses rows that overlap either.
    A dropped row takes its pair partner with it. Duplicates inside one split are left alone."""
    unknown = {r["split"] for r in rows} - set(order)
    if unknown:
        raise ValueError(f"unknown split(s): {sorted(unknown)}")
    dropped, counts = set(), {"exact": 0, "near": 0}
    for n, protected in enumerate(order):
        for later in order[n + 1:]:
            ia = [i for i, r in enumerate(rows) if r["split"] == protected and i not in dropped]
            ib = [j for j, r in enumerate(rows) if r["split"] == later and j not in dropped]
            if not ia or not ib:
                continue
            kinds = {}
            for _, j, kind in find_overlaps([rows[i]["prompt"] for i in ia], [rows[j]["prompt"] for j in ib], threshold):
                kinds[ib[j]] = "exact" if "exact" in (kind, kinds.get(ib[j])) else "near"
            for kind in kinds.values():
                counts[kind] += 1
            hit_pairs = {rows[j]["pair_id"] for j in kinds if rows[j]["pair_id"] is not None}
            dropped |= set(kinds)
            dropped |= {j for j in ib if rows[j]["pair_id"] in hit_pairs}
    per_split = {}
    for i in sorted(dropped):
        per_split[rows[i]["split"]] = per_split.get(rows[i]["split"], 0) + 1
    return [r for i, r in enumerate(rows) if i not in dropped], {**counts, "dropped": per_split}