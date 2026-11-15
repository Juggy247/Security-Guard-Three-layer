"""Checks that keep dataset v2 honest: stable split assignment and duplicate detection."""
import hashlib
import re
from collections import defaultdict

SHINGLE = 3          # words per shingle for the near-duplicate check


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