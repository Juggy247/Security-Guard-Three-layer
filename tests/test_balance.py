import random
from collections import Counter

from scripts.balance import band_of, balance


def rows(split, tokens, attacks, benign):
    return ([{"split": split, "label": 1, "tokens": tokens, "id": f"{split}-a{tokens}-{i}"} for i in range(attacks)]
            + [{"split": split, "label": 0, "tokens": tokens, "id": f"{split}-b{tokens}-{i}"} for i in range(benign)])


def counts(selected):
    return Counter((r["split"], r["band"], r["label"]) for r in selected)


# ---- band_of: [lo, hi) bands in tokens, [CLS] and [SEP] included ----

def test_band_of_edges():
    assert band_of(1) == "1-32"
    assert band_of(31) == "1-32"
    assert band_of(32) == "32-64"
    assert band_of(127) == "64-128"
    assert band_of(128) == "128-256"
    assert band_of(511) == "256-512"
    assert band_of(512) == "512+"
    assert band_of(5000) == "512+"


# ---- balance: equal attacks and benign inside every split and band ----

def test_balance_keeps_the_smaller_side_and_equalises():
    data = rows("train", 20, 10, 4) + rows("train", 100, 3, 8)
    selected, report = balance(data, cap=100, rng=random.Random(0))
    c = counts(selected)
    assert c[("train", "1-32", 1)] == c[("train", "1-32", 0)] == 4
    assert c[("train", "64-128", 1)] == c[("train", "64-128", 0)] == 3
    assert report[("train", "1-32")] == {"attack": 10, "benign": 4, "kept": 4}


def test_balance_respects_the_cap_per_label():
    selected, report = balance(rows("train", 20, 50, 50), cap=7, rng=random.Random(0))
    assert len(selected) == 14 and report[("train", "1-32")]["kept"] == 7


def test_balance_drops_a_band_that_has_only_one_label():
    data = rows("train", 20, 5, 5) + rows("train", 300, 6, 0)          # long band: attacks only, like v1
    selected, report = balance(data, cap=100, rng=random.Random(0))
    assert {r["band"] for r in selected} == {"1-32"}
    assert report[("train", "256-512")] == {"attack": 6, "benign": 0, "kept": 0}


def test_balance_never_mixes_splits():
    data = rows("train", 20, 6, 6) + rows("test", 20, 2, 9)
    selected, report = balance(data, cap=100, rng=random.Random(0))
    c = counts(selected)
    assert c[("train", "1-32", 1)] == 6 and c[("test", "1-32", 1)] == 2 and c[("test", "1-32", 0)] == 2


def test_balance_is_deterministic_and_leaves_the_input_untouched():
    data = rows("train", 20, 10, 10)
    before = [dict(r) for r in data]
    a, _ = balance(data, cap=5, rng=random.Random(3))
    b, _ = balance(data, cap=5, rng=random.Random(3))
    assert a == b
    assert data == before                              # no "band" key added to the caller's rows
    assert [r["id"] for r in a] != [r["id"] for r in balance(data, cap=5, rng=random.Random(4))[0]]