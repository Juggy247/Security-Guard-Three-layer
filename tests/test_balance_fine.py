import random
from collections import Counter

from scripts.balance import balance


def rows(split, tokens, attacks, benign, hard=False):
    return ([{"split": split, "label": 1, "tokens": tokens, "id": f"{split}-a{tokens}-{i}"} for i in range(attacks)]
            + [{"split": split, "label": 0, "tokens": tokens, "id": f"{split}-b{tokens}-{i}", "hard": hard} for i in range(benign)])


def by_length(selected):
    return Counter((r["tokens"], r["label"]) for r in selected)


def test_with_a_width_attacks_and_benign_are_matched_length_by_length():
    data = rows("train", 20, 3, 10) + rows("train", 25, 3, 1) + rows("train", 30, 0, 10)
    selected, report = balance(data, cap=100, rng=random.Random(0), width=1)
    assert by_length(selected) == {(20, 1): 3, (20, 0): 3, (25, 1): 1, (25, 0): 1}
    assert report[("train", "1-32")] == {"attack": 6, "benign": 21, "kept": 4}


def test_without_a_width_only_the_band_totals_are_matched():
    data = rows("train", 20, 3, 10) + rows("train", 25, 3, 1) + rows("train", 30, 0, 10)
    selected, _ = balance(data, cap=100, rng=random.Random(0))
    c = by_length(selected)
    assert sum(v for (t, l), v in c.items() if l == 1) == sum(v for (t, l), v in c.items() if l == 0) == 6
    assert c[(20, 1)] != c[(20, 0)] or c[(25, 1)] != c[(25, 0)]


def test_lengths_inside_one_bin_are_matched_together():
    data = rows("train", 20, 1, 0) + rows("train", 21, 1, 0) + rows("train", 23, 0, 1) + rows("train", 24, 0, 1)
    selected, _ = balance(data, cap=100, rng=random.Random(0), width=5)          # bin 20-24
    assert Counter(r["label"] for r in selected) == {1: 2, 0: 2}
    assert balance(data, cap=100, rng=random.Random(0), width=1)[0] == []


def test_a_bin_never_reaches_across_a_band_edge():
    data = rows("train", 31, 2, 0) + rows("train", 40, 0, 2)                      # one bin of 64, but two bands
    assert balance(data, cap=100, rng=random.Random(0), width=64)[0] == []


def test_the_cap_applies_to_the_band_and_keeps_every_length_matched():
    data = rows("train", 20, 50, 50) + rows("train", 21, 50, 50) + rows("train", 22, 50, 50)
    selected, report = balance(data, cap=30, rng=random.Random(0), width=1)
    c = by_length(selected)
    assert report[("train", "1-32")]["kept"] == 30 and len(selected) == 60
    assert all(c[(t, 1)] == c[(t, 0)] for t in (20, 21, 22))


def test_preferred_benign_rows_are_still_favoured_inside_a_length():
    data = rows("train", 20, 4, 0) + rows("train", 20, 0, 4, hard=True) + rows("train", 20, 0, 40)
    selected, _ = balance(data, cap=100, rng=random.Random(0), width=1, prefer=lambda r: r.get("hard"), prefer_share=1.0)
    assert [r["hard"] for r in selected if r["label"] == 0] == [True] * 4


def test_the_width_makes_no_difference_to_who_is_chosen_when_it_is_left_out():
    data = rows("train", 20, 5, 9) + rows("validation", 70, 4, 4)
    assert balance(data, cap=3, rng=random.Random(2)) == balance(data, cap=3, rng=random.Random(2), width=None)


def test_balance_with_a_width_is_deterministic():
    data = rows("train", 20, 30, 30) + rows("train", 21, 30, 30)
    assert balance(data, 25, random.Random(1), width=1) == balance(data, 25, random.Random(1), width=1)
    assert balance(data, 25, random.Random(1), width=1) != balance(data, 25, random.Random(2), width=1)
