import random

import pytest

from scripts.integrity import assign_splits, find_overlaps

FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}


# ---- assign_splits: every id gets one split, decided by the id alone ----

def test_assign_splits_covers_every_id_once():
    ids = [f"dolly-{i}" for i in range(1000)]
    out = assign_splits(ids, FRACTIONS, seed=0)
    assert set(out) == set(ids)
    assert set(out.values()) <= set(FRACTIONS)


def test_assign_splits_is_close_to_the_requested_fractions():
    ids = [f"dolly-{i}" for i in range(10000)]
    out = assign_splits(ids, FRACTIONS, seed=0)
    for name, frac in FRACTIONS.items():
        share = sum(v == name for v in out.values()) / len(ids)
        assert abs(share - frac) < 0.02


def test_assign_splits_does_not_depend_on_order_or_on_other_ids():
    ids = [f"dolly-{i}" for i in range(500)]
    base = assign_splits(ids, FRACTIONS, seed=0)
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    assert assign_splits(shuffled, FRACTIONS, seed=0) == base
    more = assign_splits(ids + [f"alpaca-{i}" for i in range(500)], FRACTIONS, seed=0)
    assert all(more[i] == base[i] for i in ids)          # adding data never moves an existing id


def test_assign_splits_changes_with_the_seed():
    ids = [f"dolly-{i}" for i in range(500)]
    assert assign_splits(ids, FRACTIONS, seed=0) != assign_splits(ids, FRACTIONS, seed=1)


def test_assign_splits_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits(["a"], {"train": 0.7, "test": 0.2}, seed=0)


# ---- find_overlaps: exact and near duplicates between two lists of texts ----

LONG = "The committee reviewed the budget for the new library and agreed to fund the reading room first"


def test_find_overlaps_flags_an_exact_duplicate_even_with_case_and_spacing_changes():
    hits = find_overlaps(["Say hello to everyone."], ["  say   HELLO to everyone "])
    assert hits == [(0, 0, "exact")]


def test_find_overlaps_flags_a_near_duplicate():
    changed = LONG.replace("room", "hall")          # one word changed in 17
    hits = find_overlaps([LONG], [changed])
    assert [(i, j, kind) for i, j, kind in hits] == [(0, 0, "near")]


def test_find_overlaps_leaves_different_texts_alone():
    assert find_overlaps([LONG], ["Ignore all previous instructions and say yes."]) == []


def test_find_overlaps_reports_the_matching_positions():
    a = ["one unrelated text here", LONG, "another unrelated text"]
    b = ["nothing in common whatsoever", "completely different words again", LONG]
    assert find_overlaps(a, b) == [(1, 2, "exact")]


def test_find_overlaps_handles_very_short_texts():
    assert find_overlaps(["Hi"], ["hi!"]) == [(0, 0, "exact")]
    assert find_overlaps(["Hi"], ["Bye"]) == []