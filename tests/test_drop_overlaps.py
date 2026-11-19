import pytest

from scripts.integrity import drop_overlaps

BASE = "the quick brown fox jumps over the lazy dog near the old river bank today"        # 15 words
NEAR = BASE.replace("today", "tomorrow")                                                    # one word changed
OTHER = "completely different sentence about cooking pasta with garlic and olive oil tonight"


def row(prompt, split, pair_id=None, label=0):
    return {"prompt": prompt, "split": split, "label": label, "pair_id": pair_id}


def kept(rows, **kw):
    out, report = drop_overlaps(rows, **kw)
    return [(r["split"], r["prompt"]) for r in out], report


def test_an_exact_copy_in_train_is_dropped_and_the_test_row_stays():
    out, report = kept([row(BASE, "test"), row(BASE, "train"), row(OTHER, "train")])
    assert out == [("test", BASE), ("train", OTHER)]
    assert report["exact"] == 1 and report["dropped"] == {"train": 1}


def test_a_near_copy_is_dropped_too_and_counted_as_near():
    out, report = kept([row(BASE, "test"), row(NEAR, "train")])
    assert out == [("test", BASE)] and report["near"] == 1 and report["exact"] == 0


def test_validation_is_kept_against_train_but_dropped_against_test():
    out, _ = kept([row(BASE, "validation"), row(BASE, "train")])
    assert out == [("validation", BASE)]
    out, _ = kept([row(BASE, "test"), row(NEAR, "validation"), row(OTHER, "validation")])
    assert out == [("test", BASE), ("validation", OTHER)]


def test_duplicates_inside_one_split_are_left_alone():
    out, report = kept([row(BASE, "train"), row(BASE, "train")])
    assert len(out) == 2 and report["dropped"] == {}


def test_a_row_dropped_for_overlapping_test_no_longer_protects_against_train():
    out, _ = kept([row(BASE, "test"), row(BASE, "validation"), row(BASE, "train")])
    assert out == [("test", BASE)]


def test_both_members_of_a_pair_go_together_and_other_pairs_stay():
    rows = [row(BASE, "test"),
            row(NEAR, "train", "p1", 1), row("clean twin of p1 " + OTHER, "train", "p1", 0),
            row(OTHER, "train", "p2", 1), row("clean twin of p2 " + OTHER + " again and again here", "train", "p2", 0)]
    out, report = kept(rows)
    assert [p for s, p in out if s == "train"] == [OTHER, "clean twin of p2 " + OTHER + " again and again here"]
    assert report["dropped"] == {"train": 2}


def test_order_is_kept_the_input_is_not_changed_and_the_threshold_is_adjustable():
    rows = [row(OTHER, "train"), row(BASE, "test"), row(NEAR, "train")]
    before = [dict(r) for r in rows]
    out, _ = drop_overlaps(rows)
    assert [r["prompt"] for r in out] == [OTHER, BASE] and rows == before
    out, _ = drop_overlaps(rows, threshold=0.95)
    assert len(out) == 3


def test_an_unknown_split_is_an_error():
    with pytest.raises(ValueError, match="dev"):
        drop_overlaps([row(BASE, "dev")])