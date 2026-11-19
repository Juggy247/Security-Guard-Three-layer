import pytest

from scripts.visibility import mark_visibility


def pair(pid, split="train", start=50, payload=10, total=200):
    common = {"split": split, "pair_id": pid, "frame_id": "doc-" + pid, "source": "dolly"}
    return [{**common, "prompt": "attack " + pid, "label": 1, "tokens": total, "payload_start": start},
            {**common, "prompt": "clean " + pid, "label": 0, "tokens": total - payload, "payload_start": None}]


def single(label=0, split="train"):
    return {"prompt": "x", "label": label, "tokens": 20, "payload_start": None, "pair_id": None, "split": split}


def flags(rows, **kw):
    out, _ = mark_visibility(rows, **kw)
    return {r["pair_id"]: r["visible"] for r in out if r["label"] == 1 and r["pair_id"]}


def test_a_payload_that_ends_inside_the_window_is_visible():
    assert flags(pair("a", start=80, payload=10)) == {"a": True}


def test_a_payload_that_ends_beyond_the_window_is_hidden_even_if_it_starts_inside():
    assert flags(pair("a", split="test", start=120, payload=10)) == {"a": False}


def test_the_window_boundary_is_inclusive():
    assert flags(pair("a", start=116, payload=10)) == {"a": True}                    # ends at 126
    assert flags(pair("b", split="test", start=117, payload=10)) == {"b": False}     # ends at 127


def test_the_window_can_be_changed():
    assert flags(pair("a", start=200, payload=10, total=400), window=300) == {"a": True}


def test_both_members_of_a_pair_get_the_same_flag():
    out, _ = mark_visibility(pair("a", split="test", start=300, payload=10, total=400))
    assert [r["visible"] for r in out] == [False, False]


def test_hidden_pairs_are_dropped_from_train_and_validation_but_kept_in_test():
    rows = pair("t", "train", 300, total=400) + pair("v", "validation", 300, total=400) + pair("x", "test", 300, total=400)
    out, _ = mark_visibility(rows)
    assert {r["pair_id"] for r in out} == {"x"} and [r["visible"] for r in out] == [False, False]


def test_visible_pairs_are_kept_in_every_split():
    rows = pair("t", "train") + pair("v", "validation") + pair("x", "test")
    out, _ = mark_visibility(rows)
    assert len(out) == 6 and all(r["visible"] is True for r in out)


def test_rows_outside_pairs_pass_through_with_no_flag_and_inputs_are_not_changed():
    rows = [single(0), single(1, "test")] + pair("a")
    before = [dict(r) for r in rows]
    out, _ = mark_visibility(rows)
    assert [r["visible"] for r in out[:2]] == [None, None] and rows == before


def test_report_counts_pairs_hidden_and_dropped_per_split():
    rows = pair("a", "train") + pair("b", "train", 300, total=400) + pair("c", "test", 300, total=400)
    _, report = mark_visibility(rows)
    assert report == {"train": {"pairs": 2, "hidden": 1, "dropped": 1}, "test": {"pairs": 1, "hidden": 1, "dropped": 0}}


def test_an_incomplete_pair_or_a_payload_of_no_length_is_an_error():
    with pytest.raises(ValueError, match="a"):
        mark_visibility(pair("a")[:1])
    with pytest.raises(ValueError, match="a"):
        mark_visibility(pair("a", payload=0))