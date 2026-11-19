import pytest

from scripts.length_audit import length_audit


def row(label, tokens, split="train", band="32-64", pair_id=None):
    return {"label": label, "content_tokens": tokens, "split": split, "band": band, "pair_id": pair_id}


def test_lengths_that_separate_the_classes_give_an_auc_of_one_in_either_direction():
    longer = [row(1, 50), row(1, 51), row(0, 40), row(0, 41)]
    shorter = [row(1, 40), row(1, 41), row(0, 50), row(0, 51)]
    assert length_audit(longer)["train|32-64"]["auc"] == 1.0
    assert length_audit(shorter)["train|32-64"]["auc"] == 1.0


def test_identical_lengths_give_an_auc_of_one_half():
    rows = [row(1, 40), row(1, 45), row(0, 40), row(0, 45)]
    assert length_audit(rows)["train|32-64"]["auc"] == 0.5


def test_ties_count_half():
    rows = [row(1, 1), row(1, 2), row(0, 2), row(0, 3)]          # attack wins in 0.5 of 4 pairs: 0.125, folded to 0.875
    assert length_audit(rows)["train|32-64"]["auc"] == pytest.approx(0.875)


def test_every_split_and_band_is_reported_with_its_sample_counts():
    rows = [row(1, 50), row(0, 45), row(0, 46), row(1, 100, "test", "64-128"), row(0, 90, "test", "64-128")]
    out = length_audit(rows)
    assert set(out) == {"train|32-64", "test|64-128"}
    assert (out["train|32-64"]["attack"], out["train|32-64"]["benign"]) == (1, 2)
    assert (out["test|64-128"]["attack"], out["test|64-128"]["benign"]) == (1, 1)


def test_a_band_with_only_one_class_has_no_auc():
    assert length_audit([row(1, 50), row(1, 51)])["train|32-64"]["auc"] is None


def test_pairs_are_audited_separately_from_unpaired_rows():
    pairs = [row(1, 50, pair_id="a"), row(0, 50, pair_id="a"), row(1, 60, pair_id="b"), row(0, 60, pair_id="b")]
    single = [row(1, 55), row(1, 56), row(0, 40), row(0, 41)]
    cell = length_audit(pairs + single)["train|32-64"]
    assert cell["unpaired"] == {"attack": 2, "benign": 2, "auc": 1.0}
    assert (cell["attack"], cell["benign"]) == (4, 4) and cell["auc"] < 1.0
