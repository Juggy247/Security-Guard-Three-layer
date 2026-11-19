import csv
import json

from scripts.build_v2 import COLUMNS, write_dataset


def row(prompt, split, label=0, **kw):
    r = dict.fromkeys(COLUMNS)
    r.update(prompt=prompt, split=split, label=label, tokens=7, content_tokens=5, source="frames", **kw)
    return r


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_one_csv_per_split_with_the_expected_header_and_counts(tmp_path):
    rows = [row("a", "train"), row("b", "train"), row("c", "validation"), row("d", "test")]
    write_dataset(rows, {}, tmp_path)
    for split, n in (("train", 2), ("validation", 1), ("test", 1)):
        path = tmp_path / f"dataset_v2_{split}.csv"
        assert open(path).readline().strip().split(",") == COLUMNS and len(read(path)) == n


def test_missing_values_become_empty_cells_and_flags_are_written_as_text(tmp_path):
    write_dataset([row("a", "test", visible=False, position=0.5)], {}, tmp_path)
    got = read(tmp_path / "dataset_v2_test.csv")[0]
    assert got["pair_id"] == "" and got["visible"] == "False" and got["position"] == "0.5" and got["label"] == "0"


def test_prompts_with_newlines_commas_and_quotes_come_back_unchanged(tmp_path):
    text = 'Line one,\n\n"quoted", and a tab\there\n'
    write_dataset([row(text, "train")], {}, tmp_path)
    assert read(tmp_path / "dataset_v2_train.csv")[0]["prompt"] == text


def test_missing_folders_are_created_and_an_empty_split_still_gets_a_header(tmp_path):
    out = tmp_path / "a" / "b"
    write_dataset([row("a", "train")], {}, out)
    assert read(out / "dataset_v2_test.csv") == [] and open(out / "dataset_v2_test.csv").readline().strip().split(",") == COLUMNS


def test_the_report_is_saved_as_json(tmp_path):
    report = {"counts": {"train": {"attack": 2, "benign": 2}}, "overlaps": {"exact": 1, "near": 0, "dropped": {}}}
    write_dataset([], report, tmp_path)
    assert json.load(open(tmp_path / "report.json")) == report


def test_extra_keys_on_a_row_are_ignored(tmp_path):
    write_dataset([{**row("a", "train"), "extra": 1}], {}, tmp_path)
    assert "extra" not in read(tmp_path / "dataset_v2_train.csv")[0]
