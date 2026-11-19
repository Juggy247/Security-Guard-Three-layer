import csv
import hashlib

import pytest

from scripts.build_v2 import DEFAULTS
from scripts.freeze_dataset import FILES, manifest


def make_folder(tmp_path):
    for split in ("train", "validation", "test"):
        with open(tmp_path / f"dataset_v2_{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt", "label"])
            w.writerow(["one line", 0])
            w.writerow(["two\nlines inside one prompt", 1])
    (tmp_path / "report.json").write_text('{"pair_check": {"pairs": 2}}')
    return tmp_path


def test_every_file_is_listed_with_its_sha256_and_size(tmp_path):
    folder = make_folder(tmp_path)
    m = manifest(folder, commit="abc123", dirty=False, seed=0)
    assert set(m["files"]) == set(FILES)
    for name, info in m["files"].items():
        data = (folder / name).read_bytes()
        assert info["sha256"] == hashlib.sha256(data).hexdigest() and info["bytes"] == len(data)


def test_csv_rows_are_counted_as_records_not_lines(tmp_path):
    m = manifest(make_folder(tmp_path), commit="abc123", dirty=False, seed=0)
    assert m["files"]["dataset_v2_train.csv"]["rows"] == 2
    assert "rows" not in m["files"]["report.json"]


def test_the_manifest_records_commit_dirty_flag_seed_tokenizer_and_config(tmp_path):
    m = manifest(make_folder(tmp_path), commit="abc123", dirty=True, seed=7)
    assert (m["commit"], m["dirty"], m["seed"]) == ("abc123", True, 7)
    assert m["tokenizer"] == "distilbert-base-uncased" and m["config"] == DEFAULTS


def test_a_missing_file_is_an_error(tmp_path):
    folder = make_folder(tmp_path)
    (folder / "dataset_v2_test.csv").unlink()
    with pytest.raises(FileNotFoundError):
        manifest(folder, commit="abc123", dirty=False, seed=0)


def test_a_changed_file_changes_its_hash(tmp_path):
    folder = make_folder(tmp_path)
    before = manifest(folder, "abc123", False, 0)["files"]["dataset_v2_train.csv"]["sha256"]
    (folder / "dataset_v2_train.csv").write_text("prompt,label\nchanged,1\n")
    assert manifest(folder, "abc123", False, 0)["files"]["dataset_v2_train.csv"]["sha256"] != before
