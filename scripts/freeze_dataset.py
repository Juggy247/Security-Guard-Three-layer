"""Write manifest.json next to a built dataset: what it was built from and a hash of every file.

Run from the project root after make_dataset_v2, with the seed that build used:
    python -m scripts.freeze_dataset data/v2/final --seed 0
"""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.build_v2 import DEFAULTS

FILES = ["dataset_v2_train.csv", "dataset_v2_validation.csv", "dataset_v2_test.csv", "report.json"]


def manifest(folder, commit, dirty, seed, tokenizer="distilbert-base-uncased"):
    """Describe the built dataset in `folder`: size, record count (CSV) and SHA-256 of every file."""
    files = {}
    for name in FILES:
        path = Path(folder) / name
        data = path.read_bytes()
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if name.endswith(".csv"):
            with open(path, newline="") as f:
                files[name]["rows"] = sum(1 for _ in csv.reader(f)) - 1
    return {"commit": commit, "dirty": dirty, "seed": seed, "tokenizer": tokenizer, "config": DEFAULTS, "files": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--seed", type=int, default=0, help="the --seed the dataset was built with")
    a = ap.parse_args()
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    m = manifest(a.folder, git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no")), a.seed)
    (a.folder / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    print(f"wrote {a.folder / 'manifest.json'}  commit {m['commit'][:10]}  dirty={m['dirty']}")
    for name, info in m["files"].items():
        print(f"  {name:<26}{info.get('rows', ''):>6}  {info['sha256'][:16]}")


if __name__ == "__main__":
    main()
