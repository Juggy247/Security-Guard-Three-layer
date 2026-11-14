"""Turn the downloaded Dolly and Alpaca files into benign prompt pools."""
import json
import re

BANDS = [(1, 15), (15, 60), (60, 150), (150, 400), (400, 10**9)]       # words per prompt
TRICKY = re.compile(
    r"\b(ignor\w*|disregard\w*|overrid\w*|bypass\w*|forget\w*|jailbreak\w*|system prompt|instructions?)\b",
    re.IGNORECASE,
)


def load_dolly(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            rows.append({"id": f"dolly-{i}", "instruction": r["instruction"].strip(),
                         "context": r["context"].strip(), "category": r["category"]})
    return rows


def load_alpaca(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [{"id": f"alpaca-{i}", "instruction": r["instruction"].strip(),
             "context": r["input"].strip(), "category": "alpaca"} for i, r in enumerate(data)]


def to_prompt(row, rng):
    """Instruction alone, or instruction and context in a random order (so the question is not always first)."""
    if not row["context"]:
        return row["instruction"]
    first, second = (row["instruction"], row["context"]) if rng.random() < 0.5 else (row["context"], row["instruction"])
    return f"{first}\n\n{second}"


def is_tricky(text):
    return bool(TRICKY.search(text))


def by_band(texts):
    pools = {band: [] for band in BANDS}
    for t in texts:
        n = len(t.split())
        for lo, hi in BANDS:
            if lo <= n < hi:
                pools[(lo, hi)].append(t)
                break
    return pools