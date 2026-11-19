"""Authored benign documents for dataset v2: load, validate, and flag defects."""
import csv
import re

FIELDS = ["id", "genre", "hard_negative", "form", "target_words", "provenance", "author", "brief", "text"]
PROVENANCE = ("human", "human_llm_assisted", "llm_edited", "llm")
HARD_NEGATIVE = ("none", "trigger_word", "ai_topic", "role_language", "quoted_instruction")
META = re.compile(
    r"(\bin this (export|document|text|extract) (may|might)\b"
    r"|\brather than giving\b"
    r"|\b(mentioned|described) in this (account|document|text) are fictional\b)", re.I)


def load_documents(path):
    """Return (documents, ids skipped for empty text); raise one ValueError listing every bad row."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        rows = list(reader)
    errors, seen, docs, skipped = [], set(), [], []
    for r in rows:
        i, bad = r["id"], []
        if i in seen:
            bad.append("duplicate id")
        seen.add(i)
        if r["provenance"] not in PROVENANCE:
            bad.append(f"provenance {r['provenance']!r} not in {PROVENANCE}")
        if r["hard_negative"] not in HARD_NEGATIVE:
            bad.append(f"hard_negative {r['hard_negative']!r} not in {HARD_NEGATIVE}")
        if not r["author"].strip():
            bad.append("author is empty")
        try:
            target = int(r["target_words"])
            if target <= 0:
                raise ValueError
        except ValueError:
            bad.append(f"target_words {r['target_words']!r} is not a positive integer")
        if bad:
            errors.append(f"{i}: " + "; ".join(bad))
        elif not r["text"].strip():
            skipped.append(i)
        else:
            docs.append({**r, "target_words": target, "words": len(r["text"].split())})
    if errors:
        raise ValueError("\n".join(errors))
    return docs, skipped


def document_problems(doc, low=0.85, high=1.15):
    """Defects in one loaded document: length off target, unbalanced ** markers, sentences that explain the test."""
    problems = []
    ratio = doc["words"] / doc["target_words"]
    if not low <= ratio <= high:
        problems.append(f"length {doc['words']} words is {ratio:.2f} of target {doc['target_words']}")
    if doc["text"].count("**") % 2:
        problems.append("unbalanced ** marker")
    m = META.search(doc["text"])
    if m:
        problems.append(f"meta sentence: {m.group(0)!r}")
    return problems