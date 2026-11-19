"""Benign pool rows and attack/clean pairs for dataset v2, split by the id of the source row."""
import random
import re
from collections import defaultdict

from scripts.public_pools import is_tricky, to_prompt
from scripts.dataset_v2 import SIDES, SPLIT_SIDE, _units, available, expand, long_pair


def _source(row_id):
    return row_id.split("-")[0]


def pool_rows(rows, splits, count_tokens, rng, exclude_ids=()):
    """One benign row per Dolly or Alpaca row, in the split its id was assigned to."""
    out = []
    for row in rows:
        if row["id"] in exclude_ids or not row["instruction"]:
            continue
        prompt = to_prompt(row, rng)
        out.append({
            "prompt": prompt, "label": 0, "attack_type": "safe_tricky" if is_tricky(prompt) else "safe",
            "source": _source(row["id"]), "split": splits[row["id"]], "frame_id": row["id"], "pair_id": None,
            "tokens": count_tokens(prompt), "position": None, "payload_start": None,
        })
    return out


SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.IGNORECASE)
MARKUP = re.compile(r"[\[\]<>{}#|\\*`@?!]|https?:|www\.")
INSTRUCTION_VERBS = {"please", "do", "don't", "let", "write", "list", "give", "tell", "explain", "describe", "summarize",
                     "summarise", "answer", "respond", "reply", "say", "note", "remember", "consider", "make", "use",
                     "ensure", "provide", "create", "generate", "identify", "translate", "rewrite", "repeat", "print",
                     "output", "show", "follow", "act", "pretend", "imagine", "assume"}


def _harmless(sentence):
    """A plain declarative sentence: one capitalised sentence with a single final period, nothing that reads like an
    instruction, a question or markup, and no word that the sanitizer treats as a trigger."""
    return (sentence[:1].isupper() and sentence.endswith(".") and sentence.count(".") == 1
            and not MARKUP.search(sentence) and not SECOND_PERSON.search(sentence) and not is_tricky(sentence)
            and sentence.split()[0].lower().strip(",") not in INSTRUCTION_VERBS)


def filler_bank(rows, splits, sizes, count_tokens, rng, per_size=50):
    """Harmless sentences for the clean twins: {split: {token count: [sentences]}} for the counts in sizes.

    Each sentence comes from a passage of the same split, so a filler never carries text across splits."""
    seen = defaultdict(set)
    for row in sorted(rows, key=lambda r: r["id"]):
        for text, _ in _units(row["context"]):
            if _harmless(text.strip()):
                seen[splits[row["id"]]].add(text.strip())
    bank = {}
    for split, sentences in seen.items():
        found = defaultdict(list)
        ordered = sorted(sentences)
        rng.shuffle(ordered)
        for text in ordered:
            n = count_tokens(text)
            if n in sizes and len(found[n]) < per_size:
                found[n].append(text)
                if all(len(found.get(k, ())) >= per_size for k in sizes):
                    break
        if found:
            bank[split] = dict(found)
    return bank


def pair_rows(rows, splits, payloads, budgets, positions, per_split, count_tokens, rng, min_fill=0.8, fillers=None):
    """An attack prompt (payload inside the passage) and its clean twin for each usable passage.

    With fillers ({split: {token count: [sentences]}}) the clean twin gets a filler of the payload's length in its
    place, so both have the same token count; only payloads that have such a filler in the split are used."""
    sizes = {p: count_tokens(p) for side in payloads.values() for p in side} if fillers is not None else {}
    candidates = sorted((r for r in rows if r["context"]), key=lambda r: r["id"])
    rng.shuffle(candidates)
    made = defaultdict(int)
    out = []
    for row in candidates:
        split = splits[row["id"]]
        if made[split] >= per_split:
            continue
        options, bank = payloads[SPLIT_SIDE[split]], None
        if fillers is not None:
            bank = fillers.get(split, {})
            options = [p for p in options if bank.get(sizes[p])]
            if not options:
                continue
        payload = rng.choice(options)
        position = rng.choice(positions)
        order = list(budgets)
        rng.shuffle(order)
        for budget in order:
            try:
                pair = long_pair(row, payload, position, budget, count_tokens, rng, bank)
            except ValueError:
                continue
            if pair["tokens"] < min_fill * budget:
                continue
            made[split] += 1
            pid = f"{row['id']}|{budget}|{position}"
            common = {"source": _source(row["id"]), "split": split, "frame_id": row["id"], "pair_id": pid}
            out.append({**common, "prompt": pair["attack"], "label": 1, "attack_type": "embedded",
                        "tokens": pair["tokens"], "position": position, "payload_start": pair["payload_start"],
                        "payload_tokens": pair["payload_tokens"]})
            out.append({**common, "prompt": pair["clean"], "label": 0, "attack_type": "safe",
                        "tokens": count_tokens(pair["clean"]), "position": None, "payload_start": None,
                        "payload_tokens": None})
            break
    return out

def document_pairs(docs, tasks, payloads, budgets, positions, count_tokens, rng, fillers=None):
    """Attack/clean pairs from the authored documents: every document at every budget and position.

    With fillers ({token count: [sentences]}, the test bank) each clean twin gets one of the payload's length."""
    rows = [{"id": f"doc-{d['id']}", "instruction": rng.choice(tasks), "context": d["text"]} for d in docs]
    meta = {f"doc-{d['id']}": {"doc_id": d["id"], **{k: d[k] for k in ("genre", "hard_negative", "form", "provenance")}}
            for d in docs}
    splits = {r["id"]: "test" for r in rows}
    bank = None if fillers is None else {"test": fillers}
    out = []
    for budget in budgets:
        for position in positions:
            for r in pair_rows(rows, splits, payloads, [budget], [position], len(rows), count_tokens, rng, fillers=bank):
                out.append({**r, **meta[r["frame_id"]]})
    return out

def payload_bank(frames, per_family, rng, families=None):
    """Attack sentences for the embedded pairs: up to per_family distinct texts per attack family, for each side."""
    names = families or list(frames["attack"])
    bank = {}
    for side in SIDES:
        texts = []
        for name in names:
            family = frames["attack"][name]
            k = min(per_family, available(family, side))
            if k:
                texts += [item["text"] for item in expand(family, side, k, rng)]
        bank[side] = texts
    return bank