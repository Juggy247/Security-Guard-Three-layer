"""Building blocks for dataset v2."""
import base64
import codecs
import itertools
import re

SIDES = ("train", "test")
SPLIT_SIDE = {"train": "train", "validation": "train", "test": "test"}     # validation reuses the train frames
SCHEMES = ("base64", "rot13", "leet", "reversed", "spaced")
BENIGN_TYPE = {"compound": "safe", "tricky": "safe_tricky"}
LEET = str.maketrans("aeiostAEIOST", "431057431057")      # a->4 e->3 i->1 o->0 s->5 t->7
SLOT = re.compile(r"\{(\w+)\}")
BREAK = re.compile(r"((?<=[.!?])\s+|[ \t]*\n\s*)")      # after . ! ? and at every line break


def _candidates(family, side, exclude=()):
    """Every distinct text the family can build on one side, as {text: frame_id}."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    candidates = {}
    for frame in family["frames"]:
        if frame["split"] != side:
            continue
        names = sorted(set(SLOT.findall(frame["text"])))
        banks = [family["slots"][name][side] for name in names]
        for combo in itertools.product(*banks):
            values = dict(zip(names, combo))
            text = SLOT.sub(lambda m: values[m.group(1)], frame["text"])
            text = text[0].upper() + text[1:]
            if text not in exclude:
                candidates.setdefault(text, frame["id"])
    return candidates


def available(family, side, exclude=()):
    """How many distinct texts expand() can still return."""
    return len(_candidates(family, side, exclude))


def expand(family, side, n, rng, exclude=()):
    """n distinct texts built only from the frames and slot values of one side."""
    candidates = _candidates(family, side, exclude)
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} distinct {side} texts available, wanted {n}")
    chosen = rng.sample(sorted(candidates), n)
    return [{"text": t, "frame_id": candidates[t]} for t in chosen]


def encode(text, scheme):
    if scheme == "base64":
        return base64.b64encode(text.encode()).decode()
    if scheme == "rot13":
        return codecs.encode(text, "rot13")
    if scheme == "leet":
        return text.translate(LEET)
    if scheme == "reversed":
        return text[::-1]
    if scheme == "spaced":
        return "  ".join(" ".join(word) for word in text.split())
    raise ValueError(f"unknown scheme: {scheme}")


def _units(text):
    """(sentence, whitespace that followed it) pairs; the text is cut after . ! ? and at line breaks."""
    parts = BREAK.split(text.strip())
    return [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]


def embed(payload, carrier, position, budget, count_tokens, filler=None):
    """Put payload into carrier at a relative position (0 = start, 1 = end) within a token budget.

    With a filler (same token count as the payload) the twin carries the filler in the payload's place,
    so attack and twin have the same length and the same layout."""
    if not 0 <= position <= 1:
        raise ValueError("position must be between 0 and 1")
    n_payload = count_tokens(payload)
    if filler is not None and count_tokens(filler) != n_payload:
        raise ValueError("filler must have the same token count as the payload")
    room = budget - n_payload
    if room < 0:
        raise ValueError("payload alone exceeds the budget")
    kept, used = [], 0
    for unit in _units(carrier):
        cost = count_tokens(unit[0])
        if used + cost > room:
            break
        kept.append(unit)
        used += cost
    k = round(position * len(kept))
    before, after = kept[:k], kept[k:]
    join = lambda units: "".join(text + sep for text, sep in units)
    seps = [sep for _, sep in kept if sep]                   # at the very end of the carrier reuse the last separator seen
    boundary = (before[-1][1] if before else kept[0][1] if kept else "") or (seps[-1] if seps else " ")
    lead = join(before) + ("" if not before or before[-1][1] else boundary)
    build = lambda insert: lead + insert + (boundary + join(after).rstrip() if after else "")
    text = build(payload)
    return {
        "text": text,
        "tokens": count_tokens(text),
        "payload_start": count_tokens(lead.rstrip()) if before else 0,
        "payload_tokens": n_payload,
        "twin": build(filler) if filler is not None and kept else join(kept).rstrip(),     # same text without the payload
    }


def sample_by_length(texts, lo, hi, n, rng):
    pool = sorted({t for t in texts if lo <= len(t.split()) < hi})
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} texts with {lo}-{hi - 1} words, wanted {n}")
    return rng.sample(pool, n)


def _swap(family, phrases, swap_slot, side):
    """The family with one slot's values replaced by attack phrases."""
    if swap_slot not in family["slots"]:
        raise ValueError(f"family has no slot {swap_slot!r}")
    if side not in phrases:
        raise ValueError(f"phrases have no {side!r} side")
    return {"slots": {**family["slots"], swap_slot: phrases}, "frames": family["frames"]}


def mirror(family, phrases, swap_slot, side, n, rng, exclude=()):
    """Expand a benign family with one slot's values replaced by attack phrases.

    The attack then has exactly the same sentence shape as the benign prompts it mirrors."""
    return expand(_swap(family, phrases, swap_slot, side), side, n, rng, exclude)


def long_pair(row, payload, position, budget, count_tokens, rng, fillers=None):
    """An attack prompt and its clean twin from one Dolly row: the payload goes into the passage.

    With fillers ({token count: [sentences]}) the twin gets one that has the payload's token count in its place,
    so both prompts have the same length."""
    if not row["context"]:
        raise ValueError("row has no passage to put the payload in")
    n_payload = count_tokens(payload)
    filler = None
    if fillers is not None:
        if not fillers.get(n_payload):
            raise ValueError(f"no filler with {n_payload} tokens")
        filler = rng.choice(fillers[n_payload])
    room = budget - count_tokens(row["instruction"])
    r = embed(payload, row["context"], position, room, count_tokens, filler)
    if not r["twin"]:
        raise ValueError("no passage sentence fits in the budget")
    first = rng.random() < 0.5
    join = (lambda body: f"{row['instruction']}\n\n{body}") if first else (lambda body: f"{body}\n\n{row['instruction']}")
    attack, clean = join(r["text"]), join(r["twin"])
    idx = attack.index(payload)
    return {
        "attack": attack,
        "clean": clean,
        "tokens": count_tokens(attack),
        "payload_start": count_tokens(attack[:idx]) if attack[:idx].strip() else 0,
        "payload_tokens": n_payload,
        "instruction_first": first,
    }


def _row(prompt, label, attack_type, source, split, frame_id):
    return {"prompt": prompt, "label": label, "attack_type": attack_type, "source": source,
            "split": split, "frame_id": frame_id, "pair_id": None}


def _take(family, side, n, rng, exclude):
    k = min(n, available(family, side, exclude))
    return expand(family, side, k, rng, exclude) if k else []


def frame_rows(frames, split, per_family, rng, exclude=(), standalone_benign=("compound", "tricky")):
    """Labelled rows for one split from the frames file. per_family is a maximum per family.

    Validation uses the train frames, so pass the train prompts as exclude to keep them apart."""
    if split not in SPLIT_SIDE:
        raise ValueError(f"split must be one of {tuple(SPLIT_SIDE)}")
    side = SPLIT_SIDE[split]
    wrappers = frames.get("encoded_wrappers", {}).get(side)
    if not wrappers:
        raise ValueError(f"frames need encoded_wrappers for the {side} side")
    used = set(exclude)
    rows = []

    def add(items, label, attack_type, source):
        for item in items:
            used.add(item["text"])
            rows.append(_row(item["text"], label, attack_type, source, split, item["frame_id"]))

    for name, family in frames["attack"].items():
        add(_take(family, side, per_family, rng, used), 1, name, "frames")
    for name in standalone_benign:
        add(_take(frames["benign"][name], side, per_family, rng, used), 0, BENIGN_TYPE.get(name, "safe"), "frames")
    for slot in ("task2", "task1"):           # attack phrase in the second, then in the first task
        family = _swap(frames["benign"]["compound"], frames["mirror_phrases"], slot, side)
        add(_take(family, side, per_family // 2, rng, used), 1, "mirror", "mirror")
    for i, item in enumerate(_take(frames["attack"]["override"], side, per_family, rng, ())):
        scheme = SCHEMES[i % len(SCHEMES)]
        prompt = rng.choice(wrappers).replace("{text}", encode(item["text"], scheme))
        if prompt not in used:
            used.add(prompt)
            rows.append(_row(prompt, 1, "encoded", "encoded", split, f"{item['frame_id']}+{scheme}"))
    return rows