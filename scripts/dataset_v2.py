"""Building blocks for dataset v2."""
import base64
import codecs
import itertools
import re

SIDES = ("train", "test")
LEET = str.maketrans("aeiostAEIOST", "431057431057")      # a->4 e->3 i->1 o->0 s->5 t->7
SLOT = re.compile(r"\{(\w+)\}")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def expand(family, side, n, rng, exclude=()):
    """n distinct texts built only from the frames and slot values of one side."""
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


def embed(payload, carrier, position, budget, count_tokens):
    """Put payload into carrier at a relative position (0 = start, 1 = end) within a token budget."""
    if not 0 <= position <= 1:
        raise ValueError("position must be between 0 and 1")
    room = budget - count_tokens(payload)
    if room < 0:
        raise ValueError("payload alone exceeds the budget")
    kept, used = [], 0
    for sentence in SENTENCE_END.split(carrier.strip()):
        cost = count_tokens(sentence)
        if used + cost > room:
            break
        kept.append(sentence)
        used += cost
    k = round(position * len(kept))
    before, after = kept[:k], kept[k:]
    text = " ".join(before + [payload] + after)
    return {
        "text": text,
        "tokens": count_tokens(text),
        "payload_start": count_tokens(" ".join(before)) if before else 0,
    }


def sample_by_length(texts, lo, hi, n, rng):
    pool = sorted({t for t in texts if lo <= len(t.split()) < hi})
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} texts with {lo}-{hi - 1} words, wanted {n}")
    return rng.sample(pool, n)



def mirror(family, phrases, swap_slot, side, n, rng, exclude=()):
    """Expand a benign family with one slot's values replaced by attack phrases.

    The attack then has exactly the same sentence shape as the benign prompts it mirrors."""
    if swap_slot not in family["slots"]:
        raise ValueError(f"family has no slot {swap_slot!r}")
    if side not in phrases:
        raise ValueError(f"phrases have no {side!r} side")
    swapped = {"slots": {**family["slots"], swap_slot: phrases}, "frames": family["frames"]}
    return expand(swapped, side, n, rng, exclude)