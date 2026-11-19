"""Checks on attack/clean pairs with the token counter that built them."""
from collections import defaultdict


def check_pairs(rows, count_tokens, payloads, window):
    """Raise ValueError unless every pair holds under count_tokens: equal length, payload span, visibility.

    Returns {"pairs": number of pairs checked}."""
    members = defaultdict(dict)
    for r in rows:
        if r["pair_id"]:
            members[r["pair_id"]][r["label"]] = r
    longest_first = sorted(payloads, key=len, reverse=True)
    problems = []
    for pid, m in sorted(members.items()):
        if set(m) != {0, 1}:
            problems.append(f"{pid}: needs one attack and one clean row")
            continue
        attack, clean = m[1], m[0]
        counted = {"attack": count_tokens(attack["prompt"]), "clean": count_tokens(clean["prompt"])}
        if counted["attack"] != counted["clean"]:
            problems.append(f"{pid}: token count {counted['attack']} (attack) != {counted['clean']} (clean)")
        for name, row in (("attack", attack), ("clean", clean)):
            if row["content_tokens"] != counted[name]:
                problems.append(f"{pid}: content_tokens {row['content_tokens']} but {counted[name]} counted ({name})")
        payload = next((p for p in longest_first if p in attack["prompt"]), None)
        if payload is None:
            problems.append(f"{pid}: payload not found in the attack prompt")
            continue
        idx = attack["prompt"].index(payload)
        before = attack["prompt"][:idx]
        start, n = (count_tokens(before) if before.strip() else 0), count_tokens(payload)
        if start != attack["payload_start"]:
            problems.append(f"{pid}: payload_start {attack['payload_start']} but {start} counted")
        if n != attack["payload_tokens"]:
            problems.append(f"{pid}: payload_tokens {attack['payload_tokens']} but {n} counted")
        if count_tokens(attack["prompt"][:idx + len(payload)]) != start + n:
            problems.append(f"{pid}: the tokens up to the end of the payload do not add up")
        for row in (attack, clean):
            if row["visible"] != (start + n <= window):
                problems.append(f"{pid}: visible is {row['visible']} but the payload ends at token {start + n}")
    if problems:
        raise ValueError(f"{len(problems)} pair problems: " + "; ".join(problems[:5]))
    return {"pairs": len(members)}
