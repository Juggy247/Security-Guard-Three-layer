"""Which embedded payloads the classifier can actually see inside its token window."""
from collections import defaultdict


def mark_visibility(rows, window=126, keep_hidden_in=("test",)):
    """Add visible = True/False to the rows of every attack/clean pair (None for other rows).

    A payload is visible when it ends inside the first `window` content tokens. A pair whose payload is
    hidden looks identical to its clean twin for the model, so it is dropped from every split except
    those in keep_hidden_in, where it is kept as a measurement. Returns (rows, report)."""
    pairs = defaultdict(dict)
    for r in rows:
        if r["pair_id"] is not None:
            pairs[r["pair_id"]][r["label"]] = r
    visible = {}
    for pid, members in pairs.items():
        if set(members) != {0, 1}:
            raise ValueError(f"pair {pid} needs exactly one attack and one clean row")
        length = members[1]["tokens"] - members[0]["tokens"]
        if length <= 0:
            raise ValueError(f"pair {pid}: the attack is not longer than its clean twin")
        visible[pid] = members[1]["payload_start"] + length <= window
    out, report = [], {}
    for r in rows:
        pid = r["pair_id"]
        if pid is None:
            out.append({**r, "visible": None})
            continue
        ok, counts = visible[pid], report.setdefault(r["split"], {"pairs": 0, "hidden": 0, "dropped": 0})
        if r["label"] == 1:
            counts["pairs"] += 1
            counts["hidden"] += not ok
        if ok or r["split"] in keep_hidden_in:
            out.append({**r, "visible": ok})
        elif r["label"] == 1:
            counts["dropped"] += 1
    return out, report