"""How well length alone separates attacks from benign prompts, per split and band."""
from collections import defaultdict


def _auc(attack, benign):
    """Probability that a random attack is longer than a random benign prompt (ties count half), folded so that
    0.5 means no signal and 1.0 means length alone separates the classes. None when a class is missing."""
    if not attack or not benign:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in attack for b in benign)
    auc = wins / (len(attack) * len(benign))
    return round(max(auc, 1 - auc), 3)


def length_audit(rows):
    """Per 'split|band': attack and benign counts and the length-only AUC, for all rows and for unpaired rows alone.

    Pairs have equal lengths by construction, so they pull the overall AUC toward 0.5; the unpaired figure shows
    what is left for a model to exploit."""
    cells = defaultdict(lambda: {"all": {0: [], 1: []}, "unpaired": {0: [], 1: []}})
    for r in rows:
        cell = cells[f"{r['split']}|{r['band']}"]
        cell["all"][r["label"]].append(r["content_tokens"])
        if not r["pair_id"]:
            cell["unpaired"][r["label"]].append(r["content_tokens"])
    out = {}
    for key, cell in sorted(cells.items()):
        a, u = cell["all"], cell["unpaired"]
        out[key] = {"attack": len(a[1]), "benign": len(a[0]), "auc": _auc(a[1], a[0]),
                    "unpaired": {"attack": len(u[1]), "benign": len(u[0]), "auc": _auc(u[1], u[0])}}
    return out
