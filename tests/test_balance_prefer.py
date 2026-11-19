import random

from scripts.balance import balance


def rows(n_attack, n_pref, n_other, tokens=10):
    out = [{"prompt": f"a{i}", "label": 1, "tokens": tokens, "split": "train", "pref": False} for i in range(n_attack)]
    out += [{"prompt": f"p{i}", "label": 0, "tokens": tokens, "split": "train", "pref": True} for i in range(n_pref)]
    out += [{"prompt": f"o{i}", "label": 0, "tokens": tokens, "split": "train", "pref": False} for i in range(n_other)]
    return out


def pick(data, **kw):
    chosen, report = balance(data, cap=1000, rng=random.Random(0), **kw)
    benign = [r for r in chosen if r["label"] == 0]
    return len([r for r in benign if r["pref"]]), len([r for r in benign if not r["pref"]]), report


def test_preferred_rows_are_all_kept_when_they_fit_in_their_share():
    assert pick(rows(40, 10, 90), prefer=lambda r: r["pref"], prefer_share=0.5)[:2] == (10, 30)


def test_preferred_rows_are_limited_to_their_share():
    assert pick(rows(40, 100, 100), prefer=lambda r: r["pref"], prefer_share=0.5)[:2] == (20, 20)


def test_preferred_rows_fill_the_gap_when_other_rows_run_out():
    assert pick(rows(40, 100, 5), prefer=lambda r: r["pref"], prefer_share=0.5)[:2] == (35, 5)


def test_without_prefer_nothing_changes_and_the_counts_stay_equal():
    chosen, report = balance(rows(40, 100, 100), cap=1000, rng=random.Random(0))
    assert sum(r["label"] == 1 for r in chosen) == sum(r["label"] == 0 for r in chosen) == 40
    assert report[("train", "1-32")] == {"attack": 40, "benign": 200, "kept": 40}


def test_attack_rows_are_not_affected_by_prefer():
    data = rows(40, 100, 100)
    for r in data[:40]:
        r["pref"] = True                                   # flagging attacks must not change how many are kept
    chosen, _ = balance(data, cap=1000, rng=random.Random(0), prefer=lambda r: r["pref"], prefer_share=0.5)
    assert sum(r["label"] == 1 for r in chosen) == 40 == sum(r["label"] == 0 for r in chosen)
