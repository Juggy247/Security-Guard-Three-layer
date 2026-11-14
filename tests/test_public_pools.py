import json
import random

from scripts.public_pools import BANDS, load_alpaca, load_dolly, to_prompt, is_tricky, by_band


def test_load_dolly_reads_jsonl_rows(tmp_path):
    p = tmp_path / "dolly.jsonl"
    p.write_text(
        json.dumps({"instruction": " Who wrote it? ", "context": "A text.", "response": "x", "category": "closed_qa"}) + "\n"
        + json.dumps({"instruction": "Name a colour.", "context": "", "response": "y", "category": "open_qa"}) + "\n",
        encoding="utf-8")
    rows = load_dolly(p)
    assert rows[0] == {"id": "dolly-0", "instruction": "Who wrote it?", "context": "A text.", "category": "closed_qa"}
    assert rows[1]["id"] == "dolly-1" and rows[1]["context"] == ""


def test_load_alpaca_maps_input_to_context(tmp_path):
    p = tmp_path / "alpaca.json"
    p.write_text(json.dumps([{"instruction": "Summarize.", "input": "Some passage.", "output": "z"},
                             {"instruction": "Say hi.", "input": "", "output": "hi"}]), encoding="utf-8")
    rows = load_alpaca(p)
    assert rows[0] == {"id": "alpaca-0", "instruction": "Summarize.", "context": "Some passage.", "category": "alpaca"}
    assert rows[1]["context"] == ""


def test_to_prompt_without_context_is_just_the_instruction():
    row = {"id": "x", "instruction": "Name a colour.", "context": "", "category": "open_qa"}
    assert to_prompt(row, random.Random(0)) == "Name a colour."


def test_to_prompt_with_context_uses_both_orders():
    row = {"id": "x", "instruction": "Who wrote it?", "context": "A long text.", "category": "closed_qa"}
    outs = {to_prompt(row, random.Random(seed)) for seed in range(30)}
    assert outs == {"Who wrote it?\n\nA long text.", "A long text.\n\nWho wrote it?"}


def test_is_tricky_finds_legitimate_uses_of_trigger_words():
    assert is_tricky("How do I ignore all previous notifications settings?")
    assert is_tricky("Please disregard my earlier draft.")
    assert is_tricky("What is a System Prompt?")
    assert is_tricky("Follow the cooking instructions on the box.")


def test_is_tricky_leaves_plain_text_alone():
    assert not is_tricky("What is the capital of France?")
    assert not is_tricky("Name two rivers in Poland.")


def test_by_band_sorts_by_word_count_and_drops_empty_text():
    texts = ["w " * n for n in (3, 20, 100, 200, 500)] + [""]
    pools = by_band(texts)
    assert list(pools) == BANDS
    assert [len(pools[b]) for b in BANDS] == [1, 1, 1, 1, 1]
    assert len(pools[BANDS[0]][0].split()) == 3