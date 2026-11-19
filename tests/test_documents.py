import csv

import pytest

from scripts.documents import document_problems, load_documents

FIELDS = ["id", "genre", "hard_negative", "form", "target_words", "provenance", "author", "brief", "text"]


def row(i="d01", **kw):
    base = {"id": i, "genre": "email", "hard_negative": "none", "form": "plain", "target_words": "10",
            "provenance": "llm", "author": "claude", "brief": "b", "text": "one two three four five six seven eight nine ten"}
    base.update(kw)
    return base


def write(path, rows, fields=FIELDS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_documents_returns_rows_with_numbers(tmp_path):
    docs, skipped = load_documents(write(tmp_path / "d.csv", [row("d01"), row("d02", text="a b c")]))
    assert [d["id"] for d in docs] == ["d01", "d02"] and skipped == []
    assert docs[0]["target_words"] == 10 and docs[0]["words"] == 10 and docs[1]["words"] == 3


def test_load_documents_keeps_multiline_text_exactly(tmp_path):
    text = "Line one\n\nLine two, with \"quotes\"\n"
    docs, _ = load_documents(write(tmp_path / "d.csv", [row(text=text)]))
    assert docs[0]["text"] == text


def test_load_documents_skips_rows_without_text_and_lists_them(tmp_path):
    docs, skipped = load_documents(write(tmp_path / "d.csv", [row("d01"), row("d02", text="  \n"), row("d03", text="")]))
    assert [d["id"] for d in docs] == ["d01"] and skipped == ["d02", "d03"]


def test_load_documents_rejects_a_file_with_the_wrong_columns(tmp_path):
    path = write(tmp_path / "d.csv", [{k: v for k, v in row().items() if k != "form"}], [f for f in FIELDS if f != "form"])
    with pytest.raises(ValueError, match="form"):
        load_documents(path)


def test_load_documents_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="d01.*duplicate"):
        load_documents(write(tmp_path / "d.csv", [row("d01"), row("d01")]))


def test_load_documents_rejects_unknown_provenance_and_hard_negative(tmp_path):
    with pytest.raises(ValueError, match="d01.*provenance"):
        load_documents(write(tmp_path / "d.csv", [row("d01", provenance="llm_generated")]))
    with pytest.raises(ValueError, match="d01.*hard_negative"):
        load_documents(write(tmp_path / "d.csv", [row("d01", hard_negative="")]))
    with pytest.raises(ValueError, match="d01.*target_words"):
        load_documents(write(tmp_path / "d.csv", [row("d01", target_words="many")]))


def test_load_documents_reports_every_bad_row_in_one_error(tmp_path):
    path = write(tmp_path / "d.csv", [row("d01", provenance="x"), row("d02"), row("d03", author="")])
    with pytest.raises(ValueError) as e:
        load_documents(path)
    assert "d01" in str(e.value) and "d03" in str(e.value) and "d02" not in str(e.value)


def doc(**kw):
    return {**row(), "target_words": 10, "words": 10, **kw}


def test_document_problems_is_empty_for_a_clean_document():
    assert document_problems(doc()) == []


def test_document_problems_flags_a_length_outside_the_range():
    assert any("length" in p for p in document_problems(doc(words=7)))
    assert any("length" in p for p in document_problems(doc(words=12)))
    assert document_problems(doc(words=9)) == [] and document_problems(doc(words=11)) == []


def test_document_problems_flags_unbalanced_bold_markers_only():
    assert any("**" in p for p in document_problems(doc(text="WARRANTY,** EVEN IF")))
    assert document_problems(doc(text="**Attendees:** Ana and **Absent:** Bo")) == []


def test_document_problems_flags_sentences_that_explain_the_test():
    for text in ("A few labels in this export may contain l/1 errors.",
                 "The account describes an incident rather than giving commands to anyone.",
                 "All staff mentioned in this account are fictional."):
        assert any("meta" in p for p in document_problems(doc(text=text))), text