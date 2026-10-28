import pytest

from three_layer_firewall.db.logger import RequestLogger, sha256

DETAILS = {
    "san_blocked": False, "san_rule_id": None, "san_flag": None, "san_ms": 0.2,
    "clf_attack_prob": 0.03, "clf_threshold": 0.7, "clf_ms": 12.5,
    "prompt_tokens": 9, "truncated": False,
}


def passed_scan():
    return {"blocked": False, "layer": "input_passed", "details": dict(DETAILS)}


def blocked_scan(layer):
    return {"blocked": True, "layer": layer, "details": dict(DETAILS)}


def make(store_text=False):
    return RequestLogger(None, store_text=store_text, firewall_version="test", config={"x": 1})


def test_sha256_known_value():
    assert sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("layer", ["sanitizer", "classifier"])
def test_input_block_sets_blocked_by(layer):
    row = make().build_row(endpoint="/scan", prompt="hi", scan_result=blocked_scan(layer))
    assert row.blocked is True and row.blocked_by == layer


def test_output_block_sets_blocked_by():
    out = {"blocked": True, "rule_id": "leak:x", "similarity": None, "ms": 1.5}
    row = make().build_row(endpoint="/chat", prompt="hi", scan_result=passed_scan(), output_result=out)
    assert row.blocked is True and row.blocked_by == "output_filter"
    assert row.out_blocked is True and row.out_rule_id == "leak:x" and row.out_ms == 1.5


def test_passed_request_has_no_blocker():
    row = make().build_row(endpoint="/scan", prompt="hi", scan_result=passed_scan())
    assert row.blocked is False and row.blocked_by is None
    assert row.out_blocked is None and row.out_rule_id is None   # /scan never runs layer 3


def test_text_not_stored_by_default():
    row = make().build_row(endpoint="/chat", prompt="two words", scan_result=passed_scan(),
                           llm_response="secret answer")
    assert row.prompt_text is None and row.llm_response is None
    assert row.prompt_sha256 == sha256("two words")      # the hash is always kept
    assert row.prompt_chars == 9 and row.prompt_words == 2


def test_text_stored_when_enabled():
    row = make(store_text=True).build_row(endpoint="/chat", prompt="two words",
                                          scan_result=passed_scan(), llm_response="answer")
    assert row.prompt_text == "two words" and row.llm_response == "answer"


def test_details_are_copied():
    row = make().build_row(endpoint="/scan", prompt="hi", scan_result=passed_scan())
    assert row.prompt_tokens == 9 and row.truncated is False
    assert row.clf_attack_prob == 0.03 and row.clf_threshold == 0.7
    assert row.san_ms == 0.2 and row.clf_ms == 12.5
    assert row.firewall_version == "test" and row.config == {"x": 1}


def test_system_prompt_hash_only_when_given():
    rl = make()
    without = rl.build_row(endpoint="/scan", prompt="hi", scan_result=passed_scan())
    with_sp = rl.build_row(endpoint="/chat", prompt="hi", scan_result=passed_scan(),
                           system_prompt="You are helpful.")
    assert without.system_prompt_sha256 is None
    assert with_sp.system_prompt_sha256 == sha256("You are helpful.")


def test_log_does_not_raise_when_db_is_down(caplog):
    def broken_factory():
        raise RuntimeError("db down")
    RequestLogger(broken_factory).log(endpoint="/scan", prompt="hi", scan_result=passed_scan())
    assert "Could not write request log row" in caplog.text


def test_log_does_not_raise_on_bad_input(caplog):
    RequestLogger(lambda: None).log(endpoint="/scan", prompt="hi", scan_result={})  # no "blocked" key
    assert "Could not write request log row" in caplog.text


def test_log_without_session_factory_is_a_noop():
    assert RequestLogger(None).log(endpoint="/scan", prompt="hi", scan_result={}) is None