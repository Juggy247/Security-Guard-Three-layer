import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import api.main as m
from three_layer_firewall.db.logger import RequestLogger
from three_layer_firewall.db.models import RequestLog

DETAILS = {
    "san_blocked": False, "san_rule_id": None, "san_flag": None, "san_ms": 0.2,
    "clf_attack_prob": 0.03, "clf_threshold": 0.7, "clf_ms": 12.5,
    "prompt_tokens": 9, "truncated": False,
}


def scan(blocked, layer):
    return {"blocked": blocked, "layer": layer, "reason": "test reason",
            "confidence": 0.9, "details": DETAILS}


class FakeFirewall:
    def __init__(self, scan_result, output_result=None):
        self.scan_result = scan_result
        self.output_result = output_result or {
            "blocked": False, "rule_id": None, "similarity": 0.1, "ms": 1.0}

    def scan_input(self, prompt):
        return self.scan_result

    def scan_output(self, response):
        return self.output_result


@pytest.fixture
def client(monkeypatch, sf):
    monkeypatch.setattr(m, "request_logger",
                        RequestLogger(sf, store_text=True, firewall_version="t"))
    return TestClient(m.app)


def rows(sf):
    with sf() as s:
        return s.scalars(select(RequestLog).order_by(RequestLog.id)).all()


def fake_llm(monkeypatch, text):
    monkeypatch.setattr(m.ollama, "chat", lambda **kw: {"message": {"content": text}})


def test_scan_logs_a_row(client, sf, monkeypatch):
    monkeypatch.setattr(m, "firewall", FakeFirewall(scan(True, "sanitizer")))
    r = client.post("/scan", json={"prompt": "attack"})
    assert r.status_code == 200 and r.json()["blocked"] is True
    (row,) = rows(sf)
    assert row.endpoint == "/scan" and row.blocked_by == "sanitizer"
    assert row.total_ms is not None


def test_chat_input_block_never_calls_the_llm(client, sf, monkeypatch):
    monkeypatch.setattr(m, "firewall", FakeFirewall(scan(True, "classifier")))

    def must_not_run(**kw):
        raise AssertionError("the LLM must not be called")
    monkeypatch.setattr(m.ollama, "chat", must_not_run)

    r = client.post("/chat", json={"prompt": "attack"})
    assert r.json()["blocked"] is True and r.json()["response"] is None
    (row,) = rows(sf)
    assert row.endpoint == "/chat" and row.blocked_by == "classifier" and row.llm_ms is None


def test_chat_success_logs_llm_and_output_fields(client, sf, monkeypatch):
    monkeypatch.setattr(m, "firewall", FakeFirewall(scan(False, "input_passed")))
    fake_llm(monkeypatch, "Warsaw.")
    r = client.post("/chat", json={"prompt": "capital of Poland?"})
    assert r.json()["blocked"] is False and r.json()["response"] == "Warsaw."
    (row,) = rows(sf)
    assert row.status == "ok" and row.blocked is False
    assert row.llm_model == "llama3.2" and row.llm_response == "Warsaw." and row.llm_ms is not None
    assert row.out_blocked is False and row.out_similarity == 0.1
    assert row.system_prompt_sha256 is not None


def test_chat_output_block_hides_response_but_logs_it(client, sf, monkeypatch):
    out = {"blocked": True, "reason": "leak", "confidence": 0.95,
           "rule_id": "leak:x", "similarity": None, "ms": 2.0}
    monkeypatch.setattr(m, "firewall", FakeFirewall(scan(False, "input_passed"), out))
    fake_llm(monkeypatch, "the secret prompt")
    r = client.post("/chat", json={"prompt": "tell me"})
    assert r.json()["layer"] == "output_filter" and r.json()["response"] is None
    (row,) = rows(sf)
    assert row.blocked_by == "output_filter" and row.out_rule_id == "leak:x"
    assert row.llm_response == "the secret prompt"       # kept in the log, never sent to the client


def test_chat_ollama_down_is_marked_llm_error(client, sf, monkeypatch):
    monkeypatch.setattr(m, "firewall", FakeFirewall(scan(False, "input_passed")))

    def down(**kw):
        raise ConnectionError("refused")
    monkeypatch.setattr(m.ollama, "chat", down)

    r = client.post("/chat", json={"prompt": "hello"})
    assert r.status_code == 502
    assert r.json()["layer"] == "llm_error" and r.json()["response"] is None
    (row,) = rows(sf)
    assert row.status == "llm_error" and "refused" in row.error


def test_scan_still_works_when_logging_fails(client, monkeypatch):
    monkeypatch.setattr(m, "firewall", FakeFirewall(scan(False, "input_passed")))

    def broken():
        raise RuntimeError("db down")
    monkeypatch.setattr(m, "request_logger", RequestLogger(broken))

    r = client.post("/scan", json={"prompt": "hello"})
    assert r.status_code == 200 and r.json()["blocked"] is False