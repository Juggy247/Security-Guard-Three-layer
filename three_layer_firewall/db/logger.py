import hashlib
import logging

from sqlalchemy.orm import Session, sessionmaker

from three_layer_firewall.db.models import RequestLog

logger = logging.getLogger(__name__)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RequestLogger:
    """Turns a firewall result into a RequestLog row and saves it.

    If session_factory is None, nothing is saved (logging switched off).
    A failure while logging is reported but never breaks the API request.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None, *,
                 store_text: bool = False, firewall_version: str | None = None,
                 config: dict | None = None):
        self.session_factory = session_factory
        self.store_text = store_text          # keep prompt/response text, or only hashes + numbers
        self.firewall_version = firewall_version
        self.config = config

    def build_row(self, *, endpoint, prompt, scan_result, system_prompt=None,
                  output_result=None, llm_model=None, llm_response=None,
                  llm_ms=None, status="ok", error=None, total_ms=None,
                  run_label=None) -> RequestLog:
        d = scan_result.get("details", {})

        # final verdict: the first layer that blocked
        if scan_result["blocked"]:
            blocked, blocked_by = True, scan_result["layer"]
        elif output_result is not None and output_result["blocked"]:
            blocked, blocked_by = True, "output_filter"
        else:
            blocked, blocked_by = False, None

        return RequestLog(
            endpoint=endpoint,
            prompt_sha256=sha256(prompt),
            prompt_text=prompt if self.store_text else None,
            prompt_chars=len(prompt),
            prompt_words=len(prompt.split()),
            prompt_tokens=d.get("prompt_tokens"),
            truncated=d.get("truncated"),
            system_prompt_sha256=sha256(system_prompt) if system_prompt is not None else None,
            blocked=blocked,
            blocked_by=blocked_by,
            status=status,
            error=error,
            san_blocked=d.get("san_blocked"),
            san_rule_id=d.get("san_rule_id"),
            san_flag=d.get("san_flag"),
            san_ms=d.get("san_ms"),
            clf_attack_prob=d.get("clf_attack_prob"),
            clf_threshold=d.get("clf_threshold"),
            clf_ms=d.get("clf_ms"),
            llm_model=llm_model,
            llm_response=llm_response if self.store_text else None,
            llm_ms=llm_ms,
            out_blocked=output_result["blocked"] if output_result else None,
            out_rule_id=output_result.get("rule_id") if output_result else None,
            out_similarity=output_result.get("similarity") if output_result else None,
            out_ms=output_result.get("ms") if output_result else None,
            total_ms=total_ms,
            firewall_version=self.firewall_version,
            config=self.config,
            run_label=run_label,
        )

    def log(self, **fields) -> None:
        """Build and save one row. Never raises."""
        if self.session_factory is None:
            return
        try:
            row = self.build_row(**fields)
            with self.session_factory() as session:
                session.add(row)
                session.commit()
        except Exception:
            logger.exception("Could not write request log row")