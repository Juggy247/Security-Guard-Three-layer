from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class RequestLog(Base):
    """One row per API call (/scan or /chat)."""
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(BigInteger,primary_key=True,autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    endpoint: Mapped[str] = mapped_column(String(16))

    #prompt
    prompt_sha256: Mapped[str] = mapped_column(String(64), index=True)
    prompt_text: Mapped[str | None] = mapped_column(Text)
    prompt_chars: Mapped[int] = mapped_column(Integer)
    prompt_words: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    truncated: Mapped[bool | None] = mapped_column(Boolean)
    system_prompt_sha256: Mapped[str | None] = mapped_column(String(64)) 

    # final verdict 
    blocked: Mapped[bool] = mapped_column(Boolean, index=True)
    blocked_by: Mapped[str | None] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    error: Mapped[str | None] = mapped_column(Text)

    # layer-1 sanitizer
    san_blocked: Mapped[bool | None] = mapped_column(Boolean)
    san_rule_id: Mapped[str | None] = mapped_column(String(128))
    san_flag: Mapped[bool | None] = mapped_column(Boolean)
    san_ms: Mapped[float | None] = mapped_column(Float)

    # layer 2 - classifier
    clf_attack_prob: Mapped[float | None] = mapped_column(Float)
    clf_threshold: Mapped[float | None] = mapped_column(Float)
    clf_ms: Mapped[float | None] = mapped_column(Float)

    # LLM
    llm_model: Mapped[str | None] = mapped_column(String(64))
    llm_response: Mapped[str | None] = mapped_column(Text)
    llm_ms: Mapped[float | None] = mapped_column(Float)

    # layer 3 - output filter
    out_blocked: Mapped[bool | None] = mapped_column(Boolean)
    out_rule_id: Mapped[str | None] = mapped_column(String(128))
    out_similarity: Mapped[float | None] = mapped_column(Float)
    out_ms: Mapped[float | None] = mapped_column(Float)

    # totals and context
    total_ms: Mapped[float | None] = mapped_column(Float)
    firewall_version: Mapped[str | None] = mapped_column(String(40))
    config: Mapped[dict | None] = mapped_column(JSONB)

    # for later experiments
    run_label: Mapped[str | None] = mapped_column(String(64), index=True)
    true_label: Mapped[int | None] = mapped_column(SmallInteger)
