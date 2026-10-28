import os
from pathlib import Path
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from three_layer_firewall.sanitizer import PromptSanitizer
from three_layer_firewall.output_filter import OutputFilter
from three_layer_firewall.firewall import Firewall
from three_layer_firewall.db.models import Base
from three_layer_firewall.db.session import make_session_factory

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful customer service assistant for AcmeCorp. "
    "Only answer questions about our products."
)

@pytest.fixture(scope="session")
def sanitizer():
    return PromptSanitizer()

@pytest.fixture(scope="session")
def output_filter():
    return OutputFilter(system_prompt=SYSTEM_PROMPT)

@pytest.fixture(scope="session")
def firewall():
    return Firewall(system_prompt=SYSTEM_PROMPT)

@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not set")
    eng = create_engine(make_url(url).set(database="firewall_test"))
    assert eng.url.database.endswith("_test")      # never touch the real logs
    try:
        with eng.connect() as conn:
            conn.execute(text("select 1"))
    except Exception as e:
        pytest.skip(f"test database not reachable: {type(e).__name__}")
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def sf(engine):
    """Session factory on an empty table."""
    with engine.begin() as conn:
        conn.execute(text("truncate requests restart identity"))
    return make_session_factory(engine)