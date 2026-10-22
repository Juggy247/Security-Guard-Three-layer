import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str | None = None) -> Engine:
    """Create a database engine. Reads DATABASE_URL unless a url is given."""
    return create_engine(url or os.environ["DATABASE_URL"], pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)