"""Shared fixtures. Every test gets its own temporary database."""
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_tickets.db")


@pytest.fixture
def client(db_path):
    return TestClient(create_app(db_path=db_path))


@pytest.fixture
def live_server(db_path):
    """Run the real app on a free port in a background thread (for UI tests)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(create_app(db_path=db_path), host="127.0.0.1", port=port,
                       log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)
