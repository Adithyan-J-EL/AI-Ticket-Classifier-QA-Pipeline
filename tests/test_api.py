import pytest
from fastapi.testclient import TestClient

from app.api import create_app


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_ticket_returns_classification(client):
    response = client.post("/tickets", json={"text": "I was charged twice, refund me"})
    assert response.status_code == 201
    body = response.json()
    assert body["category"] == "billing"
    assert body["priority"] == "high"
    assert body["text"] == "I was charged twice, refund me"
    assert "id" in body and 0 <= body["confidence"] <= 1


def test_ticket_is_persisted(client):
    created = client.post("/tickets", json={"text": "My package is late"}).json()
    fetched = client.get(f"/tickets/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["category"] == "shipping"
    assert fetched.json()["text"] == "My package is late"


def test_unknown_ticket_is_404(client):
    response = client.get("/tickets/12345")
    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"


@pytest.mark.parametrize("bad_id", ["abc", "1.5", "-"])
def test_non_integer_ticket_id_is_422(client, bad_id):
    assert client.get(f"/tickets/{bad_id}").status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "hi"},
        {"text": "x" * 1001},
        {"wrong_field": "abc def"},
        {"text": None},
        {"text": 12345},
        {"text": ["a", "b"]},
        {"text": "      "},          # whitespace only must not count as content
        {"text": "  hi  "},         # ...and padding must not satisfy min length
    ],
)
def test_invalid_payload_is_rejected(client, payload):
    assert client.post("/tickets", json=payload).status_code == 422


@pytest.mark.parametrize("length, expected", [(4, 422), (5, 201), (1000, 201), (1001, 422)])
def test_text_length_boundaries(client, length, expected):
    assert client.post("/tickets", json={"text": "a" * length}).status_code == expected


def test_malformed_json_is_422(client):
    response = client.post("/tickets", content="{not json",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_invalid_requests_are_not_stored(client):
    client.post("/tickets", json={"text": "hi"})
    assert client.get("/tickets").json() == []


def test_text_is_trimmed_before_saving(client):
    body = client.post("/tickets", json={"text": "   refund me please   "}).json()
    assert body["text"] == "refund me please"


def test_list_returns_created_tickets_newest_first(client):
    client.post("/tickets", json={"text": "login not working"})
    client.post("/tickets", json={"text": "invoice is wrong"})
    tickets = client.get("/tickets").json()
    assert [t["text"] for t in tickets] == ["invoice is wrong", "login not working"]


def test_empty_list_is_ok(client):
    response = client.get("/tickets")
    assert response.status_code == 200 and response.json() == []


def test_list_limit_parameter(client):
    for i in range(5):
        client.post("/tickets", json={"text": f"refund request {i}"})
    assert len(client.get("/tickets?limit=2").json()) == 2


@pytest.mark.parametrize("limit", [0, -1, 201, "abc"])
def test_list_limit_out_of_range_is_422(client, limit):
    assert client.get(f"/tickets?limit={limit}").status_code == 422


def test_unicode_ticket_roundtrip(client):
    created = client.post("/tickets", json={"text": "Mi paquete llegó tarde ☕ 日本語"}).json()
    assert client.get(f"/tickets/{created['id']}").json()["text"] == created["text"]


def test_hostile_text_is_stored_not_executed(client):
    evil = "<script>alert(1)</script> '; DROP TABLE tickets; --"
    created = client.post("/tickets", json={"text": evil}).json()
    assert client.get(f"/tickets/{created['id']}").json()["text"] == evil
    assert len(client.get("/tickets").json()) == 1


@pytest.mark.parametrize("method, path", [("delete", "/tickets/1"), ("put", "/tickets"),
                                          ("post", "/health")])
def test_unsupported_methods_are_405(client, method, path):
    assert getattr(client, method)(path).status_code == 405


def test_dashboard_serves_html(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ticket-input" in response.text


def test_data_survives_app_restart(db_path):
    first = TestClient(create_app(db_path=db_path))
    ticket_id = first.post("/tickets", json={"text": "refund please"}).json()["id"]
    second = TestClient(create_app(db_path=db_path))  # new app, same file
    assert second.get(f"/tickets/{ticket_id}").status_code == 200


def test_api_works_with_fake_classifier(db_path):
    """Dependency injection lets us test the API without the real model."""
    class FakeClassifier:
        def classify(self, text):
            return {"category": "general", "priority": "low", "confidence": 1.0}

    client = TestClient(create_app(db_path=db_path, classifier=FakeClassifier()))
    body = client.post("/tickets", json={"text": "anything at all"}).json()
    assert body["category"] == "general"


def test_classifier_crash_returns_500_and_stores_nothing(db_path):
    class BrokenClassifier:
        def classify(self, text):
            raise RuntimeError("model unavailable")

    client = TestClient(create_app(db_path=db_path, classifier=BrokenClassifier()),
                        raise_server_exceptions=False)
    assert client.post("/tickets", json={"text": "refund please"}).status_code == 500
    assert client.get("/tickets").json() == []
