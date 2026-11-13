"""FastAPI app: classify support tickets and store them."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from app import database
from app.classifier import KeywordClassifier

DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"


class TicketIn(BaseModel):
    # strip first, so "      " (spaces only) fails min_length instead of passing
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=5, max_length=1000)


def create_app(db_path: str = "tickets.db", classifier=None) -> FastAPI:
    """App factory so tests can inject a temp database / fake classifier."""
    app = FastAPI(title="Ticket Triage API")
    classifier = classifier or KeywordClassifier()
    database.init_db(db_path)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/tickets", status_code=201)
    def create_ticket(ticket: TicketIn):
        result = classifier.classify(ticket.text)
        ticket_id = database.save_ticket(db_path, ticket.text, result)
        return {"id": ticket_id, "text": ticket.text, **result}

    @app.get("/tickets")
    def get_tickets(limit: int = Query(50, ge=1, le=200)):
        return database.list_tickets(db_path, limit=limit)

    @app.get("/tickets/{ticket_id}")
    def get_one(ticket_id: int):
        ticket = database.get_ticket(db_path, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_FILE.read_text(encoding="utf-8")

    return app


app = create_app(db_path=os.environ.get("TICKET_DB", "tickets.db"))
