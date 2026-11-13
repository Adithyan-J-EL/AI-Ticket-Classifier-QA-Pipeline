"""Browser tests with Playwright, using the Page Object Model:
selectors live in one class, so tests stay short and readable."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui


class DashboardPage:
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.url = f"{base_url}/dashboard"
        self.input = page.locator("#ticket-input")
        self.submit = page.locator("#submit-btn")
        self.category = page.locator("#result-category")
        self.priority = page.locator("#result-priority")
        self.error = page.locator("#error-msg")
        self.ticket_items = page.locator("#ticket-list li")

    def open(self):
        self.page.goto(self.url)

    def submit_ticket(self, text: str):
        self.input.fill(text)
        self.submit.click()


def test_submit_ticket_shows_classification(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("I was charged twice, refund me urgently")
    expect(dashboard.category).to_have_text("billing")
    expect(dashboard.priority).to_have_text("high")


def test_ticket_appears_in_recent_list(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("My package delivery is late")
    expect(dashboard.ticket_items).to_have_count(1)
    expect(dashboard.ticket_items.first).to_contain_text("shipping")


def test_short_input_shows_error(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("hi")
    expect(dashboard.error).to_contain_text("at least 5 characters")


def test_error_is_cleared_after_a_valid_submission(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("hi")
    expect(dashboard.error).to_contain_text("at least 5 characters")
    dashboard.submit_ticket("I forgot my password")
    expect(dashboard.error).to_have_text("")
    expect(dashboard.category).to_have_text("account")


def test_old_result_is_hidden_when_next_submission_fails(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("refund my payment")
    expect(page.locator("#result")).to_be_visible()
    dashboard.submit_ticket("hi")
    expect(page.locator("#result")).to_be_hidden()   # no stale answer next to an error


def test_whitespace_only_input_is_rejected(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("          ")
    expect(dashboard.error).to_contain_text("at least 5 characters")
    expect(dashboard.ticket_items).to_have_count(0)


def test_html_in_ticket_is_shown_as_text_not_executed(page, live_server):
    """XSS check: a hostile ticket must render literally."""
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    payload = "<img src=x onerror=alert(1)> refund"
    dashboard.submit_ticket(payload)
    expect(dashboard.ticket_items).to_have_count(1)
    expect(dashboard.ticket_items.first).to_contain_text(payload)
    assert dialogs == []
    assert page.locator("#ticket-list img").count() == 0


def test_recent_list_shows_newest_first_and_caps_at_ten(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    for i in range(12):
        dashboard.submit_ticket(f"refund request number {i}")
        expect(dashboard.category).to_have_text("billing")
    expect(dashboard.ticket_items).to_have_count(10)
    expect(dashboard.ticket_items.first).to_contain_text("number 11")


def test_list_persists_after_page_reload(page, live_server):
    dashboard = DashboardPage(page, live_server)
    dashboard.open()
    dashboard.submit_ticket("My package delivery is late")
    expect(dashboard.ticket_items).to_have_count(1)
    page.reload()
    expect(dashboard.ticket_items).to_have_count(1)
