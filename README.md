# Support Ticket Triage: AI Quality Pipeline

A small project that builds an AI-style component (a support-ticket classifier)
and tests it at every level a QA / AI-testing engineer should: unit, API,
database, browser (E2E), and an **AI quality gate** that fails the build if the
model regresses.

**108 tests · 99% coverage · 65-case golden dataset · CI quality gate**

## What it does
A ticket comes in ("I was charged twice, refund me") and the system returns a
**category** (billing / technical / account / shipping / general) and a
**priority** (low / medium / high). Results are stored in SQLite and shown on a
web dashboard.

## Architecture
```
Browser dashboard ──► FastAPI (/tickets) ──► Classifier (swappable) 
                            │
                            └──► SQLite
Golden dataset ──► Evaluator ──► metrics + confusion matrix ──► CI gate
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
playwright install chromium

uvicorn app.api:app --reload              # API docs: /docs | dashboard: /dashboard
pytest -m "not ui" --cov                  # unit + API + DB + evaluation (coverage gate 90%)
pytest -m ui                              # browser tests
python -m evaluation.evaluator            # AI quality report (exit 1 if below threshold)
python -m evaluation.evaluator --json     # same, machine-readable

docker build -t ticket-triage . && docker run -p 8000:8000 -v tickets:/data ticket-triage
```

## API
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/tickets` | Classify and store a ticket (422 on invalid input) |
| GET | `/tickets?limit=50` | Recent tickets, newest first (limit 1-200) |
| GET | `/tickets/{id}` | One ticket (404 if missing, 422 if id is not an int) |
| GET | `/dashboard` | Web UI |

## How the AI is evaluated
1. **Accuracy** on `evaluation/golden_dataset.json` (category and priority).
2. **Per-category accuracy** and a **minimum-class floor**, so one broken class
   can't hide behind a good average.
3. **Confusion matrix**: shows which classes get mixed up.
4. **Per-tag accuracy** (typo, non-english, negation, substring-trap,
   multi-intent...): shows what kind of input the model is weak on.
5. **Consistency**: same input, repeated runs, same output.
6. **Tests of the gate itself**: a perfect model passes, an "always general"
   model fails, and a model with one broken class fails even when its average
   looks fine.

Thresholds live in `THRESHOLDS` (`evaluation/evaluator.py`).

## Known limitations (reported, not hidden)
The current model misses 3 of 65 cases on purpose-built hard inputs:

| Tag | Example | Why it fails |
|---|---|---|
| typo | "app keeps crashng on startup" | keyword rules can't see misspellings |
| non-english | "Me cobraron dos veces..." | English-only keywords |
| negation | "It is not a billing problem, my login just fails" | rules count words, not meaning |

These are exactly the cases where a real LLM classifier should win. The dataset
is the yardstick for proving it does.

## Bugs found by testing (and fixed)
| Bug | Found by | Fix |
|---|---|---|
| "download" / "countdown" triggered the keyword `down` (false high-priority outage) | `substring-trap` golden cases | whole-word matching |
| 5 spaces passed the min-length validation | API boundary tests | strip whitespace before validating |
| Confidence was 1.0 on a single keyword hit | confidence-vs-evidence test | `hits / (total + 1)` |
| Curly apostrophe (can’t) bypassed keywords | unicode case in dataset | text normalisation |
| Dashboard kept showing the old result next to a new error | UI test | hide result on error |
| Ties between categories depended on dict order | determinism test | explicit `CATEGORY_ORDER` |
| `/tickets` had no limit control | API tests | validated `limit` parameter |

## Upgrade ideas
- Replace `KeywordClassifier` with an LLM class (same `classify()` interface).
  Consistency testing then becomes meaningful, and the 3 known misses should
  turn green.
- Add LLM-as-judge metrics (DeepEval) for free-text replies.
- Add a Postman/Newman collection and a load test (Locust) for `/tickets`.
- Track metrics per commit to chart model quality over time.
