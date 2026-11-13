"""Ticket classifier.

A rule-based stand-in for an LLM. Every classifier implements the same
`classify()` interface, so a real LLM-backed class (OpenAI, Gemini, Claude...)
can be swapped in without touching the API or the tests.

Design notes (each one fixes a bug found while testing the first version):
  * Keywords match on WORD BOUNDARIES. Substring matching made "download" hit
    the keyword "down" and "pricey" hit "price".
  * Text is normalised first (curly apostrophes, case, repeated whitespace).
  * Ties between categories are broken by a fixed order, so output is stable.
  * Confidence reflects how much evidence there is, not just who won: a single
    keyword hit is 0.5, not 1.0.
"""
import re
from abc import ABC, abstractmethod

CATEGORY_KEYWORDS = {
    "billing": [
        "refund", "refunds", "refunded", "charged", "charge", "overcharged",
        "invoice", "invoices", "payment", "payments", "billing", "bill",
        "price", "subscription", "receipt", "credit card",
    ],
    "technical": [
        "error", "errors", "crash", "crashes", "crashing", "crashed", "bug",
        "bugs", "not working", "doesn't work", "slow", "down", "broken",
        "freezes", "frozen", "glitch", "timeout", "500",
    ],
    "account": [
        "password", "login", "log in", "sign in", "account", "email change",
        "locked", "username", "profile", "2fa", "verification code",
        "can't get in", "cannot get in", "can't access", "cannot access",
    ],
    "shipping": [
        "delivery", "delivered", "shipping", "shipped", "package", "parcel",
        "tracking", "late order", "courier", "arrived",
    ],
}

HIGH_PRIORITY_WORDS = [
    "urgent", "urgently", "immediately", "charged twice", "double charged",
    "locked", "down", "asap", "emergency", "fraud", "hacked",
]

# Fixed order = deterministic tie-breaking.
CATEGORY_ORDER = list(CATEGORY_KEYWORDS)
MEDIUM_PRIORITY_CATEGORIES = {"billing", "technical", "account"}

MAX_CHARS = 5000  # guard against pathological input reaching the regex engine


def _compile(words):
    # Whole-word / whole-phrase match. \b works for "log in" and "charged twice".
    return [(w, re.compile(rf"(?<!\w){re.escape(w)}(?!\w)")) for w in words]


_CATEGORY_PATTERNS = {cat: _compile(kws) for cat, kws in CATEGORY_KEYWORDS.items()}
_HIGH_PATTERNS = _compile(HIGH_PRIORITY_WORDS)


def normalise(text: str) -> str:
    text = text[:MAX_CHARS].lower()
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, text: str) -> dict:
        """Return {'category': str, 'priority': str, 'confidence': float}."""


class KeywordClassifier(BaseClassifier):
    def classify(self, text: str) -> dict:
        lowered = normalise(text or "")

        scores = {
            cat: sum(1 for _, pat in pats if pat.search(lowered))
            for cat, pats in _CATEGORY_PATTERNS.items()
        }
        best_score = max(scores.values())
        # first category in CATEGORY_ORDER with the top score wins ties
        best_cat = next(c for c in CATEGORY_ORDER if scores[c] == best_score)
        category = best_cat if best_score > 0 else "general"

        if any(pat.search(lowered) for _, pat in _HIGH_PATTERNS):
            priority = "high"
        elif category in MEDIUM_PRIORITY_CATEGORIES:
            priority = "medium"
        else:
            priority = "low"

        total_hits = sum(scores.values())
        # +1 keeps a lone hit from claiming certainty; ties lower it further.
        confidence = round(best_score / (total_hits + 1), 2) if total_hits else 0.2
        return {"category": category, "priority": priority, "confidence": confidence}
