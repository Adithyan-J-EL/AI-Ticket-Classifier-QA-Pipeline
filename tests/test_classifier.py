import pytest

from app.classifier import KeywordClassifier, normalise

classifier = KeywordClassifier()


@pytest.mark.parametrize(
    "text, expected_category",
    [
        ("Please refund my payment", "billing"),
        ("The app has a bug and shows an error", "technical"),
        ("I forgot my password", "account"),
        ("My package delivery is late", "shipping"),
        ("Hello there, nice weather", "general"),
    ],
)
def test_category(text, expected_category):
    assert classifier.classify(text)["category"] == expected_category


@pytest.mark.parametrize(
    "text, expected_priority",
    [
        ("Urgent: refund me immediately", "high"),
        ("I was charged twice", "high"),
        ("My account is locked", "high"),
        ("Please send my invoice", "medium"),      # billing default
        ("Where is my package", "low"),            # shipping default
        ("Thanks for everything", "low"),          # general default
    ],
)
def test_priority_rules(text, expected_priority):
    assert classifier.classify(text)["priority"] == expected_priority


# --- regression tests for real bugs found in v1 -----------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Where can I download the user guide",      # "down" inside "download"
        "The countdown to the sale looks great",    # "down" inside "countdown"
        "Pricey-looking packaging, nice touch",     # "price" inside "pricey"
    ],
)
def test_keywords_match_whole_words_only(text):
    assert classifier.classify(text) == {"category": "general", "priority": "low",
                                         "confidence": 0.2}


def test_download_is_not_a_high_priority_outage():
    assert classifier.classify("How do I download my data")["priority"] != "high"


def test_whole_word_down_still_matches():
    result = classifier.classify("the site is down")
    assert result == {"category": "technical", "priority": "high",
                      "confidence": 0.5}


def test_curly_apostrophe_is_normalised():
    straight = classifier.classify("I can't get in to my account")
    curly = classifier.classify("I can’t get in to my account")
    assert straight == curly
    assert curly["category"] == "account"


# --- robustness / edge cases ------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_or_blank_input_is_general(text):
    assert classifier.classify(text) == {"category": "general", "priority": "low",
                                         "confidence": 0.2}


def test_case_and_whitespace_do_not_matter():
    a = classifier.classify("REFUND   my\n\npayment")
    b = classifier.classify("refund my payment")
    assert a == b


def test_unicode_and_emoji_do_not_crash():
    result = classifier.classify("😡 мой пакет не пришёл 日本語 refund")
    assert result["category"] == "billing"


def test_very_long_input_is_handled_quickly():
    text = "refund " * 100_000
    assert classifier.classify(text)["category"] == "billing"


def test_special_characters_are_not_treated_as_regex():
    result = classifier.classify("refund (.*) [a-z]+ \\d{3} $^")
    assert result["category"] == "billing"


def test_tie_is_broken_deterministically_by_category_order():
    # one billing hit and one technical hit: billing comes first in the order
    results = {classifier.classify("refund error")["category"] for _ in range(20)}
    assert results == {"billing"}


# --- output contract --------------------------------------------------------

def test_output_shape_and_ranges():
    result = classifier.classify("I was charged twice")
    assert set(result) == {"category", "priority", "confidence"}
    assert result["category"] in {"billing", "technical", "account", "shipping", "general"}
    assert result["priority"] in {"low", "medium", "high"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_confidence_grows_with_evidence():
    weak = classifier.classify("refund")["confidence"]
    strong = classifier.classify("refund my invoice payment")["confidence"]
    mixed = classifier.classify("refund my error")["confidence"]
    assert strong > weak > 0
    assert mixed <= weak           # competing categories lower certainty


def test_confidence_never_claims_certainty():
    assert classifier.classify("refund refund invoice payment bill")["confidence"] < 1.0


def test_is_deterministic():
    text = "The site is slow and I see an error"
    assert len({str(classifier.classify(text)) for _ in range(10)}) == 1


def test_normalise_helper():
    assert normalise("  Hello’s   WORLD ") == "hello's world"
