"""Quality gate for the AI component itself, plus tests OF the gate."""
import json
from collections import Counter

import pytest

from app.classifier import BaseClassifier
from evaluation.evaluator import (LABELS, THRESHOLDS, evaluate_accuracy,
                                  evaluate_consistency, format_confusion,
                                  load_dataset, main, run_evaluation)


class Oracle(BaseClassifier):
    """Always right: looks the answer up in the dataset."""
    def __init__(self, dataset):
        self.answers = {c["text"]: c for c in dataset}

    def classify(self, text):
        c = self.answers[text]
        return {"category": c["category"], "priority": c["priority"], "confidence": 1.0}


class AlwaysGeneral(BaseClassifier):
    def classify(self, text):
        return {"category": "general", "priority": "low", "confidence": 0.1}


# --- the dataset itself -----------------------------------------------------

def test_golden_dataset_is_well_formed():
    for case in load_dataset():
        assert {"text", "category", "priority", "tag"} <= set(case)
        assert case["category"] in LABELS
        assert case["priority"] in {"low", "medium", "high"}


def test_golden_dataset_has_no_duplicate_texts():
    texts = [c["text"].strip().lower() for c in load_dataset()]
    assert len(texts) == len(set(texts))


def test_golden_dataset_covers_every_category_reasonably():
    counts = Counter(c["category"] for c in load_dataset())
    assert set(counts) == set(LABELS)
    assert min(counts.values()) >= 8, counts


def test_golden_dataset_includes_hard_cases():
    tags = {c["tag"] for c in load_dataset()}
    assert {"typo", "non-english", "negation", "substring-trap", "multi-intent"} <= tags


# --- the model against the gate --------------------------------------------

def test_model_meets_quality_thresholds():
    report = run_evaluation()
    for metric, minimum in THRESHOLDS.items():
        assert report[metric] >= minimum, f"{metric}={report[metric]} < {minimum}"


def test_no_category_is_completely_broken():
    report = run_evaluation()
    assert all(score >= 0.8 for score in report["per_category"].values())


def test_substring_traps_never_regress():
    """Regression guard for the 'download' -> 'down' bug."""
    assert run_evaluation()["per_tag"]["substring-trap"] == 1.0


def test_confusion_matrix_is_consistent_with_accuracy():
    report = run_evaluation()
    diag = sum(report["confusion"][l][l] for l in LABELS)
    total = sum(sum(row.values()) for row in report["confusion"].values())
    assert total == report["total_cases"]
    assert diag / total == pytest.approx(report["category_accuracy"], abs=0.001)


# --- tests OF the evaluator (does the gate really catch bad models?) ---------

def test_consistency_check_catches_flaky_model():
    class FlakyClassifier(BaseClassifier):
        def __init__(self):
            self.calls = 0

        def classify(self, text):
            self.calls += 1
            return {"category": f"cat{self.calls}", "priority": "low", "confidence": 0.5}

    assert evaluate_consistency(FlakyClassifier(), load_dataset()) < 1.0


def test_perfect_model_scores_100_percent_and_passes():
    data = load_dataset()
    report = run_evaluation(Oracle(data), data)
    assert report["category_accuracy"] == report["priority_accuracy"] == 1.0
    assert report["failures"] == [] and report["passed"] is True


def test_lazy_model_fails_the_gate():
    """A model that always says 'general' must be rejected, and per-class
    accuracy must expose it even though 'general' alone looks perfect."""
    report = run_evaluation(AlwaysGeneral())
    assert report["passed"] is False
    assert report["per_category"]["general"] == 1.0
    assert report["per_category"]["billing"] == 0.0
    assert report["min_category_accuracy"] == 0.0


def test_one_weak_class_fails_gate_even_if_average_is_good():
    data = load_dataset()

    class BreaksBilling(Oracle):
        def classify(self, text):
            result = super().classify(text)
            if result["category"] == "billing":
                result["category"] = "general"
            return result

    report = run_evaluation(BreaksBilling(data), data)
    assert report["category_accuracy"] > 0.7       # average still looks OK
    assert report["passed"] is False               # ...but the gate says no


def test_empty_dataset_is_an_error_not_a_pass():
    with pytest.raises(ValueError):
        evaluate_accuracy(Oracle([]), [])


def test_failures_report_expected_and_actual():
    report = evaluate_accuracy(AlwaysGeneral(), load_dataset()[:3])
    failure = report["failures"][0]
    assert {"text", "tag", "expected", "got"} <= set(failure)


def test_confusion_matrix_formats_all_labels():
    text = format_confusion(run_evaluation()["confusion"])
    for label in LABELS:
        assert label in text


# --- CLI / CI contract ------------------------------------------------------

def test_main_exits_zero_when_gate_passes(capsys):
    assert main([]) == 0
    assert "Evaluation Report" in capsys.readouterr().out


def test_main_json_output_is_valid(capsys):
    assert main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_main_exits_one_when_gate_fails(monkeypatch, capsys):
    monkeypatch.setitem(THRESHOLDS, "category_accuracy", 1.01)  # impossible bar
    assert main([]) == 1
    assert "FAIL" in capsys.readouterr().out
