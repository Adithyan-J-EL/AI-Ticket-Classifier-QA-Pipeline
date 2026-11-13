"""Evaluate a classifier against a golden dataset.

Ideas that matter when testing AI systems:
  1. Accuracy against known-good answers (golden dataset)
  2. Per-category breakdown, so one weak class can't hide behind a good average
  3. A confusion matrix, so you see WHICH classes get mixed up, not just how many
  4. Per-tag breakdown (typos, non-English, negation...) to locate weak spots
  5. Consistency: the same input should give the same output every time

Run directly:  python -m evaluation.evaluator          (human-readable)
               python -m evaluation.evaluator --json   (machine-readable)
Exits with code 1 if a threshold is missed, so CI can fail the build.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from app.classifier import CATEGORY_ORDER, BaseClassifier, KeywordClassifier

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
LABELS = CATEGORY_ORDER + ["general"]

THRESHOLDS = {
    "category_accuracy": 0.90,
    "priority_accuracy": 0.90,
    "consistency": 1.0,
    "min_category_accuracy": 0.80,  # the weakest single class must still pass
}


def load_dataset(path: Path = DATASET_PATH) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ratio(correct: int, total: int) -> float:
    return round(correct / total, 3) if total else 0.0


def evaluate_accuracy(classifier: BaseClassifier, dataset: list) -> dict:
    if not dataset:
        raise ValueError("dataset is empty")

    cat_correct = pri_correct = 0
    per_category = defaultdict(lambda: [0, 0])  # [correct, total]
    per_tag = defaultdict(lambda: [0, 0])
    confusion = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}
    failures = []

    for case in dataset:
        got = classifier.classify(case["text"])
        cat_ok = got["category"] == case["category"]
        pri_ok = got["priority"] == case["priority"]
        cat_correct += cat_ok
        pri_correct += pri_ok

        per_category[case["category"]][1] += 1
        per_category[case["category"]][0] += cat_ok
        tag = case.get("tag", "untagged")
        per_tag[tag][1] += 1
        per_tag[tag][0] += cat_ok and pri_ok

        row = confusion.setdefault(case["category"], {})
        row[got["category"]] = row.get(got["category"], 0) + 1

        if not (cat_ok and pri_ok):
            failures.append({"text": case["text"], "tag": tag,
                             "expected": {"category": case["category"],
                                          "priority": case["priority"]},
                             "got": got})

    n = len(dataset)
    return {
        "total_cases": n,
        "category_accuracy": _ratio(cat_correct, n),
        "priority_accuracy": _ratio(pri_correct, n),
        "per_category": {c: _ratio(*v) for c, v in per_category.items()},
        "min_category_accuracy": min(_ratio(*v) for v in per_category.values()),
        "per_tag": {t: _ratio(*v) for t, v in per_tag.items()},
        "confusion": confusion,
        "failures": failures,
    }


def evaluate_consistency(classifier: BaseClassifier, dataset: list, runs: int = 5) -> float:
    """Fraction of inputs that give identical output on every repeated run.

    Deterministic code scores 1.0. A real LLM at temperature > 0 will not,
    which is exactly why this check exists.
    """
    stable = 0
    for case in dataset:
        outputs = {json.dumps(classifier.classify(case["text"]), sort_keys=True)
                   for _ in range(runs)}
        stable += len(outputs) == 1
    return _ratio(stable, len(dataset))


def run_evaluation(classifier: BaseClassifier = None, dataset: list = None) -> dict:
    classifier = classifier or KeywordClassifier()
    dataset = dataset if dataset is not None else load_dataset()
    report = evaluate_accuracy(classifier, dataset)
    report["consistency"] = evaluate_consistency(classifier, dataset)
    report["passed"] = all(report[k] >= t for k, t in THRESHOLDS.items())
    return report


def format_confusion(confusion: dict) -> str:
    width = max(len(label) for label in LABELS) + 2
    lead = len("actual \\ predicted") + 2
    lines = ["actual \\ predicted".ljust(lead) + "".join(l.ljust(width) for l in LABELS)]
    for actual in LABELS:
        row = confusion.get(actual, {})
        lines.append(actual.ljust(lead) + "".join(str(row.get(p, 0)).ljust(width) for p in LABELS))
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    report = run_evaluation()

    if "--json" in argv:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1

    print(f"=== Evaluation Report ({report['total_cases']} cases) ===")
    for key in THRESHOLDS:
        status = "PASS" if report[key] >= THRESHOLDS[key] else "FAIL"
        print(f"{key:24} {report[key]:.3f}  (min {THRESHOLDS[key]})  {status}")
    print("\nper category:", report["per_category"])
    print("per tag     :", report["per_tag"])
    print("\nconfusion matrix:\n" + format_confusion(report["confusion"]))
    if report["failures"]:
        print(f"\n{len(report['failures'])} miss(es):")
    for f in report["failures"]:
        print(f"  [{f['tag']}] {f['text']}\n      expected {f['expected']}  got {f['got']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
