import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_detection_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Basic accuracy; teams may extend to precision/recall per class
    total = 0
    correct = 0
    per_class = {"Human": {"tp": 0, "fp": 0, "fn": 0},
                 "AI": {"tp": 0, "fp": 0, "fn": 0},
                 "Hybrid": {"tp": 0, "fp": 0, "fn": 0}}

    for r in rows:
        true_label = r.get("label_true")
        pred_label = r.get("label_pred")
        if not true_label or not pred_label:
            continue
        total += 1
        if true_label == pred_label:
            correct += 1
            per_class[true_label]["tp"] += 1
        else:
            per_class[pred_label]["fp"] += 1 if pred_label in per_class else 0
            per_class[true_label]["fn"] += 1 if true_label in per_class else 0

    accuracy = correct / total if total else 0.0

    def f1(cls: str) -> float:
        tp = per_class[cls]["tp"]
        fp = per_class[cls]["fp"]
        fn = per_class[cls]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    macro_f1 = sum(f1(c) for c in ["Human", "AI", "Hybrid"]) / 3.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "total_evaluated": total,
    }


def summarize_feedback(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Placeholder counts; detailed quality scoring should be applied per team rubric
    with_json = sum(1 for r in rows if r.get("feedback_json") is not None)
    avg_length = 0.0
    n = 0
    for r in rows:
        txt = r.get("feedback_raw") or ""
        if txt:
            avg_length += len(txt.split())
            n += 1
    avg_length = (avg_length / n) if n else 0.0
    return {"count": len(rows), "with_json_like": with_json, "avg_word_count": avg_length}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI pipeline outputs")
    parser.add_argument("--detect", type=str, required=True, help="Path to predictions_detection.jsonl")
    parser.add_argument("--feedback", type=str, required=True, help="Path to predictions_feedback.jsonl")
    parser.add_argument("--out", type=str, default="outputs/eval.json", help="Where to write summary")
    args = parser.parse_args()

    det_rows = load_json(Path(args.detect))
    fb_rows = load_json(Path(args.feedback))

    det_metrics = compute_detection_metrics(det_rows)
    fb_summary = summarize_feedback(fb_rows)

    summary = {
        "detection": det_metrics,
        "feedback": fb_summary,
        "notes": "Apply the evaluation team's detailed criteria externally; this file aggregates basics.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()