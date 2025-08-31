import json
from typing import Dict, Any, List, Tuple


def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rubric_to_text(rubric: Dict[str, Any]) -> str:
    lines: List[str] = []
    rubric_id = rubric.get("rubric_id", "rubric")
    lines.append(f"Rubric ID: {rubric_id}")
    for crit in rubric.get("criteria", []):
        lines.append(f"- {crit.get('criterion_id', '')}: {crit.get('name', '')}")
        pd = crit.get("performance_descriptors", {})
        for level in ["excellent", "good", "average", "needs_improvement", "poor"]:
            if level in pd:
                lines.append(f"  {level}: {pd[level]}")
    return "\n".join(lines)


def parse_json_safely(text: str) -> Tuple[bool, Any]:
    text = text.strip()
    # Some models wrap JSON in code fences or add preamble; attempt naive extraction
    try:
        return True, json.loads(text)
    except Exception:
        pass
    # Attempt to find the first '{' to last '}'
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            return True, json.loads(candidate)
    except Exception:
        pass
    return False, None


def score_detection(pred_label: str, true_label: str) -> int:
    return int(pred_label.strip().lower() == true_label.strip().lower())


def normalize_rating(value: str) -> str:
    allowed = {"excellent", "good", "average", "needs_improvement", "poor"}
    v = (value or "").strip().lower()
    return v if v in allowed else "average"


def safe_int_score(text: str) -> int:
    text = (text or "").strip()
    for ch in text:
        if ch.isdigit():
            try:
                n = int(ch)
                if 1 <= n <= 5:
                    return n
            except Exception:
                continue
    return 3

