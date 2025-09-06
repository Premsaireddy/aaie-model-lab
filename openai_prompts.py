import json
import re
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = "You are a careful academic assistant. Be precise."


def render_chat_messages(system_prompt: str, user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_detection_user_block(submission: str, few_shots: List[Dict[str, Any]]) -> str:
    shot_texts: List[str] = []
    for shot in few_shots or []:
        shot_texts.append(
            f'Submission: """{shot.get("final_submission", "")}"""\n'
            f'Your analysis (2–4 bullet points): <analysis>\n'
            f'Label: {shot.get("label_type", "")}\n'
        )
    examples_block = "\n\n".join(shot_texts) if shot_texts else "/* no examples available */"

    return (
        "You are an AI text-source classifier for academic integrity.\n"
        "Decide whether the student submission is Human, AI, or Hybrid (AI-assisted).\n"
        "Guidelines:\n"
        "- Consider discourse features (specificity, subjectivity, personal context), style consistency, local/global coherence, repetitiveness, and cliché patterns.\n"
        "- Hybrid = meaningful human writing with some AI assistance, or explicit admission of mixed use.\n"
        "Examples:\n"
        f"{examples_block}\n"
        "Now analyze the NEW submission and respond in plain text with the following structure:\n"
        "Label: ...\n"
        "Rationale:\n"
        "- point 1\n"
        "- point 2\n"
        "Flags: ...\n"
        "NEW submission:\n"
        f'"""{submission}"""'
    )


def build_feedback_user_block(domain: str, assignment_prompt: str, rubric_text: str, submission: str) -> str:
    return (
        "You are a supportive assessor. Provide actionable feedback aligned to the rubric.\n"
        "Return plain structured text only (no JSON, no files).\n"
        "Sections to include:\n"
        "1) Overall Summary: 2–4 sentences on strengths and priorities.\n"
        "2) Criteria Feedback: for each rubric criterion include:\n"
        "   - Criterion\n"
        "   - Rating (excellent, good, average, needs_improvement, poor)\n"
        "   - Evidence (1–3 bullet points citing excerpts or behaviors)\n"
        "   - Improvement Tip: one concrete step\n"
        "3) Suggested Grade: short string (optional)\n"
        "Context:\n"
        f"- Domain: {domain}\n"
        f"- Assignment prompt: {assignment_prompt}\n"
        "Rubric (verbatim):\n"
        f"{rubric_text}\n"
        "Student submission:\n"
        f'"""{submission}"""'
    )


def format_rubric(rubric: Dict[str, Any]) -> str:
    parts: List[str] = [f"Rubric ID: {rubric['rubric_id']}", "", "Criteria:"]
    for item in rubric.get('criteria', []):
        parts.append(
            f"Criterion: {item['criterion_id']}\n"
            f"Name: {item['name']}\n"
            f"Description: {item['description']}\n"
            f"Performance Descriptors:"
        )
        for key, val in (item.get('performance_descriptors') or {}).items():
            parts.append(f"  - {key}: {val}")
    return "\n".join(parts)


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    l = label.strip().lower()
    if "human" in l:
        return "Human"
    if l == "ai" or " ai" in l or l.startswith("ai"):
        return "AI"
    if "hybrid" in l:
        return "Hybrid"
    return None


def extract_detection_fields(text: str) -> Dict[str, Any]:
    # Try JSON first
    try:
        obj = json.loads(text)
        label = _normalize_label(obj.get("label"))
        rationale = obj.get("rationale")
        if isinstance(rationale, str):
            rationale = [r.strip("- ").strip() for r in rationale.split("\n") if r.strip()]
        flags = obj.get("flags") if isinstance(obj.get("flags"), list) else None
        return {
            "label": label,
            "rationale": rationale if isinstance(rationale, list) else None,
            "flags": flags,
        }
    except Exception:
        pass

    # Fallback: regex extraction from plain text
    label_match = re.search(r"(?im)^\s*Label\s*:\s*([^\n\r]+)", text)
    label = _normalize_label(label_match.group(1) if label_match else None)

    rat_block = None
    rat_start = re.search(r"(?im)^\s*Rationale\s*:", text)
    if rat_start:
        start_idx = rat_start.end()
        end_match = re.search(r"(?im)^\s*Flags\s*:", text[start_idx:])
        rat_block = text[start_idx:start_idx + end_match.start()] if end_match else text[start_idx:]

    rationale = None
    if rat_block:
        lines = [ln.strip() for ln in rat_block.splitlines()]
        rationale = [re.sub(r"^\s*[-•]\s*", "", ln).strip() for ln in lines if ln.strip()]

    flags = None
    flags_match = re.search(r"(?im)^\s*Flags\s*:\s*([^\n\r]+)", text)
    if flags_match:
        flags = [f.strip() for f in re.split(r"[;,]", flags_match.group(1)) if f.strip()]

    return {"label": label, "rationale": rationale, "flags": flags}


def attempt_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None