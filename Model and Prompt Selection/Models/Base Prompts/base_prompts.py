from typing import List, Dict, Any

SYSTEM_PROMPT = "You are a careful academic assistant. Be precise and return strict JSON."

def build_detection_prompt(submission: str, few_shots: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Academic Integrity Detector Prompt
    ----------------------------------
    Purpose:
        Classifies student submissions as Human, AI, or Hybrid (AI-assisted).

    Technique:
        - Role-based prompting
        - Few-shot support
        - CoT (reasoning encouraged but hidden from output)
        - Strict JSON schema output

    JSON Schema (strict):
        {
          "label": "Human|AI|Hybrid",
          "rationale": "1–3 short bullet points of evidence",
          "flags": ["style_inconsistency","high_verbatim","generic_phrasing","none"]
        }
    """
    # Build few-shot block
    shot_texts = []
    for s in few_shots:
        shot_texts.append(
            f'Submission: """{s.get("final_submission","")}"""\n'
            f'Your analysis (2–4 bullet points): <analysis>\n'
            f'Label: {s.get("label_type","")}\n'
        )
    examples_block = "\n\n".join(shot_texts) if shot_texts else "/* no examples available */"

    # User-facing content
    user = f"""
You are an AI text-source classifier for academic integrity.
Decide whether the student submission is Human, AI, or Hybrid (AI-assisted).

Guidelines:
- Consider discourse features (specificity, subjectivity, personal context), style consistency, local/global coherence, repetitiveness, and cliché patterns.
- Hybrid = meaningful human writing with some AI assistance (ideas, phrasing, structure), or explicit admission of mixed use.

Examples:
{examples_block}

Now analyze the NEW submission step by step and return STRICT JSON.
NEW submission:
\"\"\"{submission}\"\"\"\n
Think briefly, then answer only with the JSON object.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_feedback_prompt(domain: str, assignment_prompt: str, rubric_text: str, submission: str) -> List[Dict[str, str]]:
    """
    Rubric-Aligned Feedback Prompt
    ------------------------------
    Purpose:
        Generates structured, supportive feedback for a student submission.

    Technique:
        - Role-based prompting
        - Rubric-grounded evaluation
        - Structured JSON report

    JSON Schema (strict):
        {
          "overall_summary": "2–4 sentence overview",
          "criteria_feedback": [
              {
                "criterion_id": "...",
                "rating": "excellent|good|average|needs_improvement|poor",
                "evidence": ["bullet","points"],
                "improvement_tip": "one concrete step"
              },
              ...
          ],
          "suggested_grade": "short string (optional)"
        }
    """
    user = f"""
You are a supportive assessor. Provide actionable feedback aligned to the rubric.
Return a STRUCTURED report (no extraneous text).

Sections:
1) "overall_summary": 2–4 sentences on strengths and priorities.
2) "criteria_feedback": array of items, one per rubric criterion with fields:
   - "criterion_id"
   - "rating": one of ["excellent","good","average","needs_improvement","poor"]
   - "evidence": 1–3 bullet points citing concrete excerpt(s) or behaviors
   - "improvement_tip": one concrete next step

Context:
- Domain: {domain}
- Assignment prompt: {assignment_prompt}

Rubric (verbatim):
{rubric_text}

Student submission:
\"\"\"{submission}\"\"\"\n

Constraints:
- Be concise but specific. Do not invent rubric fields. If evidence is insufficient, say so.
- Output MUST be valid JSON with the exact top-level keys: overall_summary, criteria_feedback, suggested_grade.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

@staticmethod
def self_eval_prompt(rubric: Dict, essay: str, feedback: str) -> str:
    crit = [c.get('name','Criterion') for c in rubric.get('criteria',[])]
    crit_str = ", ".join(crit) if crit else "the rubric"
    return (
        "You are a strict but fair assessor. Rate how well the FEEDBACK addresses the rubric for the ESSAY.\n"
        "Rate on a 1-5 scale (integers only). Provide ONLY the number.\n\n"
        f"ESSAY:\n{essay}\n\nRUBRIC CRITERIA: {crit_str}\n\nFEEDBACK:\n{feedback}\n\nRATING (1-5): "
    )