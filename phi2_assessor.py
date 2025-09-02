import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_ID = "microsoft/phi-2"


def render_chat_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
	"""
	Render chat messages with tokenizer's chat template if available; otherwise fallback.
	"""
	apply_chat = getattr(tokenizer, "apply_chat_template", None)
	if callable(apply_chat):
		try:
			return apply_chat(messages, tokenize=False, add_generation_prompt=True)
		except Exception:
			pass
	# Fallback: simple role-tagged transcript
	parts: List[str] = []
	for m in messages:
		parts.append(f"{m['role'].capitalize()}: {m['content']}")
	parts.append("Assistant:")
	return "\n".join(parts)


SYSTEM_PROMPT = "You are a careful academic assistant. Be precise and give clear structured output (not JSON, not CSV, no files)."


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
	shot_texts: List[str] = []
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


def self_eval_prompt(rubric: Dict[str, Any], essay: str, feedback: str) -> str:
	criteria_names = [c.get('name', 'Criterion') for c in rubric.get('criteria', [])]
	criteria_joined = ", ".join(criteria_names) if criteria_names else "(no criteria provided)"
	return (
		"You are performing self-evaluation of the generated feedback against the rubric.\n\n"
		f"Rubric criteria: {criteria_joined}\n\n"
		"Essay:\n" + essay + "\n\n"
		"Feedback JSON:\n" + feedback + "\n\n"
		"Write 3 short bullets on: fidelity to rubric, evidence quality, and actionability."
	)


def format_rubric(rubric: Dict[str, Any]) -> str:
	formatted_rubric = f"Rubric ID: {rubric['rubric_id']}\n\nCriteria:\n"
	for item in rubric['criteria']:
		formatted_rubric += f"Criterion: {item['criterion_id']}\nName: {item['name']}\nDescription: {item['description']}\nPerformance Descriptors:\n"
		for key, val in item['performance_descriptors'].items():
			formatted_rubric += f"  - {key}: {val}\n"
	return formatted_rubric


def init_pipeline() -> Tuple[Any, Any, int]:
	using_gpu = torch.cuda.is_available()
	device = 0 if using_gpu else -1
	print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		device_map="auto" if using_gpu else None,
		torch_dtype=torch.float16 if using_gpu else torch.float32,
		low_cpu_mem_usage=True,
		trust_remote_code=True,
	)
	text_gen = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		device=device,
		max_new_tokens=768,
		do_sample=True,
		temperature=0.7,
		top_p=0.95,
		top_k=50,
	)
	return tokenizer, text_gen, tokenizer.eos_token_id


def generate_text(text_gen, prompt_text: str, eos_token_id: int) -> str:
	outputs = text_gen(prompt_text, return_full_text=False, eos_token_id=eos_token_id)
	return outputs[0]["generated_text"].strip()


def run_dataset(tokenizer, text_gen, eos_token_id: int, data_path: Path) -> None:
	if not data_path.exists():
		print(f"Warning: {data_path} not found; skipping.")
		return
	with data_path.open() as f:
		data = json.load(f)

	rubric_text = format_rubric(data['rubric'])
	few_shots = data.get("few_shots", [])

	for i, submission in enumerate(data['submissions'], 1):
		submission_text = submission['final_submission']
		label_type = submission.get("label_type", "Unknown")

		feedback_messages = build_feedback_prompt(
			domain=data['domain'],
			assignment_prompt=data.get("prompt", "Analyze student submission and provide detailed feedback"),
			rubric_text=rubric_text,
			submission=submission_text,
		)
		feedback_prompt_text = render_chat_messages(tokenizer, feedback_messages)
		feedback_response = generate_text(text_gen, feedback_prompt_text, eos_token_id)

		detection_messages = build_detection_prompt(submission_text, few_shots)
		detection_prompt_text = render_chat_messages(tokenizer, detection_messages)
		detection_response = generate_text(text_gen, detection_prompt_text, eos_token_id)

		print(f"\n--- {data_path.name} | SUBMISSION {i} (Label: {label_type}) ---")
		print("\n--- RUBRIC-ALIGNED FEEDBACK ---\n")
		print(feedback_response)
		print("\n--- ACADEMIC INTEGRITY DETECTION ---\n")
		print(detection_response)


def main() -> None:
	tokenizer, text_gen, eos_token_id = init_pipeline()
	for name in ["psychology.json", "engineering.json"]:
		run_dataset(tokenizer, text_gen, eos_token_id, Path(name))


if __name__ == "__main__":
	main()

