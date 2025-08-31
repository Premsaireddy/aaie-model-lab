import argparse
import json
import os
from typing import List, Dict, Any

from model_clients import get_client
from utils import load_dataset, rubric_to_text, parse_json_safely, score_detection, safe_int_score

# Import base_prompts via file path (spaces in directories prevent normal import)
import importlib.util

BASE_PROMPTS_PATH = "/workspace/Model and Prompt Selection/Models/Base Prompts/base_prompts.py"
spec = importlib.util.spec_from_file_location("base_prompts", BASE_PROMPTS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load base_prompts from {BASE_PROMPTS_PATH}")
base_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_prompts)  # type: ignore
build_detection_prompt = base_prompts.build_detection_prompt
build_feedback_prompt = base_prompts.build_feedback_prompt
self_eval_prompt = base_prompts.self_eval_prompt


def render_chat_to_text(messages: List[Dict[str, str]]) -> str:
    # Simple concatenation for decoder-only models
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"<{role}>\n{m.get('content','').strip()}\n")
    parts.append("<ASSISTANT>\n")
    return "\n".join(parts)


def run_detection(client, submission: str, few_shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    messages = build_detection_prompt(submission=submission, few_shots=few_shots)
    prompt = render_chat_to_text(messages)
    raw = client.generate(prompt, max_new_tokens=256, temperature=0.2)
    ok, obj = parse_json_safely(raw)
    if not ok or not isinstance(obj, dict):
        return {"label": "Hybrid", "rationale": ["parse_error"], "flags": ["none"], "_raw": raw}
    return obj


def run_feedback(client, domain: str, assignment_prompt: str, rubric_text: str, submission: str) -> Dict[str, Any]:
    messages = build_feedback_prompt(domain=domain, assignment_prompt=assignment_prompt, rubric_text=rubric_text, submission=submission)
    prompt = render_chat_to_text(messages)
    raw = client.generate(prompt, max_new_tokens=512, temperature=0.3)
    ok, obj = parse_json_safely(raw)
    if not ok or not isinstance(obj, dict):
        return {
            "overall_summary": "Parsing failed.",
            "criteria_feedback": [],
            "suggested_grade": "",
            "_raw": raw,
        }
    return obj


def run_self_eval(client, rubric: Dict[str, Any], essay: str, feedback_obj: Dict[str, Any]) -> int:
    prompt = self_eval_prompt(rubric=rubric, essay=essay, feedback=json.dumps(feedback_obj))
    raw = client.generate(prompt, max_new_tokens=8, temperature=0.0)
    return safe_int_score(raw)


def build_few_shots(submissions: List[Dict[str, Any]], k: int = 2) -> List[Dict[str, Any]]:
    return submissions[:k] if submissions else []


def evaluate_dataset(client, dataset_path: str, output_path: str) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    domain = data.get("domain", "")
    assignment_prompt = data.get("prompt", "")
    rubric = data.get("rubric", {})
    rubric_text = rubric_to_text(rubric)
    submissions = data.get("submissions", [])

    # Few-shots from first two entries
    shots = build_few_shots(submissions, k=2)

    rows: List[Dict[str, Any]] = []
    correct = 0
    for idx, s in enumerate(submissions):
        essay = s.get("final_submission", "")
        true_label = s.get("label_type", "")

        det = run_detection(client, submission=essay, few_shots=shots)
        fb = run_feedback(client, domain=domain, assignment_prompt=assignment_prompt, rubric_text=rubric_text, submission=essay)
        self_score = run_self_eval(client, rubric=rubric, essay=essay, feedback_obj=fb)

        pred_label = str(det.get("label", "Hybrid"))
        correct += score_detection(pred_label, true_label)

        rows.append({
            "dataset": os.path.basename(dataset_path),
            "idx": idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "detection": det,
            "feedback": fb,
            "self_eval_score": self_score,
        })

    acc = correct / max(1, len(submissions))
    summary = {"dataset": os.path.basename(dataset_path), "num": len(submissions), "accuracy": acc}

    # Write outputs
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phi-2 on rubric feedback and AI detection")
    parser.add_argument("--backend", default="mock", choices=["mock", "hf", "transformers", "ollama"], help="Model backend")
    parser.add_argument("--model_id", default="microsoft/phi-2", help="Model id/name for backend")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of submissions per dataset (0=all)")
    parser.add_argument("--outdir", default="/workspace/Evaluation/results", help="Directory to write JSONL outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    datasets = [
        "/workspace/Model and Prompt Selection/Models/Training Data/accounting.json",
        "/workspace/Model and Prompt Selection/Models/Training Data/psychology.json",
        "/workspace/Model and Prompt Selection/Models/Training Data/engineering.json",
        "/workspace/Model and Prompt Selection/Models/Training Data/it.json",
        "/workspace/Model and Prompt Selection/Models/Training Data/teaching.json",
    ]

    # Optionally limit submissions for quick check by trimming files in-memory
    summaries = []
    for ds in datasets:
        # Quick trim if --limit
        if args.limit and args.limit > 0:
            data = load_dataset(ds)
            data["submissions"] = data.get("submissions", [])[: args.limit]
            tmp_path = os.path.join(args.outdir, f"tmp_{os.path.basename(ds)}")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            use_path = tmp_path
        else:
            use_path = ds
        out_path = os.path.join(args.outdir, os.path.basename(ds) + ".jsonl")
        client = get_client(args.backend, args.model_id)
        summary = evaluate_dataset(client, use_path, out_path)
        summaries.append(summary)

    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

