import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    OpenAI = None  # type: ignore

from openai_prompts import (
    SYSTEM_PROMPT,
    render_chat_messages,
    build_detection_user_block,
    build_feedback_user_block,
    format_rubric,
    extract_detection_fields,
    attempt_json,
)


def ensure_api_key() -> None:
    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or create a .env file with OPENAI_API_KEY=..."
        )


def init_openai_client() -> Any:
    ensure_api_key()
    if OpenAI is None:  # pragma: no cover
        raise RuntimeError("openai package is not available. Please install requirements.")
    return OpenAI()


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def generate_chat_completion(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    # Prefer chat.completions; fallback to responses API
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception:
        pass

    # fallback to responses API
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
        max_output_tokens=max_tokens,
        top_p=top_p,
    )
    # responses API can have output_text helper
    if hasattr(resp, "output_text"):
        return resp.output_text.strip()  # type: ignore[attr-defined]

    # Otherwise, assemble from content parts
    try:
        chunks = []
        for item in (resp.output or []):  # type: ignore[attr-defined]
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(getattr(c, "text", ""))
        return "".join(chunks).strip()
    except Exception:
        return ""


def run_and_export_predictions(
    client: Any,
    model: str,
    data_paths: List[Path],
    output_dir: Path,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_tokens: int = 800,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detect_path = output_dir / "predictions_detection.jsonl"
    feedback_path = output_dir / "predictions_feedback.jsonl"

    all_detection_data: List[Dict[str, Any]] = []
    all_feedback_data: List[Dict[str, Any]] = []

    for data_path in data_paths:
        if not data_path.exists():
            print(f"Warning: {data_path} not found; skipping.")
            continue

        with data_path.open() as f:
            data = json.load(f)

        rubric_text = format_rubric(data.get("rubric", {}))
        few_shots = data.get("few_shots", [])
        domain = data.get("domain", "General")
        assign_prompt = data.get("prompt", "Analyze student submission and provide detailed feedback")

        for i, submission in enumerate(data.get("submissions", []), 1):
            submission_text = submission.get("final_submission", "")
            label_true = submission.get("label_type", "Unknown")

            # Feedback (ZSL: intentionally no examples)
            feedback_user = build_feedback_user_block(
                domain=domain,
                assignment_prompt=assign_prompt,
                rubric_text=rubric_text,
                submission=submission_text,
            )
            feedback_messages = render_chat_messages(SYSTEM_PROMPT, feedback_user)
            feedback_text = generate_chat_completion(
                client,
                feedback_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            feedback_obj = attempt_json(feedback_text)

            all_feedback_data.append({
                "dataset": data_path.name,
                "submission_index": i,
                "feedback_raw": feedback_text,
                "feedback_json": feedback_obj,
            })

            # Detection (FSL: uses few_shots if available)
            detection_user = build_detection_user_block(submission_text, few_shots)
            detection_messages = render_chat_messages(SYSTEM_PROMPT, detection_user)
            detection_text = generate_chat_completion(
                client,
                detection_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            det = extract_detection_fields(detection_text)

            all_detection_data.append({
                "dataset": data_path.name,
                "submission_index": i,
                "label_true": label_true,
                "label_pred": det.get("label"),
                "rationale": det.get("rationale"),
                "flags": det.get("flags"),
                "raw": detection_text,
            })

    with detect_path.open("w", encoding="utf-8") as df:
        json.dump(all_detection_data, df, indent=4, ensure_ascii=False)

    with feedback_path.open("w", encoding="utf-8") as ff:
        json.dump(all_feedback_data, ff, indent=4, ensure_ascii=False)

    print(f"Wrote: {detect_path}")
    print(f"Wrote: {feedback_path}")
    return detect_path, feedback_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI ChatGPT-4.1 ZSL/FSL pipeline")
    parser.add_argument(
        "--data",
        nargs="+",
        type=str,
        required=True,
        help="One or more dataset JSON file paths",
    )
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = init_openai_client()
    data_paths = [Path(p) for p in args.data]
    output_dir = Path(args.out)

    run_and_export_predictions(
        client=client,
        model=args.model,
        data_paths=data_paths,
        output_dir=output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()