#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any


def get_device() -> str:
	# Lazy import to avoid dependency during --validate
	import torch  # type: ignore
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def load_phi2(model_name: str = "microsoft/phi-2", load_in_4bit: bool = True, device_map: str = "auto"):
	# Lazy imports to avoid dependency during --validate
	from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
	import torch  # type: ignore

	kwargs = {}
	if load_in_4bit:
		try:
			kwargs.update(dict(load_in_4bit=True, device_map=device_map))
		except Exception:
			pass

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	try:
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
			**kwargs,
		)
	except Exception:
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
			device_map=device_map,
		)

	model.config.pad_token_id = tokenizer.pad_token_id
	model.eval()
	return tokenizer, model


def sliding_window_perplexity(text: str, tokenizer: Any, model: Any, max_len: int = 2048, stride: int = 1024) -> float:
	# Lazy import to avoid dependency during --validate
	import torch  # type: ignore

	enc = tokenizer(text, return_tensors="pt")
	input_ids = enc["input_ids"].to(model.device)

	nlls = []
	seq_len = input_ids.size(1)

	for i in range(0, seq_len, stride):
		begin = i
		end = min(i + max_len, seq_len)
		trg_len = end - i

		input_ids_slice = input_ids[:, begin:end]
		with torch.no_grad():
			out = model(input_ids_slice, labels=input_ids_slice)
			neg_log_likelihood = out.loss * trg_len
		nlls.append(neg_log_likelihood)

		if end == seq_len:
			break

	nll = torch.stack(nlls).sum()
	ppl = torch.exp(nll / seq_len)
	return float(ppl)


def map_label(raw: str) -> int:
	raw = (raw or "").strip().lower()
	if raw in ("ai", "hybrid"):
		return 1
	return 0


def pick_threshold(perplexities: List[float], y_true: List[int]) -> Tuple[float, Dict[str, float]]:
	# Lazy import to avoid dependency during --validate
	from sklearn.metrics import precision_recall_fscore_support, accuracy_score  # type: ignore

	candidates = sorted(set(perplexities))
	best = {"f1": -1.0}
	best_t = None
	for t in candidates:
		y_pred = [1 if p < t else 0 for p in perplexities]
		p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
		acc = accuracy_score(y_true, y_pred)
		if f1 > best["f1"]:
			best = {"precision": p, "recall": r, "f1": f1, "acc": acc}
			best_t = t
	return best_t, best


def generate_feedback(model: Any, tokenizer: Any, essay: str, rubric: Dict, max_new_tokens: int = 220) -> str:
	criteria_lines = []
	for c in rubric.get("criteria", []):
		criteria_lines.append(f"- {c['name']}: {c['description']}")

	prompt = (
		"You are an academic writing assistant. Read the student's essay and provide constructive, rubric-aligned feedback. "
		"Focus on strengths, specific areas to improve, and one actionable suggestion per criterion. Be concise and professional.\n\n"
		f"RUBRIC ({rubric.get('rubric_id','')} / {rubric.get('domain','')}):\n"
		+ "\n".join(criteria_lines)
		+ "\n\nSTUDENT ESSAY:\n"
		+ essay.strip()
		+ "\n\nFEEDBACK:\n"
	)

	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
	# Lazy import to avoid dependency during --validate
	import torch  # type: ignore
	with torch.no_grad():
		out = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			top_p=0.9,
			temperature=0.7,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
	text = tokenizer.decode(out[0], skip_special_tokens=True)
	return text.split("FEEDBACK:", 1)[-1].strip()


def main():
	parser = argparse.ArgumentParser(description="Phi-2 AI detection and feedback CLI")
	parser.add_argument("--essay", type=str, required=False, help="Path to essay .txt file")
	parser.add_argument("--calibration", type=str, required=False, help="Path to calibration JSON with fields: perplexities, y_true")
	parser.add_argument("--rubric", type=str, required=False, help="Path to rubric JSON (AAIE schema)")
	parser.add_argument("--model", type=str, default="microsoft/phi-2", help="HF model name or path")
	parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
	parser.add_argument("--max-new-tokens", type=int, default=220)
	parser.add_argument("--validate", action="store_true", help="Skip model download; just validate inputs")
	args = parser.parse_args()

	if args.validate:
		missing = []
		if args.essay and not os.path.exists(args.essay):
			missing.append(f"essay not found: {args.essay}")
		if args.calibration and not os.path.exists(args.calibration):
			missing.append(f"calibration not found: {args.calibration}")
		if args.rubric and not os.path.exists(args.rubric):
			missing.append(f"rubric not found: {args.rubric}")
		if missing:
			print("Validation failed:\n" + "\n".join(missing))
			sys.exit(2)
		print("Validation succeeded. All specified files exist.")
		return

	if not args.essay or not args.calibration or not args.rubric:
		print("--essay, --calibration, and --rubric are required unless --validate is used", file=sys.stderr)
		sys.exit(2)

	with open(args.essay, "r", encoding="utf-8") as f:
		essay_text = f.read()
	with open(args.calibration, "r", encoding="utf-8") as f:
		cal = json.load(f)
	with open(args.rubric, "r", encoding="utf-8") as f:
		rubric = json.load(f)

	perplexities = cal.get("perplexities", [])
	y_true = cal.get("y_true", [])
	if not perplexities or not y_true:
		print("Calibration file must include non-empty 'perplexities' and 'y_true' arrays", file=sys.stderr)
		sys.exit(2)

	print("Loading model...", flush=True)
	tokenizer, model = load_phi2(args.model, load_in_4bit=not args.no_4bit)

	print("Computing essay perplexity...", flush=True)
	enessay_ppl = sliding_window_perplexity(essay_text, tokenizer, model)
	print(f"Essay perplexity: {essay_ppl:.2f}")

	threshold, metrics = pick_threshold(perplexities, y_true)
	print(f"Calibrated threshold: {threshold:.2f}")
	print(f"Calibration metrics: {metrics}")

	prediction = 1 if essay_ppl < threshold else 0
	label = "AI/Hybrid" if prediction == 1 else "Human"
	print(f"Prediction: {label}")

	if prediction == 1:
		print("Generating feedback...", flush=True)
		feedback = generate_feedback(model, tokenizer, essay_text, rubric, max_new_tokens=args.max_new_tokens)
		print("\nFeedback:\n" + feedback)
	else:
		print("Essay classified as Human. No feedback generated.")


if __name__ == "__main__":
	main()