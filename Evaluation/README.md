Phi-2 Evaluation: Rubric Feedback + AI Detection
================================================

This CLI evaluates a model (default: Phi-2) on two tasks using the provided datasets:
- Generate rubric-aligned feedback for each submission
- Detect AI/Human/Hybrid authorship

It also self-rates the generated feedback against the rubric (1–5 scale).

Install
-------

Create a virtual environment and install requirements:

```
python -m venv .venv
. .venv/bin/activate
pip install -r /workspace/Evaluation/requirements.txt
```

Backends
--------

- mock: No external deps; fast pipeline validation
- hf: Hugging Face Inference API (set `HF_API_TOKEN`)
- transformers: Local transformers (downloads weights; requires sufficient RAM/GPU)
- ollama: Local Ollama server (`ollama serve`; model e.g. `phi:2`)

Run
---

Quick dry run (mock backend):

```
python /workspace/Evaluation/run_phi2_eval.py --backend mock --limit 2
```

Using Hugging Face Inference API:

```
export HF_API_TOKEN=YOUR_TOKEN
python /workspace/Evaluation/run_phi2_eval.py --backend hf --model_id microsoft/phi-2
```

Outputs
-------

Writes one JSONL per dataset to `/workspace/Evaluation/results/`, with fields:
- dataset, idx, true_label, pred_label
- detection (model JSON)
- feedback (model JSON)
- self_eval_score (1–5)

At the end, prints a summary with detection accuracy per dataset.

