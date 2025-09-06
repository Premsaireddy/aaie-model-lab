# model-lab
Central repository for the Model Training Team, focused on building, training, and evaluating AI/ML models. Includes model development scripts, training experiments, evaluation results, and related documentation.

## OpenAI ChatGPT-4.1 ZSL/FSL Pipeline

This repo includes a runner that performs:
- Zero-Shot Learning (ZSL) for Feedback Generation (no example prompts)
- Few-Shot Learning (FSL) for AI Detection (optional examples provided per dataset)

### Setup
1. Create a `.env` (copy `.env.example`) and set `OPENAI_API_KEY`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run
Provide your dataset json files (must contain `rubric`, `submissions`, optional `few_shots`).
```bash
python openai_runner.py --data accounting.json engineering.json it.json psychology.json teaching.json --out outputs --model gpt-4.1
```

Outputs are written to `outputs/`:
- `predictions_detection.jsonl`
- `predictions_feedback.jsonl`

### Evaluate
```bash
python evaluate_outputs.py --detect outputs/predictions_detection.jsonl --feedback outputs/predictions_feedback.jsonl --out outputs/eval.json
```

See `REPORT.md` for the methodology and a template to record results and observations (includes note about unavailable examples for ZSL feedback).
