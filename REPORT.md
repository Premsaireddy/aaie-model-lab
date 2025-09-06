## ZSL/FSL Evaluation Report (ChatGPT-4.1)

- Model: `gpt-4.1`
- Feedback Generation: Zero-Shot Learning (no examples provided by design)
- AI Detection: Few-Shot Learning (uses dataset-provided examples when available)

### Methodology
- Feedback (ZSL): Prompt instructs supportive, rubric-aligned feedback with structured sections. No example prompts are included (intentional for ZSL).
- Detection (FSL): Prompt includes a short set of examples (if available in each dataset) with expected labels, then requests a structured label/rationale/flags output.
- Datasets: Provided as JSON files with `rubric`, `few_shots` (optional), `submissions`.
- Outputs: Two JSON files written to `outputs/` directory: `predictions_feedback.jsonl`, `predictions_detection.jsonl`.

### Notes on Feedback ZSL
- No example prompts were used. This is intentional to evaluate pure zero-shot generalization.
- Prompt emphasizes rubric alignment and specific evidence-based feedback.

### Evaluation Criteria
- Apply the evaluation team's rubric/criteria to both tasks:
  - Feedback: coverage of rubric criteria, specificity of evidence, actionable tips, clarity, tone.
  - Detection: correctness of label (Human/AI/Hybrid), coherence of rationale, correct flag usage.

### Results Summary (to be completed after run)
- Feedback quality ratings: <fill in>
- Detection accuracy/F1: <fill in>

### Observations
- Strengths: <fill in>
- Weaknesses / Failure modes: <fill in>
- Recommendations: <fill in>

### Repro Steps
1. Create `.env` with `OPENAI_API_KEY`.
2. Install deps: `pip install -r requirements.txt`.
3. Run:
   - `python openai_runner.py --data accounting.json engineering.json it.json psychology.json teaching.json --out outputs --model gpt-4.1`
4. Evaluate outputs using the team's criteria. Update this report accordingly.