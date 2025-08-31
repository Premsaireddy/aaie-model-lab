# Phi2 Model Evaluation Framework

This framework provides comprehensive testing of the Microsoft Phi2 model for:
1. **Feedback Generation** using rubrics from training datasets
2. **Feedback Quality Rating** using automated assessment
3. **AI Detection** in student submissions

## 🗂️ Project Structure

```
/workspace/
├── Model and Prompt Selection/Models/
│   ├── Base Prompts/base_prompts.py       # Original prompt templates
│   └── Training Data/                     # 5 training datasets
│       ├── accounting.json
│       ├── psychology.json
│       ├── engineering.json
│       ├── it.json
│       └── teaching.json
├── phi2_evaluation_framework.py           # Main evaluation framework
├── base_prompts_helper.py                 # Helper for prompt functions
├── run_phi2_tests.py                      # Full evaluation runner
├── quick_phi2_test.py                     # Quick test with subset
├── mock_phi2_test.py                      # Demo without model download
├── results_analyzer.py                    # Results analysis dashboard
├── test_framework_demo.py                 # Framework functionality demo
└── requirements.txt                       # Python dependencies
```

## 📊 Training Datasets

The framework uses 5 domain-specific datasets:

| Domain | Submissions | Criteria | Assignment Focus |
|--------|-------------|----------|------------------|
| **Accounting** | 6 | 4 | Blockchain impact on accounting practices |
| **Psychology** | 6 | 4 | Cognitive biases in decision-making |
| **Engineering** | 6 | 5 | Manufacturing production line setup |
| **IT** | 6 | 5 | AI role in cybersecurity |
| **Teaching** | 6 | 5 | Early literacy development research |

Each dataset contains:
- Domain-specific assignment prompt
- Detailed rubric with performance descriptors
- 6 labeled submissions (Human/AI/Hybrid)

## 🚀 Quick Start

### 1. Demo Mode (No Model Download)
```bash
# Test framework functionality without downloading Phi2
python3 test_framework_demo.py

# Run mock evaluation with simulated responses
python3 mock_phi2_test.py

# Analyze mock results
python3 results_analyzer.py
```

### 2. Quick Test (Single Dataset)
```bash
# Test with real Phi2 model on Psychology dataset only
python3 quick_phi2_test.py
```

### 3. Full Evaluation (All Datasets)
```bash
# Run comprehensive evaluation across all 5 datasets
python3 run_phi2_tests.py
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup
```bash
# Install dependencies
pip install --break-system-packages torch transformers numpy pandas accelerate

# Or use requirements file
pip install --break-system-packages -r requirements.txt
```

## 📈 Evaluation Metrics

### Feedback Generation Quality
- **Rating Scale**: 1-5 (using self-evaluation prompt)
- **Metrics**: Average rating, distribution, domain-specific performance
- **JSON Schema Validation**: Checks for proper structure

### AI Detection Performance
- **Labels**: Human, AI, Hybrid
- **Metrics**: Overall accuracy, per-label precision/recall, confusion matrix
- **Few-shot Learning**: Uses other submissions as examples

### Performance Metrics
- **Processing Time**: Per submission and overall
- **Error Rate**: Failed generations/detections
- **Efficiency**: Submissions per second

## 📋 Core Functionality

### 1. Feedback Generation (`build_feedback_prompt`)
```python
# Generates structured feedback using rubric criteria
{
  "overall_summary": "2–4 sentence overview",
  "criteria_feedback": [
    {
      "criterion_id": "c1",
      "rating": "excellent|good|average|needs_improvement|poor",
      "evidence": ["bullet", "points"],
      "improvement_tip": "concrete next step"
    }
  ],
  "suggested_grade": "optional grade"
}
```

### 2. AI Detection (`build_detection_prompt`)
```python
# Classifies submissions as Human/AI/Hybrid
{
  "label": "Human|AI|Hybrid",
  "rationale": ["evidence", "points"],
  "flags": ["style_inconsistency", "generic_phrasing", "none"]
}
```

### 3. Feedback Rating (`self_eval_prompt`)
- Rates feedback quality on 1-5 scale
- Considers rubric alignment and specificity
- Provides numeric score for automated analysis

## 📊 Sample Results

### Mock Evaluation Results
```
📊 MOCK EVALUATION RESULTS
========================================
Total submissions tested: 10
Average feedback rating: 4.10/5.0
AI detection accuracy: 40.0%
Average processing time: 1.00s

Domain-specific Performance:
- Accounting: 4.5/5.0 rating, 50% detection accuracy
- Psychology: 3.5/5.0 rating, 50% detection accuracy  
- Engineering: 3.0/5.0 rating, 50% detection accuracy
- IT: 4.5/5.0 rating, 0% detection accuracy
- Teaching: 5.0/5.0 rating, 50% detection accuracy
```

## 🔍 Key Features

### Rubric-Aligned Feedback
- Uses domain-specific rubrics with detailed criteria
- Generates structured JSON feedback
- Provides evidence-based ratings and improvement tips

### Multi-Class AI Detection
- Distinguishes between Human, AI, and Hybrid submissions
- Uses few-shot learning with submission examples
- Provides rationale and flags for decisions

### Comprehensive Evaluation
- Tests across 5 different academic domains
- Measures both quality and accuracy metrics
- Tracks processing performance and errors

### Flexible Testing Options
- **Demo Mode**: Framework testing without model download
- **Quick Test**: Single dataset evaluation
- **Full Evaluation**: All datasets with complete analysis

## 📁 Output Files

Results are saved to `/workspace/evaluation_results/` (or respective test directories):

- `detailed_results.json`: Complete evaluation data
- `evaluation_results.csv`: Tabular data for analysis
- `evaluation_report.txt`: Human-readable summary
- `summary_report.txt`: Key metrics overview

## 🎯 Use Cases

1. **Model Performance Assessment**: Evaluate Phi2's capability for educational feedback
2. **Rubric Validation**: Test rubric effectiveness across domains
3. **AI Detection Research**: Study patterns in AI vs human writing
4. **Educational Technology**: Automated assessment tool development

## 🔧 Customization

### Adding New Datasets
1. Create JSON file following the existing format
2. Include domain, prompt, rubric, and labeled submissions
3. Place in `Model and Prompt Selection/Models/Training Data/`

### Modifying Evaluation Criteria
- Edit rubric criteria in dataset JSON files
- Adjust rating scales in `self_eval_prompt`
- Customize feedback schema in `build_feedback_prompt`

### Model Configuration
- Change model in `Phi2EvaluationFramework.__init__()`
- Adjust generation parameters (temperature, max_tokens)
- Modify device settings for CPU/GPU usage

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Model Download Fails**
   ```bash
   # Test with mock mode first
   python3 mock_phi2_test.py
   ```

3. **Import Errors**
   ```bash
   # Verify dependencies
   python3 test_framework_demo.py
   ```

### Performance Tips
- Use GPU for faster inference
- Reduce max_new_tokens for quicker generation
- Test with subset before full evaluation

## 📚 Technical Details

### Model: Microsoft Phi2
- **Size**: 2.7B parameters
- **Type**: Causal language model
- **Strengths**: Reasoning, code generation, academic tasks
- **License**: MIT

### Prompt Engineering
- **System Role**: "Careful academic assistant"
- **Output Format**: Strict JSON schema
- **Few-shot Learning**: Uses submission examples
- **Chain of Thought**: Encourages reasoning

### Evaluation Methodology
- **Cross-validation**: Uses other submissions as few-shots
- **Multiple Metrics**: Quality, accuracy, performance
- **Error Handling**: Graceful degradation with logging

## 🎉 Next Steps

1. **Run Full Evaluation**: `python3 run_phi2_tests.py`
2. **Analyze Results**: `python3 results_analyzer.py`
3. **Compare Models**: Extend framework for other models
4. **Fine-tune**: Use results to improve prompts/model

---

*Built for comprehensive evaluation of AI models in educational assessment tasks.*