# Zero-Shot and Few-Shot Learning with OpenAI GPT-4

## Overview

This project implements **Zero-Shot Learning (ZSL) for Feedback Generation** and **Few-Shot Learning (FSL) for AI Detection** using OpenAI's GPT-4 (ChatGPT-4.1) model. The system evaluates outputs according to established team-defined criteria and provides comprehensive reporting and visualization.

## Key Features

### 1. Zero-Shot Learning (ZSL) for Feedback Generation
- Generates academic feedback **without any example prompts** (true zero-shot)
- Aligns feedback with provided rubrics
- Structured output format for consistency
- Comprehensive evaluation metrics

### 2. Few-Shot Learning (FSL) for AI Detection
- Detects AI-generated content using small example sets
- Classifies submissions as Human, AI, or Hybrid
- Provides confidence levels and detailed rationale
- Extracts specific indicators and evidence

### 3. Evaluation Framework
- **Team-defined criteria** for both tasks
- Automated scoring and metrics calculation
- Statistical analysis and performance tracking
- Comprehensive reporting system

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd zsl-fsl-experiment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### 1. Create Sample Data
```bash
python run_experiment.py create-data
```

This creates sample datasets for testing:
- `accounting.json`
- `engineering.json`
- `it.json`
- `psychology.json`
- `teaching.json`

### 2. Run the Full Experiment
```bash
python run_experiment.py --visualize --dashboard
```

### 3. Analyze Results
```bash
python run_experiment.py analyze outputs/results_*.json --dashboard --excel
```

## Project Structure

```
├── openai_zsl_fsl_pipeline.py  # Main ZSL/FSL implementation
├── visualization_analysis.py    # Analysis and visualization tools
├── config.py                   # Configuration settings
├── run_experiment.py           # Main runner script
├── create_sample_data.py       # Sample data generator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── outputs/                    # Generated outputs
    ├── results_*.json          # Experiment results
    ├── evaluation_report_*.md  # Detailed reports
    └── visualizations/         # Charts and dashboards
```

## Detailed Usage

### Running Experiments

#### Basic Run
```bash
python run_experiment.py
```

#### With Custom Model
```bash
python run_experiment.py --model gpt-4-turbo-preview
```

#### Specific Datasets Only
```bash
python run_experiment.py --datasets accounting.json psychology.json
```

#### Full Analysis Pipeline
```bash
python run_experiment.py --visualize --dashboard --excel
```

### Configuration

Edit `config.py` to customize:
- API settings
- Model selection
- Evaluation weights
- Rate limiting
- Output formats

### Using the Pipeline Directly

```python
from openai_zsl_fsl_pipeline import OpenAIZSLFSLPipeline
from pathlib import Path

# Initialize pipeline
pipeline = OpenAIZSLFSLPipeline(
    api_key="your-api-key",
    model="gpt-4-turbo-preview"
)

# Run experiment
data_paths = [Path("accounting.json")]
results = pipeline.run_experiment(data_paths)

# Generate report
report_path = pipeline.generate_report(results)
```

## Evaluation Criteria

### Feedback Generation (ZSL)
1. **Relevance** (25%): Alignment with submission content
2. **Specificity** (20%): Actionable and detailed suggestions
3. **Alignment** (25%): Coverage of rubric criteria
4. **Constructiveness** (15%): Positive and supportive tone
5. **Completeness** (15%): Comprehensive coverage

### AI Detection (FSL)
1. **Accuracy** (40%): Correct label prediction
2. **Confidence** (20%): Model's self-assessed certainty
3. **Rationale Quality** (25%): Depth of explanation
4. **Flag Relevance** (15%): Specificity of identified indicators

## Important Notes

### Zero-Shot Learning Implementation
- **NO example prompts are provided** for feedback generation (true ZSL)
- Model relies solely on rubric and instructions
- This tests the model's inherent capability without demonstrations

### Few-Shot Learning Implementation
- Uses up to 5 examples per dataset
- Examples demonstrate different authorship patterns
- Includes Human, AI, and Hybrid examples

## Outputs

### 1. Results JSON
Complete experiment data including:
- Raw responses
- Extracted fields
- Evaluation scores
- Metadata

### 2. Evaluation Report (Markdown)
Comprehensive report with:
- Executive summary
- Performance metrics
- Confusion matrices
- Statistical analysis
- Methodology details
- Key observations

### 3. Visualizations
- Distribution plots
- Correlation matrices
- Performance comparisons
- Confusion matrices
- Interactive dashboards

### 4. Excel Export
Structured spreadsheet with:
- All results
- Evaluation scores
- Summary statistics

## API Usage and Costs

The system tracks:
- Total API calls
- Token usage
- Rate limiting
- Cost estimation

Default rate limits:
- 60 requests per minute
- 90,000 tokens per minute

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Rate Limiting**
   ```
   Solution: Adjust RATE_LIMIT_CONFIG in config.py
   ```

3. **Missing Data Files**
   ```
   Solution: Run python run_experiment.py create-data
   ```

## Advanced Features

### Custom Evaluation Weights
Modify `EVALUATION_WEIGHTS` in `config.py`:
```python
EVALUATION_WEIGHTS = {
    "feedback": {
        "relevance": 0.30,  # Increase relevance weight
        "specificity": 0.20,
        # ...
    }
}
```

### Batch Processing
Process multiple result files:
```bash
for file in outputs/results_*.json; do
    python run_experiment.py analyze "$file" --excel
done
```

### API Key Management
Use different API keys:
```bash
python run_experiment.py --api-key sk-different-key
```

## Performance Metrics

Typical performance (GPT-4):
- Feedback Generation: ~0.7-0.8 average score
- AI Detection: ~0.75-0.85 accuracy
- Processing Time: ~2-3 seconds per submission
- Token Usage: ~1000-1500 per evaluation

## Citation

If you use this code in your research, please cite:
```
@software{zsl_fsl_gpt4,
  title = {Zero-Shot and Few-Shot Learning with GPT-4},
  author = {Your Team},
  year = {2024},
  url = {repository-url}
}
```

## License

This project is provided for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration file
3. Examine the log files in `outputs/`

## Acknowledgments

- OpenAI for GPT-4 API access
- Evaluation team for criteria definition
- Contributors and testers

---

**Note**: This implementation specifically addresses the requirement that **no example prompts are available for Zero-Shot Learning** in the feedback generation task, making it a true test of the model's zero-shot capabilities.