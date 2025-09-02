# Phi-2 Academic Analyzer

An advanced academic submission analysis system powered by Microsoft's Phi-2 model, designed for comprehensive feedback generation and AI detection in educational contexts.

## 🌟 Features

- **Intelligent Feedback Generation**: Provides detailed, rubric-aligned feedback for student submissions
- **AI Detection Analysis**: Sophisticated detection of AI-generated content with confidence scoring
- **Multi-Domain Support**: Handles various academic disciplines (Psychology, Engineering, etc.)
- **Batch Processing**: Efficiently processes multiple submissions with progress tracking
- **Flexible Architecture**: Object-oriented design with configurable parameters
- **Advanced Prompt Engineering**: Optimized prompts for Phi-2 model capabilities

## 🏗️ Architecture

### Core Components

1. **Phi2ModelManager**: Handles model initialization, device optimization, and inference
2. **PromptEngineering**: Advanced prompt templates for different analysis tasks
3. **SubmissionProcessor**: Orchestrates the analysis pipeline for submissions
4. **ResultsFormatter**: Formats and saves analysis results

### Key Differences from TinyLlama Implementation

- **Object-oriented design** with clear separation of concerns
- **Advanced prompt engineering** using Phi-2's chat template format
- **Comprehensive error handling** and logging
- **Configurable parameters** through dataclass configuration
- **Batch processing capabilities** with progress tracking
- **Structured output formatting** with detailed analysis sections

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA (optional, for GPU acceleration)

## 🚀 Quick Start

### Option 1: Automatic Installation and Execution
```bash
python install_and_run.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the analyzer
python phi2_academic_analyzer.py
```

## 📊 Input Data Format

The system expects JSON files with the following structure:

```json
{
  "domain": "Psychology",
  "prompt": "Assignment description...",
  "rubric": {
    "rubric_id": "PSYC101_Essay",
    "criteria": [
      {
        "criterion_id": "analysis",
        "name": "Critical Analysis",
        "description": "Evaluation criteria...",
        "performance_descriptors": {
          "excellent": "Demonstrates exceptional...",
          "good": "Shows solid understanding...",
          "average": "Basic comprehension...",
          "needs_improvement": "Limited analysis...",
          "poor": "Minimal understanding..."
        }
      }
    ]
  },
  "submissions": [
    {
      "final_submission": "Student submission text...",
      "label_type": "Human"
    }
  ],
  "few_shots": [
    {
      "final_submission": "Example submission...",
      "label_type": "AI"
    }
  ]
}
```

## 🔧 Configuration

Modify the `AnalysisConfig` class to adjust model parameters:

```python
@dataclass
class AnalysisConfig:
    model_name: str = "microsoft/phi-2"
    max_tokens: int = 800
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    batch_size: int = 1
```

## 📈 Output Format

The system generates two types of analysis for each submission:

### 1. Academic Feedback
- Strengths identification
- Development areas
- Rubric alignment assessment
- Actionable improvement steps
- Overall evaluation summary

### 2. AI Detection Analysis
- Classification (Human/AI/Hybrid)
- Confidence level
- Key indicators with evidence
- Detailed reasoning

## 🎯 Use Cases

- **Educational Institutions**: Automated feedback for large-scale assessments
- **Academic Integrity**: Detection of AI-generated submissions
- **Writing Centers**: Consistent, detailed feedback for student writers
- **Research**: Analysis of writing patterns and authenticity

## 🔍 Technical Details

### Model Optimization
- Automatic device detection (GPU/CPU)
- Memory-efficient loading with `low_cpu_mem_usage=True`
- Half-precision inference on GPU for faster processing
- Proper padding token configuration for batch processing

### Prompt Engineering
- Uses Phi-2's chat template format (`<|im_start|>` and `<|im_end|>`)
- Context-aware prompts with role-based instructions
- Few-shot learning integration for improved accuracy
- Structured output formatting for consistent results

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions or issues, please refer to the documentation or create an issue in the repository.