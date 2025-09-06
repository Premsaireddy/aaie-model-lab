"""
Configuration file for ZSL/FSL experiments
"""

import os
from pathlib import Path
from typing import Dict, Any

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set your API key here or as environment variable
OPENAI_MODEL = "gpt-4-turbo-preview"  # GPT-4.1 model identifier

# Alternative models for testing
AVAILABLE_MODELS = {
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

# Experiment Settings
EXPERIMENT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1500,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "rate_limit_delay": 1,  # Seconds between API calls
}

# Zero-Shot Learning Settings
ZSL_CONFIG = {
    "feedback_generation": {
        "include_examples": False,  # Must be False for ZSL
        "structured_output": True,
        "rubric_integration": True,
        "max_feedback_length": 1000,
    }
}

# Few-Shot Learning Settings
FSL_CONFIG = {
    "ai_detection": {
        "max_examples": 5,  # Maximum number of examples to use
        "example_selection": "diverse",  # Options: "random", "diverse", "similar"
        "include_confidence": True,
        "include_rationale": True,
        "include_evidence": True,
    }
}

# Evaluation Criteria Weights
EVALUATION_WEIGHTS = {
    "feedback": {
        "relevance": 0.25,
        "specificity": 0.20,
        "alignment": 0.25,
        "constructiveness": 0.15,
        "completeness": 0.15,
    },
    "detection": {
        "accuracy": 0.40,
        "confidence": 0.20,
        "rationale_quality": 0.25,
        "flag_relevance": 0.15,
    }
}

# Data Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Dataset Files
DATASETS = [
    "accounting.json",
    "engineering.json",
    "it.json",
    "psychology.json",
    "teaching.json"
]

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "experiment.log",
    "console_output": True,
}

# Report Settings
REPORT_CONFIG = {
    "include_raw_responses": False,  # Include raw API responses in report
    "include_visualizations": True,
    "include_statistics": True,
    "include_examples": True,
    "max_examples_per_category": 3,
}

# Visualization Settings
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn",
    "color_palette": "husl",
    "save_formats": ["png", "pdf"],
}

# API Rate Limiting
RATE_LIMIT_CONFIG = {
    "max_requests_per_minute": 60,
    "max_tokens_per_minute": 90000,
    "retry_attempts": 3,
    "retry_delay": 5,  # Seconds
}

# Validation Settings
VALIDATION_CONFIG = {
    "min_submission_length": 50,  # Minimum characters for valid submission
    "max_submission_length": 10000,  # Maximum characters
    "required_rubric_fields": ["criteria", "rubric_id"],
    "valid_labels": ["Human", "AI", "Hybrid"],
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "api": {
            "key": OPENAI_API_KEY,
            "model": OPENAI_MODEL,
            "available_models": AVAILABLE_MODELS,
        },
        "experiment": EXPERIMENT_CONFIG,
        "zsl": ZSL_CONFIG,
        "fsl": FSL_CONFIG,
        "evaluation_weights": EVALUATION_WEIGHTS,
        "paths": {
            "data_dir": str(DATA_DIR),
            "output_dir": str(OUTPUT_DIR),
            "visualization_dir": str(VISUALIZATION_DIR),
            "reports_dir": str(REPORTS_DIR),
            "datasets": DATASETS,
        },
        "logging": LOGGING_CONFIG,
        "report": REPORT_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "rate_limit": RATE_LIMIT_CONFIG,
        "validation": VALIDATION_CONFIG,
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check API key
    if not OPENAI_API_KEY:
        errors.append("OpenAI API key not set. Please set OPENAI_API_KEY environment variable or update config.py")
    
    # Check model availability
    if OPENAI_MODEL not in AVAILABLE_MODELS.values():
        errors.append(f"Model {OPENAI_MODEL} not in available models")
    
    # Check data files exist
    for dataset in DATASETS:
        if not Path(dataset).exists() and not (DATA_DIR / dataset).exists():
            errors.append(f"Dataset file {dataset} not found")
    
    # Check evaluation weights sum to 1
    for category, weights in EVALUATION_WEIGHTS.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            errors.append(f"Evaluation weights for {category} don't sum to 1.0 (sum={total})")
    
    return errors

def create_directories():
    """Create necessary directories"""
    directories = [OUTPUT_DIR, VISUALIZATION_DIR, REPORTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration valid!")
    
    # Create directories
    create_directories()
    
    # Display configuration
    import json
    print("\nCurrent Configuration:")
    print(json.dumps(get_config(), indent=2))