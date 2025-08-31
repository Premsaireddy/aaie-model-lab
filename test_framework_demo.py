#!/usr/bin/env python3
"""
Demo script to test the framework functionality without loading the full Phi2 model
"""

import json
import os
import sys
from pathlib import Path

# Add the base prompts to path and import
sys.path.append('Model and Prompt Selection/Models/Base Prompts')
from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt

def test_prompt_generation():
    """Test that our prompt generation functions work correctly"""
    print("🧪 Testing prompt generation functions...")
    
    # Load a sample dataset
    dataset_path = Path("Model and Prompt Selection/Models/Training Data/accounting.json")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    domain = data['domain']
    prompt = data['prompt']
    rubric = data['rubric']
    submissions = data['submissions']
    
    print(f"✅ Loaded {domain} dataset with {len(submissions)} submissions")
    
    # Test feedback prompt generation
    print("\n📝 Testing feedback prompt generation...")
    
    # Build rubric text
    rubric_text = f"Rubric ID: {rubric['rubric_id']}\n\nCriteria:\n"
    for criterion in rubric['criteria']:
        rubric_text += f"\n{criterion['criterion_id']}: {criterion['name']}\n"
        rubric_text += f"Description: {criterion['description']}\n"
        rubric_text += "Performance Levels:\n"
        for level, desc in criterion['performance_descriptors'].items():
            rubric_text += f"- {level.title()}: {desc}\n"
    
    sample_submission = submissions[0]['final_submission']
    feedback_messages = build_feedback_prompt(domain, prompt, rubric_text, sample_submission)
    
    print(f"✅ Generated feedback prompt with {len(feedback_messages)} messages")
    print(f"   System prompt length: {len(feedback_messages[0]['content'])}")
    print(f"   User prompt length: {len(feedback_messages[1]['content'])}")
    
    # Test AI detection prompt generation
    print("\n🔍 Testing AI detection prompt generation...")
    
    few_shots = submissions[1:4]  # Use some submissions as few-shots
    detection_messages = build_detection_prompt(sample_submission, few_shots)
    
    print(f"✅ Generated detection prompt with {len(detection_messages)} messages")
    print(f"   System prompt length: {len(detection_messages[0]['content'])}")
    print(f"   User prompt length: {len(detection_messages[1]['content'])}")
    
    # Test self-evaluation prompt
    print("\n⭐ Testing self-evaluation prompt generation...")
    
    sample_feedback = '{"overall_summary": "Good analysis", "criteria_feedback": []}'
    eval_prompt = self_eval_prompt(rubric, sample_submission, sample_feedback)
    
    print(f"✅ Generated self-eval prompt with length: {len(eval_prompt)}")
    
    return True

def test_dataset_loading():
    """Test loading all training datasets"""
    print("\n📂 Testing dataset loading...")
    
    dataset_dir = Path("Model and Prompt Selection/Models/Training Data")
    dataset_files = ["accounting.json", "psychology.json", "engineering.json", "it.json", "teaching.json"]
    
    datasets = {}
    
    for file_name in dataset_files:
        file_path = dataset_dir / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    domain = data['domain']
                    datasets[domain] = data
                    submissions_count = len(data['submissions'])
                    criteria_count = len(data['rubric']['criteria'])
                    print(f"✅ {domain}: {submissions_count} submissions, {criteria_count} criteria")
            except Exception as e:
                print(f"❌ Error loading {file_name}: {str(e)}")
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n✅ Successfully loaded {len(datasets)} datasets")
    return len(datasets) == 5

def show_sample_data():
    """Show sample data structure"""
    print("\n📋 Sample data structure:")
    
    dataset_path = Path("Model and Prompt Selection/Models/Training Data/psychology.json")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Domain: {data['domain']}")
    print(f"Prompt: {data['prompt'][:100]}...")
    print(f"Rubric ID: {data['rubric']['rubric_id']}")
    print(f"Criteria count: {len(data['rubric']['criteria'])}")
    print(f"Submissions count: {len(data['submissions'])}")
    
    # Show first criterion
    first_criterion = data['rubric']['criteria'][0]
    print(f"\nSample criterion:")
    print(f"  ID: {first_criterion['criterion_id']}")
    print(f"  Name: {first_criterion['name']}")
    print(f"  Description: {first_criterion['description']}")
    
    # Show first submission
    first_submission = data['submissions'][0]
    print(f"\nSample submission:")
    print(f"  Label: {first_submission['label_type']}")
    print(f"  Content: {first_submission['final_submission'][:150]}...")

def main():
    """Run all demo tests"""
    print("🔬 Phi2 Evaluation Framework Demo")
    print("=" * 50)
    
    # Test dataset loading
    if not test_dataset_loading():
        print("❌ Dataset loading failed")
        return 1
    
    # Test prompt generation
    if not test_prompt_generation():
        print("❌ Prompt generation failed")
        return 1
    
    # Show sample data
    show_sample_data()
    
    print("\n🎉 All framework tests passed!")
    print("\n📝 Next steps:")
    print("   1. Run 'python run_phi2_tests.py' to start full evaluation")
    print("   2. Check /workspace/evaluation_results/ for detailed results")
    print("   3. Review the generated feedback and AI detection performance")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())