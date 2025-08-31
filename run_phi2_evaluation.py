#!/usr/bin/env python3
"""
READY-TO-RUN PHI2 EVALUATION SCRIPT
===================================

This script tests Phi2 model for:
1. Feedback generation using rubrics
2. Feedback quality rating  
3. AI detection in submissions

Usage: python3 run_phi2_evaluation.py
"""

import json
import sys
import os
import time
from pathlib import Path

# Add base prompts to Python path
sys.path.append('Model and Prompt Selection/Models/Base Prompts')

def main():
    print("🚀 PHI2 MODEL EVALUATION")
    print("=" * 50)
    
    # Step 1: Check dependencies
    print("1️⃣ Checking dependencies...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install --break-system-packages torch transformers")
        return 1
    
    # Step 2: Load datasets
    print("\n2️⃣ Loading training datasets...")
    datasets = {}
    dataset_files = ["accounting.json", "psychology.json", "engineering.json", "it.json", "teaching.json"]
    
    for file_name in dataset_files:
        file_path = Path("Model and Prompt Selection/Models/Training Data") / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                datasets[data['domain']] = data
                print(f"   ✅ {data['domain']}: {len(data['submissions'])} submissions")
    
    if not datasets:
        print("❌ No datasets found!")
        return 1
    
    # Step 3: Load Phi2 model
    print("\n3️⃣ Loading Phi2 model...")
    try:
        model_name = "microsoft/phi-2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        )
        
        print("   ✅ Phi2 model loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return 1
    
    # Step 4: Run evaluation
    print("\n4️⃣ Running evaluation...")
    all_results = []
    
    for domain, data in datasets.items():
        print(f"\n--- Testing {domain} ---")
        
        # Test first submission from each domain
        submission = data['submissions'][0]
        few_shots = data['submissions'][1:4]
        
        print(f"Testing submission (Ground truth: {submission['label_type']})")
        
        # A. Generate Feedback
        print("  📝 Generating feedback...")
        
        # Build rubric text
        rubric_text = f"Rubric: {data['rubric']['rubric_id']}\n"
        for criterion in data['rubric']['criteria']:
            rubric_text += f"{criterion['criterion_id']}: {criterion['name']}\n"
        
        feedback_messages = build_feedback_prompt(domain, data['prompt'], rubric_text, submission['final_submission'])
        feedback_prompt = f"{feedback_messages[1]['content']}\n\nResponse: "
        
        # Generate
        inputs = tokenizer(feedback_prompt, return_tensors="pt", truncation=True, max_length=1500)
        if device == "cuda":
            inputs = inputs.to(model.device)
            
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        feedback = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"     ✅ Generated ({len(feedback)} chars)")
        
        # B. Rate Feedback  
        print("  ⭐ Rating feedback...")
        
        rating_prompt = f"Rate this feedback 1-5: {feedback[:200]}...\nRating: "
        inputs = tokenizer(rating_prompt, return_tensors="pt", truncation=True, max_length=500)
        if device == "cuda":
            inputs = inputs.to(model.device)
            
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=3,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        rating_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract rating
        import re
        rating_match = re.search(r'([1-5])', rating_response)
        rating = float(rating_match.group(1)) if rating_match else 3.0
        print(f"     ✅ Rating: {rating}/5.0")
        
        # C. Detect AI
        print("  🔍 Detecting AI...")
        
        detection_messages = build_detection_prompt(submission['final_submission'], few_shots)
        detection_prompt = f"{detection_messages[1]['content']}\n\nClassification: "
        
        inputs = tokenizer(detection_prompt, return_tensors="pt", truncation=True, max_length=1500)
        if device == "cuda":
            inputs = inputs.to(model.device)
            
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        detection_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract prediction
        predicted_label = "Unknown"
        for label in ["Human", "AI", "Hybrid"]:
            if label in detection_response:
                predicted_label = label
                break
        
        actual_label = submission['label_type']
        correct = predicted_label == actual_label
        
        print(f"     ✅ Predicted: {predicted_label} (Actual: {actual_label}) {'✓' if correct else '✗'}")
        
        # Store result
        result = {
            'domain': domain,
            'feedback_rating': rating,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'correct_detection': correct,
            'feedback_preview': feedback[:300]
        }
        all_results.append(result)
    
    # Step 5: Show results
    print(f"\n5️⃣ FINAL RESULTS")
    print("=" * 30)
    
    avg_rating = sum(r['feedback_rating'] for r in all_results) / len(all_results)
    accuracy = sum(r['correct_detection'] for r in all_results) / len(all_results)
    
    print(f"📊 Summary:")
    print(f"   Domains tested: {len(all_results)}")
    print(f"   Average feedback rating: {avg_rating:.2f}/5.0")
    print(f"   AI detection accuracy: {accuracy:.1%}")
    
    print(f"\n📋 Detailed Results:")
    for result in all_results:
        status = "✓" if result['correct_detection'] else "✗"
        print(f"   {result['domain']:<20} | Rating: {result['feedback_rating']:.1f}/5.0 | Detection: {status}")
    
    # Save results
    os.makedirs("/workspace/phi2_results", exist_ok=True)
    with open("/workspace/phi2_results/evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to /workspace/phi2_results/evaluation_results.json")
    print("🎉 Evaluation completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())