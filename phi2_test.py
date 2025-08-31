#!/usr/bin/env python3
"""
Simple Phi2 Test Script
- Generate feedback using rubrics
- Rate feedback quality  
- Detect AI in submissions
"""

import json
import sys
import os
import time

# Add base prompts to path
sys.path.append('Model and Prompt Selection/Models/Base Prompts')
from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt

def run_phi2_test():
    """Run Phi2 model test"""
    
    # Load dependencies
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("🤖 Loading Phi2 model...")
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    print("✅ Model loaded")
    
    # Load dataset
    with open("Model and Prompt Selection/Models/Training Data/psychology.json", 'r') as f:
        data = json.load(f)
    
    submission = data['submissions'][0]
    few_shots = data['submissions'][1:3]
    
    print(f"\n📝 Testing submission: {submission['label_type']}")
    
    # 1. Generate Feedback
    print("\n1️⃣ GENERATING FEEDBACK...")
    rubric_text = f"Rubric: {data['rubric']['rubric_id']}\n"
    for criterion in data['rubric']['criteria']:
        rubric_text += f"{criterion['criterion_id']}: {criterion['name']}\n"
    
    feedback_messages = build_feedback_prompt(data['domain'], data['prompt'], rubric_text, submission['final_submission'])
    prompt = f"{feedback_messages[1]['content']}\n\nResponse: "
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
        
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    feedback = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"✅ Generated feedback:\n{feedback}\n")
    
    # 2. Rate Feedback
    print("2️⃣ RATING FEEDBACK...")
    rating_prompt = self_eval_prompt(data['rubric'], submission['final_submission'], feedback)
    
    inputs = tokenizer(rating_prompt, return_tensors="pt", truncation=True, max_length=1000)
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
        
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=5,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    rating_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    import re
    rating_match = re.search(r'([1-5])', rating_response)
    rating = rating_match.group(1) if rating_match else "3"
    
    print(f"✅ Feedback rating: {rating}/5")
    
    # 3. Detect AI
    print("\n3️⃣ DETECTING AI...")
    detection_messages = build_detection_prompt(submission['final_submission'], few_shots)
    prompt = f"{detection_messages[1]['content']}\n\nAnswer: "
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
        
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    detection_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract prediction
    predicted = "Unknown"
    for label in ["Human", "AI", "Hybrid"]:
        if label in detection_response:
            predicted = label
            break
    
    actual = submission['label_type']
    correct = "✓" if predicted == actual else "✗"
    
    print(f"✅ AI Detection: {predicted} (Actual: {actual}) {correct}")
    print(f"Full response: {detection_response}")
    
    print(f"\n🎉 TEST COMPLETED!")
    print(f"📊 Results: Rating={rating}/5, Detection={predicted}, Correct={correct}")

if __name__ == "__main__":
    try:
        run_phi2_test()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Install dependencies: pip install --break-system-packages torch transformers")