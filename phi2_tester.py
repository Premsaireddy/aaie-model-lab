#!/usr/bin/env python3
"""
Simple Phi2 Model Tester - Ready to Run
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any

# Add base prompts to path
sys.path.append('Model and Prompt Selection/Models/Base Prompts')

def load_datasets():
    """Load all training datasets"""
    dataset_dir = Path("Model and Prompt Selection/Models/Training Data")
    datasets = {}
    
    for file_name in ["accounting.json", "psychology.json", "engineering.json", "it.json", "teaching.json"]:
        file_path = dataset_dir / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                datasets[data['domain']] = data
                print(f"✅ Loaded {data['domain']}: {len(data['submissions'])} submissions")
    
    return datasets

def test_with_phi2():
    """Test with actual Phi2 model"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt
        
        print("🤖 Loading Phi2 model...")
        
        # Load model and tokenizer
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
        
        print("✅ Phi2 model loaded successfully")
        
        # Load datasets
        datasets = load_datasets()
        
        # Test one submission from each dataset
        results = []
        
        for domain, data in datasets.items():
            print(f"\n=== Testing {domain} ===")
            
            submission = data['submissions'][0]  # Test first submission
            few_shots = data['submissions'][1:4]  # Use others as examples
            
            print(f"Ground truth: {submission['label_type']}")
            print(f"Submission length: {len(submission['final_submission'])} chars")
            
            # 1. Generate Feedback
            print("📝 Generating feedback...")
            start_time = time.time()
            
            # Build rubric text
            rubric_text = f"Rubric ID: {data['rubric']['rubric_id']}\n\nCriteria:\n"
            for criterion in data['rubric']['criteria']:
                rubric_text += f"\n{criterion['criterion_id']}: {criterion['name']}\n"
                rubric_text += f"Description: {criterion['description']}\n"
                rubric_text += "Performance Levels:\n"
                for level, desc in criterion['performance_descriptors'].items():
                    rubric_text += f"- {level.title()}: {desc}\n"
            
            # Generate feedback
            feedback_messages = build_feedback_prompt(domain, data['prompt'], rubric_text, submission['final_submission'])
            feedback_prompt = f"{feedback_messages[0]['content']}\n\n{feedback_messages[1]['content']}\n\nAssistant: "
            
            inputs = tokenizer(feedback_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            feedback_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            feedback_time = time.time() - start_time
            
            print(f"✅ Feedback generated in {feedback_time:.1f}s")
            print(f"Feedback preview: {feedback_response[:200]}...")
            
            # 2. Rate Feedback
            print("⭐ Rating feedback...")
            start_time = time.time()
            
            rating_prompt = self_eval_prompt(data['rubric'], submission['final_submission'], feedback_response)
            rating_prompt += "\n\nRATING: "
            
            inputs = tokenizer(rating_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            rating_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            rating_time = time.time() - start_time
            
            # Extract rating
            import re
            rating_match = re.search(r'\b([1-5])\b', rating_response)
            rating = float(rating_match.group(1)) if rating_match else 3.0
            
            print(f"✅ Rating: {rating}/5.0 (generated in {rating_time:.1f}s)")
            
            # 3. Detect AI
            print("🔍 Detecting AI...")
            start_time = time.time()
            
            detection_messages = build_detection_prompt(submission['final_submission'], few_shots)
            detection_prompt = f"{detection_messages[0]['content']}\n\n{detection_messages[1]['content']}\n\nAssistant: "
            
            inputs = tokenizer(detection_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            detection_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            detection_time = time.time() - start_time
            
            # Extract label
            predicted_label = "Unknown"
            if "Human" in detection_response:
                predicted_label = "Human"
            elif "AI" in detection_response:
                predicted_label = "AI"  
            elif "Hybrid" in detection_response:
                predicted_label = "Hybrid"
            
            actual_label = submission['label_type']
            correct = predicted_label == actual_label
            
            print(f"✅ Detection: {predicted_label} (actual: {actual_label}) {'✓' if correct else '✗'}")
            print(f"   Generated in {detection_time:.1f}s")
            
            # Store results
            results.append({
                'domain': domain,
                'feedback_rating': rating,
                'predicted_label': predicted_label,
                'actual_label': actual_label,
                'correct_detection': correct,
                'total_time': feedback_time + rating_time + detection_time,
                'feedback_preview': feedback_response[:200]
            })
        
        # Summary
        print(f"\n📊 OVERALL RESULTS")
        print("=" * 40)
        
        avg_rating = sum(r['feedback_rating'] for r in results) / len(results)
        accuracy = sum(r['correct_detection'] for r in results) / len(results)
        avg_time = sum(r['total_time'] for r in results) / len(results)
        
        print(f"Domains tested: {len(results)}")
        print(f"Average feedback rating: {avg_rating:.2f}/5.0")
        print(f"AI detection accuracy: {accuracy:.1%}")
        print(f"Average processing time: {avg_time:.1f}s per submission")
        
        # Save results
        os.makedirs("/workspace/phi2_test_results", exist_ok=True)
        with open("/workspace/phi2_test_results/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to /workspace/phi2_test_results/results.json")
        
        return results
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install --break-system-packages torch transformers")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main execution"""
    print("🚀 Phi2 Model Tester")
    print("=" * 30)
    
    # Check if datasets exist
    if not Path("Model and Prompt Selection/Models/Training Data/accounting.json").exists():
        print("❌ Training datasets not found!")
        return 1
    
    # Run test
    results = test_with_phi2()
    
    if results:
        print("\n🎉 Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())