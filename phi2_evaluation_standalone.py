#!/usr/bin/env python3
"""
PHI2 MODEL EVALUATION - STANDALONE SCRIPT
==========================================

This script tests Phi2 model for:
1. Feedback generation using rubrics
2. Feedback quality rating
3. AI detection in submissions

Requirements:
- pip install torch transformers

Usage:
1. Place this script in your project folder
2. Ensure you have the base_prompts.py and training data files
3. Run: python3 phi2_evaluation_standalone.py
"""

import json
import sys
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any

# Base prompts functions (copied from original base_prompts.py)
SYSTEM_PROMPT = "You are a careful academic assistant. Be precise and return strict JSON."

def build_detection_prompt(submission: str, few_shots: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Academic Integrity Detector Prompt"""
    shot_texts = []
    for s in few_shots:
        shot_texts.append(
            f'Submission: """{s.get("final_submission","")}"""\n'
            f'Your analysis (2–4 bullet points): <analysis>\n'
            f'Label: {s.get("label_type","")}\n'
        )
    examples_block = "\n\n".join(shot_texts) if shot_texts else "/* no examples available */"

    user = f"""
You are an AI text-source classifier for academic integrity.
Decide whether the student submission is Human, AI, or Hybrid (AI-assisted).

Guidelines:
- Consider discourse features (specificity, subjectivity, personal context), style consistency, local/global coherence, repetitiveness, and cliché patterns.
- Hybrid = meaningful human writing with some AI assistance (ideas, phrasing, structure), or explicit admission of mixed use.

Examples:
{examples_block}

Now analyze the NEW submission step by step and return STRICT JSON.
NEW submission:
\"\"\"{submission}\"\"\"\n
Think briefly, then answer only with the JSON object.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

def build_feedback_prompt(domain: str, assignment_prompt: str, rubric_text: str, submission: str) -> List[Dict[str, str]]:
    """Rubric-Aligned Feedback Prompt"""
    user = f"""
You are a supportive assessor. Provide actionable feedback aligned to the rubric.
Return a STRUCTURED report (no extraneous text).

Sections:
1) "overall_summary": 2–4 sentences on strengths and priorities.
2) "criteria_feedback": array of items, one per rubric criterion with fields:
   - "criterion_id"
   - "rating": one of ["excellent","good","average","needs_improvement","poor"]
   - "evidence": 1–3 bullet points citing concrete excerpt(s) or behaviors
   - "improvement_tip": one concrete next step

Context:
- Domain: {domain}
- Assignment prompt: {assignment_prompt}

Rubric (verbatim):
{rubric_text}

Student submission:
\"\"\"{submission}\"\"\"\n

Constraints:
- Be concise but specific. Do not invent rubric fields. If evidence is insufficient, say so.
- Output MUST be valid JSON with the exact top-level keys: overall_summary, criteria_feedback, suggested_grade.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

def self_eval_prompt(rubric: Dict, essay: str, feedback: str) -> str:
    """Self-evaluation prompt for rating feedback"""
    crit = [c.get('name','Criterion') for c in rubric.get('criteria',[])]
    crit_str = ", ".join(crit) if crit else "the rubric"
    return (
        "You are a strict but fair assessor. Rate how well the FEEDBACK addresses the rubric for the ESSAY.\n"
        "Rate on a 1-5 scale (integers only). Provide ONLY the number.\n\n"
        f"ESSAY:\n{essay}\n\nRUBRIC CRITERIA: {crit_str}\n\nFEEDBACK:\n{feedback}\n\nRATING (1-5): "
    )

class Phi2Tester:
    """Simple Phi2 model tester"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if self.cuda_available() else "cpu"
    
    def cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self):
        """Load Phi2 model"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"🤖 Loading Phi2 model (device: {self.device})...")
            
            model_name = "microsoft/phi-2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using Phi2"""
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=temperature > 0
                )
            
            generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated.strip()
            
        except Exception as e:
            return f"Error: {e}"
    
    def generate_feedback(self, domain: str, assignment_prompt: str, rubric: Dict, submission: str) -> str:
        """Generate feedback using rubric"""
        
        # Build rubric text
        rubric_text = f"Rubric ID: {rubric['rubric_id']}\n\nCriteria:\n"
        for criterion in rubric['criteria']:
            rubric_text += f"\n{criterion['criterion_id']}: {criterion['name']}\n"
            rubric_text += f"Description: {criterion['description']}\n"
            rubric_text += "Performance Levels:\n"
            for level, desc in criterion['performance_descriptors'].items():
                rubric_text += f"- {level.title()}: {desc}\n"
        
        # Generate prompt
        messages = build_feedback_prompt(domain, assignment_prompt, rubric_text, submission)
        prompt = f"{messages[1]['content']}\n\nFeedback: "
        
        # Generate response
        feedback = self.generate_text(prompt, max_tokens=300, temperature=0.7)
        return feedback
    
    def rate_feedback(self, rubric: Dict, submission: str, feedback: str) -> float:
        """Rate feedback quality"""
        prompt = self_eval_prompt(rubric, submission, feedback)
        rating_response = self.generate_text(prompt, max_tokens=5, temperature=0.1)
        
        # Extract rating
        rating_match = re.search(r'([1-5])', rating_response)
        return float(rating_match.group(1)) if rating_match else 3.0
    
    def detect_ai(self, submission: str, few_shots: List[Dict]) -> Dict:
        """Detect AI in submission"""
        messages = build_detection_prompt(submission, few_shots)
        prompt = f"{messages[1]['content']}\n\nClassification: "
        
        response = self.generate_text(prompt, max_tokens=100, temperature=0.3)
        
        # Extract label
        predicted_label = "Unknown"
        for label in ["Human", "AI", "Hybrid"]:
            if label in response:
                predicted_label = label
                break
        
        return {
            "predicted_label": predicted_label,
            "response": response
        }

def main():
    """Main function"""
    print("🚀 PHI2 MODEL TEST")
    print("=" * 30)
    
    # Check dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install: pip install torch transformers")
        return 1
    
    # Check files
    base_prompts_path = "Model and Prompt Selection/Models/Base Prompts/base_prompts.py"
    dataset_path = "Model and Prompt Selection/Models/Training Data/psychology.json"
    
    if not os.path.exists(base_prompts_path):
        print(f"❌ File not found: {base_prompts_path}")
        return 1
    
    if not os.path.exists(dataset_path):
        print(f"❌ File not found: {dataset_path}")
        return 1
    
    print("✅ Required files found")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {data['domain']} dataset")
    
    # Initialize tester
    tester = Phi2Tester()
    
    if not tester.load_model():
        return 1
    
    # Test with first submission
    submission = data['submissions'][0]
    few_shots = data['submissions'][1:4]
    
    print(f"\n📝 Testing submission (Ground truth: {submission['label_type']})")
    print(f"Submission preview: {submission['final_submission'][:150]}...")
    
    # 1. Generate Feedback
    print(f"\n1️⃣ GENERATING FEEDBACK...")
    start_time = time.time()
    
    feedback = tester.generate_feedback(
        data['domain'],
        data['prompt'], 
        data['rubric'],
        submission['final_submission']
    )
    
    feedback_time = time.time() - start_time
    print(f"✅ Feedback generated in {feedback_time:.1f}s:")
    print(f"{feedback}")
    
    # 2. Rate Feedback
    print(f"\n2️⃣ RATING FEEDBACK...")
    start_time = time.time()
    
    rating = tester.rate_feedback(
        data['rubric'],
        submission['final_submission'], 
        feedback
    )
    
    rating_time = time.time() - start_time
    print(f"✅ Rating: {rating}/5.0 (generated in {rating_time:.1f}s)")
    
    # 3. Detect AI
    print(f"\n3️⃣ DETECTING AI...")
    start_time = time.time()
    
    detection_result = tester.detect_ai(submission['final_submission'], few_shots)
    
    detection_time = time.time() - start_time
    predicted = detection_result['predicted_label']
    actual = submission['label_type']
    correct = predicted == actual
    
    print(f"✅ Detection: {predicted} (Actual: {actual}) {'✓' if correct else '✗'}")
    print(f"   Generated in {detection_time:.1f}s")
    print(f"   Full response: {detection_result['response']}")
    
    # Summary
    total_time = feedback_time + rating_time + detection_time
    print(f"\n📊 SUMMARY")
    print(f"=" * 20)
    print(f"Domain: {data['domain']}")
    print(f"Feedback Rating: {rating}/5.0")
    print(f"AI Detection: {predicted} ({'Correct' if correct else 'Incorrect'})")
    print(f"Total Time: {total_time:.1f}s")
    
    # Save results
    result = {
        "domain": data['domain'],
        "ground_truth": actual,
        "feedback": feedback,
        "feedback_rating": rating,
        "ai_detection": predicted,
        "detection_correct": correct,
        "processing_time": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs("phi2_results", exist_ok=True)
    with open("phi2_results/test_result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to phi2_results/test_result.json")
    print(f"🎉 Test completed successfully!")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)