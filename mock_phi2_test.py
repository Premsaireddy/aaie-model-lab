#!/usr/bin/env python3
"""
Mock Phi2 test to demonstrate framework functionality without downloading the full model
"""

import json
import time
import random
from pathlib import Path
from base_prompts_helper import build_detection_prompt, build_feedback_prompt, self_eval_prompt

class MockPhi2Model:
    """Mock model that simulates Phi2 responses for testing"""
    
    def __init__(self):
        self.device = "cpu"
    
    def generate_feedback(self, prompt: str) -> str:
        """Generate mock feedback response"""
        # Simulate processing time
        time.sleep(0.5)
        
        # Return a realistic JSON feedback response
        feedback = {
            "overall_summary": "The submission demonstrates a solid understanding of the topic with clear explanations and relevant examples. Some areas could benefit from deeper analysis and more specific evidence.",
            "criteria_feedback": [
                {
                    "criterion_id": "c1",
                    "rating": random.choice(["excellent", "good", "average", "needs_improvement"]),
                    "evidence": ["Clear conceptual understanding demonstrated", "Examples are relevant and well-chosen"],
                    "improvement_tip": "Consider adding more specific technical details"
                },
                {
                    "criterion_id": "c2", 
                    "rating": random.choice(["good", "average", "needs_improvement"]),
                    "evidence": ["Real-world applications mentioned", "Some connections could be stronger"],
                    "improvement_tip": "Provide more detailed explanations of practical implications"
                }
            ],
            "suggested_grade": "B+"
        }
        
        return json.dumps(feedback, indent=2)
    
    def detect_ai(self, prompt: str) -> str:
        """Generate mock AI detection response"""
        time.sleep(0.3)
        
        # Randomly assign labels for demonstration
        labels = ["Human", "AI", "Hybrid"]
        label = random.choice(labels)
        
        detection_result = {
            "label": label,
            "rationale": [
                "Writing style analysis indicates formal academic tone",
                "Vocabulary usage suggests technical familiarity",
                "Structure follows typical essay format"
            ],
            "flags": random.choice([["none"], ["generic_phrasing"], ["style_inconsistency"]])
        }
        
        return json.dumps(detection_result, indent=2)
    
    def rate_feedback(self, prompt: str) -> str:
        """Generate mock feedback rating"""
        time.sleep(0.2)
        return str(random.randint(3, 5))

def run_mock_evaluation():
    """Run mock evaluation to demonstrate framework"""
    print("🎭 Mock Phi2 Evaluation (Demo Mode)")
    print("=" * 50)
    
    # Load datasets
    dataset_dir = Path("Model and Prompt Selection/Models/Training Data")
    datasets = {}
    
    for file_name in ["accounting.json", "psychology.json", "engineering.json", "it.json", "teaching.json"]:
        file_path = dataset_dir / file_name
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                datasets[data['domain']] = data
    
    print(f"📚 Loaded {len(datasets)} datasets")
    
    # Initialize mock model
    mock_model = MockPhi2Model()
    print("🤖 Mock Phi2 model initialized")
    
    # Run evaluation on a subset
    all_results = []
    
    for domain, dataset in datasets.items():
        print(f"\n=== Testing {domain} ===")
        
        submissions = dataset['submissions'][:2]  # Test first 2 submissions per domain
        few_shots = dataset['submissions'][2:5]
        
        for i, submission in enumerate(submissions):
            print(f"  Processing submission {i+1}/{len(submissions)}...")
            start_time = time.time()
            
            # Generate feedback
            rubric_text = f"Rubric ID: {dataset['rubric']['rubric_id']}\n"
            feedback_messages = build_feedback_prompt(domain, dataset['prompt'], rubric_text, submission['final_submission'])
            feedback_prompt = f"{feedback_messages[0]['content']}\n\n{feedback_messages[1]['content']}"
            
            generated_feedback = mock_model.generate_feedback(feedback_prompt)
            
            # Rate feedback
            rating_prompt = self_eval_prompt(dataset['rubric'], submission['final_submission'], generated_feedback)
            rating_response = mock_model.rate_feedback(rating_prompt)
            feedback_rating = float(rating_response)
            
            # Detect AI
            detection_messages = build_detection_prompt(submission['final_submission'], few_shots)
            detection_prompt = f"{detection_messages[0]['content']}\n\n{detection_messages[1]['content']}"
            detection_response = mock_model.detect_ai(detection_prompt)
            
            try:
                ai_detection_result = json.loads(detection_response)
            except:
                ai_detection_result = {"label": "Unknown", "rationale": ["Parse error"], "flags": ["error"]}
            
            processing_time = time.time() - start_time
            
            # Store results
            result = {
                'domain': domain,
                'submission_id': len(all_results),
                'generated_feedback': generated_feedback,
                'feedback_rating': feedback_rating,
                'ai_detection_result': ai_detection_result,
                'ground_truth_label': submission['label_type'],
                'processing_time': processing_time,
                'errors': []
            }
            
            all_results.append(result)
            
            # Show quick summary
            predicted_label = ai_detection_result.get('label', 'Unknown')
            actual_label = submission['label_type']
            correct = predicted_label == actual_label
            
            print(f"    ✅ Feedback: {feedback_rating:.1f}/5.0")
            print(f"    🔍 AI Detection: {predicted_label} (actual: {actual_label}) {'✓' if correct else '✗'}")
            print(f"    ⏱️  Time: {processing_time:.2f}s")
    
    # Calculate overall metrics
    print(f"\n📊 MOCK EVALUATION RESULTS")
    print("=" * 40)
    
    total_submissions = len(all_results)
    avg_rating = sum(r['feedback_rating'] for r in all_results) / total_submissions
    
    correct_detections = sum(1 for r in all_results 
                           if r['ai_detection_result'].get('label') == r['ground_truth_label'])
    detection_accuracy = correct_detections / total_submissions
    
    avg_time = sum(r['processing_time'] for r in all_results) / total_submissions
    
    print(f"Total submissions tested: {total_submissions}")
    print(f"Average feedback rating: {avg_rating:.2f}/5.0")
    print(f"AI detection accuracy: {detection_accuracy:.1%}")
    print(f"Average processing time: {avg_time:.2f}s")
    
    # Save mock results
    output_dir = "/workspace/mock_test_results"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/mock_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Mock results saved to {output_dir}/mock_results.json")
    print("\n🎉 Mock evaluation completed successfully!")
    print("\n📝 This demonstrates the framework functionality.")
    print("   To run with the real Phi2 model, use: python3 run_phi2_tests.py")

def main():
    try:
        run_mock_evaluation()
        return 0
    except Exception as e:
        print(f"❌ Error in mock evaluation: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())