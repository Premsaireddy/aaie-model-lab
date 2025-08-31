#!/usr/bin/env python3
"""
Quick Phi2 test with a single dataset to verify functionality
"""

import json
import time
import logging
from pathlib import Path
from phi2_evaluation_framework import Phi2EvaluationFramework

def main():
    """Run a quick test with just one dataset"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("⚡ Quick Phi2 Model Test")
    print("=" * 40)
    
    try:
        # Initialize framework
        framework = Phi2EvaluationFramework()
        
        # Load just one dataset for quick testing
        print("📚 Loading Psychology dataset for quick test...")
        dataset_path = Path("Model and Prompt Selection/Models/Training Data/psychology.json")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            framework.training_datasets[data['domain']] = data
        
        print(f"✅ Loaded {data['domain']} dataset with {len(data['submissions'])} submissions")
        
        # Load model
        print("\n🤖 Loading Phi2 model...")
        if not framework.load_model():
            print("❌ Failed to load Phi2 model!")
            return 1
        
        print("✅ Phi2 model loaded successfully")
        
        # Test with just the first 2 submissions
        print("\n🧪 Testing with first 2 submissions...")
        
        domain = data['domain']
        submissions = data['submissions'][:2]  # Test with just 2 submissions
        few_shots = data['submissions'][2:5]  # Use others as few-shots
        
        results = []
        
        for i, submission in enumerate(submissions):
            print(f"\n--- Testing submission {i+1} ---")
            start_time = time.time()
            
            # Test feedback generation
            print("📝 Generating feedback...")
            rubric_text = f"Rubric ID: {data['rubric']['rubric_id']}\n\nCriteria:\n"
            for criterion in data['rubric']['criteria']:
                rubric_text += f"\n{criterion['criterion_id']}: {criterion['name']}\n"
                rubric_text += f"Description: {criterion['description']}\n"
                rubric_text += "Performance Levels:\n"
                for level, desc in criterion['performance_descriptors'].items():
                    rubric_text += f"- {level.title()}: {desc}\n"
            
            feedback = framework.generate_feedback(
                domain, data['prompt'], data['rubric'], submission['final_submission']
            )
            
            print(f"✅ Feedback generated ({len(feedback)} chars)")
            
            # Test feedback rating
            print("⭐ Rating feedback...")
            rating = framework.rate_feedback(
                data['rubric'], submission['final_submission'], feedback
            )
            print(f"✅ Feedback rating: {rating}/5.0")
            
            # Test AI detection
            print("🔍 Detecting AI...")
            detection_result = framework.detect_ai_in_submission(
                submission['final_submission'], few_shots
            )
            predicted_label = detection_result.get('label', 'Unknown')
            actual_label = submission['label_type']
            correct = predicted_label == actual_label
            
            print(f"✅ AI Detection: {predicted_label} (actual: {actual_label}) {'✓' if correct else '✗'}")
            
            processing_time = time.time() - start_time
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            
            results.append({
                'submission_id': i,
                'feedback': feedback,
                'rating': rating,
                'predicted_label': predicted_label,
                'actual_label': actual_label,
                'correct_detection': correct,
                'processing_time': processing_time
            })
        
        # Summary
        print(f"\n📊 QUICK TEST RESULTS")
        print("=" * 30)
        avg_rating = sum(r['rating'] for r in results) / len(results)
        accuracy = sum(r['correct_detection'] for r in results) / len(results)
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"Submissions tested: {len(results)}")
        print(f"Average feedback rating: {avg_rating:.2f}/5.0")
        print(f"AI detection accuracy: {accuracy:.1%}")
        print(f"Average processing time: {avg_time:.2f}s")
        
        # Save quick results
        output_dir = "/workspace/quick_test_results"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/quick_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {output_dir}/quick_results.json")
        print("\n✅ Quick test completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during quick test: {str(e)}")
        logging.error(f"Quick test failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())