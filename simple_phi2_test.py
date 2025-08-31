#!/usr/bin/env python3
"""
Simple Phi2 Test - Minimal Dependencies
"""

import json
import sys
import os
from pathlib import Path

# Add base prompts to path  
sys.path.append('Model and Prompt Selection/Models/Base Prompts')

def simple_test():
    """Run a simple test without heavy dependencies"""
    print("🧪 Simple Phi2 Framework Test")
    print("=" * 40)
    
    try:
        # Test imports
        from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt
        print("✅ Base prompts imported successfully")
        
        # Load one dataset
        with open("Model and Prompt Selection/Models/Training Data/psychology.json", 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {data['domain']} dataset")
        
        # Test prompt generation
        submission = data['submissions'][0]
        few_shots = data['submissions'][1:3]
        
        # Test feedback prompt
        rubric_text = "Sample rubric for testing"
        feedback_messages = build_feedback_prompt(
            data['domain'], 
            data['prompt'], 
            rubric_text, 
            submission['final_submission']
        )
        
        print(f"✅ Feedback prompt generated: {len(feedback_messages)} messages")
        print(f"   System: {len(feedback_messages[0]['content'])} chars")
        print(f"   User: {len(feedback_messages[1]['content'])} chars")
        
        # Test detection prompt
        detection_messages = build_detection_prompt(submission['final_submission'], few_shots)
        
        print(f"✅ Detection prompt generated: {len(detection_messages)} messages")
        print(f"   System: {len(detection_messages[0]['content'])} chars")
        print(f"   User: {len(detection_messages[1]['content'])} chars")
        
        # Test rating prompt
        sample_feedback = '{"overall_summary": "Good work", "criteria_feedback": []}'
        rating_prompt = self_eval_prompt(data['rubric'], submission['final_submission'], sample_feedback)
        
        print(f"✅ Rating prompt generated: {len(rating_prompt)} chars")
        
        print(f"\n🎯 PROMPT EXAMPLES:")
        print(f"Feedback prompt preview:")
        print(f"'{feedback_messages[1]['content'][:200]}...'")
        
        print(f"\nDetection prompt preview:")
        print(f"'{detection_messages[1]['content'][:200]}...'")
        
        print(f"\nRating prompt preview:")
        print(f"'{rating_prompt[:200]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    if simple_test():
        print("\n🎉 Framework test passed!")
        print("\n🚀 To run with actual Phi2 model:")
        print("   python3 phi2_tester.py")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())