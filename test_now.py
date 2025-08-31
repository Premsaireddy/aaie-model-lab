#!/usr/bin/env python3
"""
IMMEDIATE TEST - Run this now!
"""

import json
import sys
import os

# Add base prompts to path
sys.path.append('Model and Prompt Selection/Models/Base Prompts')

def immediate_test():
    """Test that can run immediately"""
    print("⚡ IMMEDIATE PHI2 FRAMEWORK TEST")
    print("=" * 40)
    
    # Test 1: Check files exist
    print("1. Checking files...")
    required_files = [
        "Model and Prompt Selection/Models/Base Prompts/base_prompts.py",
        "Model and Prompt Selection/Models/Training Data/accounting.json",
        "Model and Prompt Selection/Models/Training Data/psychology.json"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            return False
    
    # Test 2: Import base prompts
    print("\n2. Testing imports...")
    try:
        from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt
        print("   ✅ Base prompts imported")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test 3: Load dataset
    print("\n3. Loading dataset...")
    try:
        with open("Model and Prompt Selection/Models/Training Data/psychology.json", 'r') as f:
            data = json.load(f)
        print(f"   ✅ Loaded {data['domain']} with {len(data['submissions'])} submissions")
    except Exception as e:
        print(f"   ❌ Dataset error: {e}")
        return False
    
    # Test 4: Generate prompts
    print("\n4. Testing prompt generation...")
    try:
        submission = data['submissions'][0]['final_submission']
        few_shots = data['submissions'][1:3]
        
        # Test feedback prompt
        feedback_messages = build_feedback_prompt(
            data['domain'], 
            data['prompt'], 
            "Sample rubric", 
            submission
        )
        print(f"   ✅ Feedback prompt: {len(feedback_messages)} messages")
        
        # Test detection prompt
        detection_messages = build_detection_prompt(submission, few_shots)
        print(f"   ✅ Detection prompt: {len(detection_messages)} messages")
        
        # Test rating prompt
        rating_prompt = self_eval_prompt(data['rubric'], submission, "Sample feedback")
        print(f"   ✅ Rating prompt: {len(rating_prompt)} chars")
        
    except Exception as e:
        print(f"   ❌ Prompt generation error: {e}")
        return False
    
    # Test 5: Show sample prompts
    print("\n5. Sample prompt content:")
    print("   📝 Feedback prompt system message:")
    print(f"      '{feedback_messages[0]['content']}'")
    
    print("\n   🔍 Detection prompt preview:")
    print(f"      '{detection_messages[1]['content'][:300]}...'")
    
    return True

def show_next_steps():
    """Show what to run next"""
    print("\n🎯 NEXT STEPS:")
    print("=" * 20)
    print("Option 1 - Quick test with real model:")
    print("   python3 phi2_tester.py")
    print()
    print("Option 2 - Simple framework test:")
    print("   python3 simple_phi2_test.py")
    print()
    print("Option 3 - Full evaluation:")
    print("   python3 run_phi2_evaluation.py")
    print()
    print("📋 All tests use the original base_prompts.py (no changes to prompts)")

def main():
    if immediate_test():
        print("\n🎉 FRAMEWORK READY!")
        show_next_steps()
        return 0
    else:
        print("\n❌ FRAMEWORK TEST FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())