#!/usr/bin/env python3
"""
Evaluation Summary and Status Checker
"""

import json
import os
from pathlib import Path
from datetime import datetime

def show_framework_overview():
    """Show overview of the evaluation framework"""
    print("🎯 PHI2 MODEL EVALUATION FRAMEWORK")
    print("=" * 60)
    print()
    
    print("📋 FRAMEWORK CAPABILITIES:")
    print("  ✅ Feedback Generation using rubrics")
    print("  ✅ Automated feedback quality rating")
    print("  ✅ AI detection in submissions")
    print("  ✅ Multi-domain evaluation (5 datasets)")
    print("  ✅ Comprehensive performance metrics")
    print("  ✅ Results analysis and reporting")
    print()
    
    print("📊 TRAINING DATASETS:")
    datasets = [
        ("Accounting", "Blockchain impact analysis", "6 submissions, 4 criteria"),
        ("Psychology", "Cognitive biases evaluation", "6 submissions, 4 criteria"),
        ("Engineering", "Production line setup", "6 submissions, 5 criteria"),
        ("IT", "AI in cybersecurity", "6 submissions, 5 criteria"),
        ("Teaching", "Early literacy research", "6 submissions, 5 criteria")
    ]
    
    for domain, topic, stats in datasets:
        print(f"  📚 {domain:<20} | {topic:<25} | {stats}")
    print()
    
    print("🔧 EVALUATION COMPONENTS:")
    print("  1. Rubric-based feedback generation")
    print("  2. Self-evaluation rating system (1-5 scale)")
    print("  3. Few-shot AI detection (Human/AI/Hybrid)")
    print("  4. Performance and error tracking")
    print("  5. Cross-domain analysis")
    print()

def check_available_tests():
    """Show available test options"""
    print("🚀 AVAILABLE TEST OPTIONS:")
    print("-" * 40)
    
    tests = [
        ("test_framework_demo.py", "Framework functionality test", "No model download", "30s"),
        ("mock_phi2_test.py", "Mock evaluation demo", "Simulated responses", "1min"),
        ("quick_phi2_test.py", "Quick real model test", "1 dataset, 2 submissions", "5-10min"),
        ("run_phi2_tests.py", "Full comprehensive test", "All 5 datasets", "20-60min")
    ]
    
    for script, description, scope, time_est in tests:
        print(f"📝 {script}")
        print(f"   Description: {description}")
        print(f"   Scope: {scope}")
        print(f"   Est. Time: {time_est}")
        print()

def check_results_status():
    """Check status of evaluation results"""
    print("📁 RESULTS STATUS:")
    print("-" * 40)
    
    result_dirs = [
        ("/workspace/mock_test_results", "Mock Test Results"),
        ("/workspace/quick_test_results", "Quick Test Results"),
        ("/workspace/evaluation_results", "Full Evaluation Results")
    ]
    
    found_results = False
    
    for dir_path, description in result_dirs:
        result_dir = Path(dir_path)
        if result_dir.exists():
            files = list(result_dir.glob("*.json")) + list(result_dir.glob("*.txt")) + list(result_dir.glob("*.csv"))
            if files:
                found_results = True
                print(f"✅ {description}:")
                for file in files:
                    size = file.stat().st_size
                    modified = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"   📄 {file.name} ({size:,} bytes, {modified})")
                print()
            else:
                print(f"📂 {description}: Directory exists but empty")
        else:
            print(f"❌ {description}: Not found")
    
    if not found_results:
        print("💡 No results found. Run an evaluation test first!")
    
    return found_results

def show_next_steps():
    """Show recommended next steps"""
    print("🎯 RECOMMENDED NEXT STEPS:")
    print("-" * 40)
    
    # Check if we have any results
    has_results = check_results_status()
    
    if not has_results:
        print("1. 🎭 Start with demo: python3 mock_phi2_test.py")
        print("2. 🧪 Test framework: python3 test_framework_demo.py")
        print("3. ⚡ Quick test: python3 quick_phi2_test.py")
        print("4. 🚀 Full evaluation: python3 run_phi2_tests.py")
    else:
        print("1. 📊 Analyze results: python3 results_analyzer.py")
        print("2. 🔍 Check progress: python3 check_phi2_progress.py")
        print("3. 📋 View detailed reports in results directories")
        print("4. 🔄 Run additional tests if needed")
    
    print()
    print("📖 For detailed documentation, see: README_PHI2_EVALUATION.md")

def main():
    """Main function"""
    show_framework_overview()
    check_available_tests()
    show_next_steps()
    
    print("🎉 Phi2 Evaluation Framework Ready!")
    print("   Choose a test option above to get started.")

if __name__ == "__main__":
    main()