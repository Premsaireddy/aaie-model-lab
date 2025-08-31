#!/usr/bin/env python3
"""
Simple test runner for Phi2 model evaluation
"""

import sys
import logging
from phi2_evaluation_framework import Phi2EvaluationFramework

def main():
    """Run the Phi2 evaluation tests"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Phi2 Model Evaluation")
    print("=" * 60)
    
    try:
        # Initialize framework
        framework = Phi2EvaluationFramework()
        
        # Load datasets
        print("📚 Loading training datasets...")
        framework.load_training_datasets()
        
        if not framework.training_datasets:
            print("❌ No training datasets found!")
            return 1
        
        print(f"✅ Loaded {len(framework.training_datasets)} datasets:")
        for domain in framework.training_datasets.keys():
            submissions_count = len(framework.training_datasets[domain]['submissions'])
            print(f"   - {domain}: {submissions_count} submissions")
        
        # Load model
        print("\n🤖 Loading Phi2 model...")
        if not framework.load_model():
            print("❌ Failed to load Phi2 model!")
            return 1
        
        print("✅ Phi2 model loaded successfully")
        
        # Run evaluation
        print("\n🧪 Running comprehensive evaluation...")
        summary_stats = framework.run_comprehensive_evaluation()
        
        # Show results
        print("\n📊 EVALUATION RESULTS")
        print("=" * 40)
        print(f"Total submissions tested: {summary_stats['total_submissions']}")
        print(f"Domains tested: {summary_stats['domains_tested']}")
        print(f"Average feedback rating: {summary_stats['avg_feedback_rating']:.2f}/5.0")
        print(f"AI detection accuracy: {summary_stats['ai_detection_accuracy']:.2%}")
        print(f"Average processing time: {summary_stats['avg_processing_time']:.2f}s")
        
        print("\n📈 Domain-specific results:")
        for domain, stats in summary_stats['domain_results'].items():
            print(f"  {domain}:")
            print(f"    - Submissions: {stats['submissions_count']}")
            print(f"    - Avg feedback rating: {stats['avg_feedback_rating']:.2f}/5.0")
            print(f"    - AI detection accuracy: {stats['ai_detection_accuracy']:.2%}")
            print(f"    - Avg processing time: {stats['avg_processing_time']:.2f}s")
        
        # Generate and save detailed report
        print("\n💾 Generating detailed report...")
        framework.save_results()
        
        print("\n✅ Evaluation completed successfully!")
        print("📁 Results saved to: /workspace/evaluation_results/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())