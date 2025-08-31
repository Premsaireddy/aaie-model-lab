#!/usr/bin/env python3
"""
Results Analysis Dashboard for Phi2 Model Evaluation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

class ResultsAnalyzer:
    """Analyze and visualize Phi2 evaluation results"""
    
    def __init__(self, results_dir: str = "/workspace/evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_data = None
        self.df = None
        
    def load_results(self):
        """Load evaluation results"""
        try:
            # Try different result file names
            possible_files = ["detailed_results.json", "mock_results.json", "quick_results.json"]
            
            for filename in possible_files:
                result_file = self.results_dir / filename
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        self.results_data = json.load(f)
                    
                    # Convert to DataFrame for analysis
                    self.df = pd.DataFrame(self.results_data)
                    print(f"✅ Loaded {len(self.results_data)} evaluation results from {filename}")
                    return True
            
            print(f"❌ No results files found in: {self.results_dir}")
            return False
                
        except Exception as e:
            print(f"❌ Error loading results: {str(e)}")
            return False
    
    def analyze_feedback_quality(self):
        """Analyze feedback generation quality"""
        if self.df is None:
            return
        
        print("\n📊 FEEDBACK QUALITY ANALYSIS")
        print("=" * 40)
        
        # Overall statistics
        valid_ratings = self.df[self.df['feedback_rating'] > 0]['feedback_rating']
        
        if len(valid_ratings) > 0:
            print(f"Total feedback samples: {len(valid_ratings)}")
            print(f"Average rating: {valid_ratings.mean():.2f}/5.0")
            print(f"Standard deviation: {valid_ratings.std():.2f}")
            print(f"Min rating: {valid_ratings.min():.1f}")
            print(f"Max rating: {valid_ratings.max():.1f}")
            
            # Rating distribution
            print(f"\nRating Distribution:")
            rating_counts = valid_ratings.value_counts().sort_index()
            for rating, count in rating_counts.items():
                percentage = (count / len(valid_ratings)) * 100
                print(f"  {rating:.1f}: {count} ({percentage:.1f}%)")
        
        # Domain-specific analysis
        print(f"\nDomain-specific Feedback Quality:")
        domain_stats = self.df.groupby('domain')['feedback_rating'].agg(['count', 'mean', 'std']).round(2)
        for domain, stats in domain_stats.iterrows():
            print(f"  {domain}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Rating: {stats['mean']:.2f}/5.0")
            print(f"    Std Dev: {stats['std']:.2f}")
    
    def analyze_ai_detection(self):
        """Analyze AI detection performance"""
        if self.df is None:
            return
        
        print("\n🔍 AI DETECTION ANALYSIS")
        print("=" * 40)
        
        # Calculate accuracy
        valid_detections = self.df[self.df['ai_detection_result'].apply(
            lambda x: isinstance(x, dict) and x.get('label') not in ['Error', 'Unknown']
        )]
        
        if len(valid_detections) > 0:
            # Extract predicted labels
            predicted_labels = valid_detections['ai_detection_result'].apply(
                lambda x: x.get('label', 'Unknown')
            )
            actual_labels = valid_detections['ground_truth_label']
            
            # Overall accuracy
            correct_predictions = (predicted_labels == actual_labels).sum()
            total_predictions = len(valid_detections)
            overall_accuracy = correct_predictions / total_predictions
            
            print(f"Total predictions: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Overall accuracy: {overall_accuracy:.2%}")
            
            # Confusion matrix
            print(f"\nConfusion Matrix:")
            labels = ['Human', 'AI', 'Hybrid']
            confusion_data = []
            
            for actual in labels:
                row = []
                for predicted in labels:
                    count = ((actual_labels == actual) & (predicted_labels == predicted)).sum()
                    row.append(count)
                confusion_data.append(row)
                
            confusion_df = pd.DataFrame(confusion_data, index=labels, columns=labels)
            print("           Predicted")
            print("Actual   ", end="")
            for col in labels:
                print(f"{col:>8}", end="")
            print()
            
            for i, actual in enumerate(labels):
                print(f"{actual:<8} ", end="")
                for j, predicted in enumerate(labels):
                    print(f"{confusion_df.iloc[i, j]:>8}", end="")
                print()
            
            # Per-label accuracy
            print(f"\nPer-label Performance:")
            for label in labels:
                label_mask = actual_labels == label
                if label_mask.sum() > 0:
                    label_correct = ((actual_labels == label) & (predicted_labels == label)).sum()
                    label_total = label_mask.sum()
                    label_accuracy = label_correct / label_total
                    print(f"  {label}: {label_correct}/{label_total} ({label_accuracy:.1%})")
        
        # Domain-specific analysis
        print(f"\nDomain-specific AI Detection:")
        for domain in self.df['domain'].unique():
            domain_data = self.df[self.df['domain'] == domain]
            domain_predicted = domain_data['ai_detection_result'].apply(
                lambda x: x.get('label', 'Unknown') if isinstance(x, dict) else 'Unknown'
            )
            domain_actual = domain_data['ground_truth_label']
            domain_correct = (domain_predicted == domain_actual).sum()
            domain_total = len(domain_data)
            domain_accuracy = domain_correct / domain_total if domain_total > 0 else 0
            
            print(f"  {domain}: {domain_correct}/{domain_total} ({domain_accuracy:.1%})")
    
    def analyze_performance_metrics(self):
        """Analyze processing time and efficiency metrics"""
        if self.df is None:
            return
        
        print("\n⚡ PERFORMANCE METRICS")
        print("=" * 40)
        
        processing_times = self.df['processing_time']
        
        print(f"Total submissions processed: {len(processing_times)}")
        print(f"Average processing time: {processing_times.mean():.2f}s")
        print(f"Median processing time: {processing_times.median():.2f}s")
        print(f"Min processing time: {processing_times.min():.2f}s")
        print(f"Max processing time: {processing_times.max():.2f}s")
        print(f"Total processing time: {processing_times.sum():.1f}s")
        
        # Domain-specific performance
        print(f"\nDomain-specific Processing Times:")
        domain_times = self.df.groupby('domain')['processing_time'].agg(['count', 'mean', 'std']).round(2)
        for domain, stats in domain_times.iterrows():
            print(f"  {domain}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Time: {stats['mean']:.2f}s")
            print(f"    Std Dev: {stats['std']:.2f}s")
    
    def analyze_errors(self):
        """Analyze errors encountered during evaluation"""
        if self.df is None:
            return
        
        print("\n🚨 ERROR ANALYSIS")
        print("=" * 40)
        
        # Count submissions with errors
        error_count = self.df['errors'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
        total_count = len(self.df)
        
        print(f"Submissions with errors: {error_count}/{total_count} ({error_count/total_count:.1%})")
        
        if error_count > 0:
            # Collect all error messages
            all_errors = []
            for errors_list in self.df['errors']:
                if isinstance(errors_list, list):
                    all_errors.extend(errors_list)
            
            if all_errors:
                print(f"\nError Types:")
                error_counts = {}
                for error in all_errors:
                    error_counts[error] = error_counts.get(error, 0) + 1
                
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {count}x: {error}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if self.df is None:
            return
        
        print("\n📋 COMPREHENSIVE SUMMARY REPORT")
        print("=" * 60)
        
        # Basic stats
        total_submissions = len(self.df)
        domains_tested = self.df['domain'].nunique()
        
        print(f"📊 Overview:")
        print(f"  Total submissions evaluated: {total_submissions}")
        print(f"  Domains tested: {domains_tested}")
        print(f"  Unique domains: {', '.join(self.df['domain'].unique())}")
        
        # Feedback quality summary
        valid_ratings = self.df[self.df['feedback_rating'] > 0]['feedback_rating']
        if len(valid_ratings) > 0:
            print(f"\n📝 Feedback Generation:")
            print(f"  Average quality rating: {valid_ratings.mean():.2f}/5.0")
            print(f"  Successful generations: {len(valid_ratings)}/{total_submissions}")
            
            # Quality categories
            excellent = (valid_ratings >= 4.5).sum()
            good = ((valid_ratings >= 3.5) & (valid_ratings < 4.5)).sum()
            average = ((valid_ratings >= 2.5) & (valid_ratings < 3.5)).sum()
            poor = (valid_ratings < 2.5).sum()
            
            print(f"  Quality breakdown:")
            print(f"    Excellent (4.5+): {excellent} ({excellent/len(valid_ratings):.1%})")
            print(f"    Good (3.5-4.4): {good} ({good/len(valid_ratings):.1%})")
            print(f"    Average (2.5-3.4): {average} ({average/len(valid_ratings):.1%})")
            print(f"    Poor (<2.5): {poor} ({poor/len(valid_ratings):.1%})")
        
        # AI detection summary
        valid_detections = self.df[self.df['ai_detection_result'].apply(
            lambda x: isinstance(x, dict) and x.get('label') not in ['Error', 'Unknown']
        )]
        
        if len(valid_detections) > 0:
            predicted_labels = valid_detections['ai_detection_result'].apply(
                lambda x: x.get('label', 'Unknown')
            )
            actual_labels = valid_detections['ground_truth_label']
            correct_predictions = (predicted_labels == actual_labels).sum()
            
            print(f"\n🔍 AI Detection:")
            print(f"  Overall accuracy: {correct_predictions/len(valid_detections):.1%}")
            print(f"  Successful detections: {len(valid_detections)}/{total_submissions}")
        
        # Performance summary
        processing_times = self.df['processing_time']
        print(f"\n⚡ Performance:")
        print(f"  Average processing time: {processing_times.mean():.2f}s per submission")
        print(f"  Total evaluation time: {processing_times.sum():.1f}s")
        
        # Save summary to file
        summary_file = self.results_dir / "summary_report.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PHI2 MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total submissions: {total_submissions}\n")
            f.write(f"Domains tested: {domains_tested}\n")
            if len(valid_ratings) > 0:
                f.write(f"Avg feedback rating: {valid_ratings.mean():.2f}/5.0\n")
            if len(valid_detections) > 0:
                f.write(f"AI detection accuracy: {correct_predictions/len(valid_detections):.1%}\n")
            f.write(f"Avg processing time: {processing_times.mean():.2f}s\n")
        
        print(f"\n💾 Summary saved to {summary_file}")

def main():
    """Main analysis function"""
    print("📈 Phi2 Results Analysis Dashboard")
    print("=" * 50)
    
    analyzer = ResultsAnalyzer()
    
    # Try to load results from different possible locations
    result_locations = [
        "/workspace/evaluation_results",
        "/workspace/quick_test_results",
        "/workspace/mock_test_results"
    ]
    
    loaded = False
    for location in result_locations:
        analyzer.results_dir = Path(location)
        if analyzer.load_results():
            loaded = True
            break
    
    if not loaded:
        print("❌ No evaluation results found!")
        print("💡 Run 'python3 run_phi2_tests.py' or 'python3 quick_phi2_test.py' first")
        return 1
    
    # Run all analyses
    analyzer.analyze_feedback_quality()
    analyzer.analyze_ai_detection()
    analyzer.analyze_performance_metrics()
    analyzer.analyze_errors()
    analyzer.generate_summary_report()
    
    print("\n🎉 Analysis completed!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())