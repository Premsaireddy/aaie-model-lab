"""
Visualization and Analysis Tools for ZSL/FSL Experiments
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ExperimentAnalyzer:
    """Analyze and visualize ZSL/FSL experiment results"""
    
    def __init__(self, results_path: str = None):
        """Initialize analyzer with results"""
        self.results = None
        if results_path:
            self.load_results(results_path)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_results(self, results_path: str):
        """Load experiment results from JSON file"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded results from {results_path}")
    
    def create_feedback_analysis(self, output_dir: Path = Path("outputs/visualizations")):
        """Create comprehensive feedback analysis visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results or not self.results.get('feedback_results'):
            print("No feedback results to analyze")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results['feedback_results'])
        
        # Extract evaluation metrics
        eval_df = pd.json_normalize(df['evaluation'])
        
        # 1. Overall Feedback Scores Distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Zero-Shot Learning - Feedback Generation Analysis', fontsize=16)
        
        # Individual metric distributions
        metrics = ['feedback_relevance', 'feedback_specificity', 'feedback_alignment',
                  'feedback_constructiveness', 'feedback_completeness']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            ax.hist(eval_df[metric], bins=20, edgecolor='black', alpha=0.7)
            ax.set_title(metric.replace('feedback_', '').replace('_', ' ').title())
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.axvline(eval_df[metric].mean(), color='red', linestyle='--', 
                      label=f'Mean: {eval_df[metric].mean():.3f}')
            ax.legend()
        
        # Overall scores
        ax = axes[1, 2]
        overall_scores = df['overall_score']
        ax.hist(overall_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.set_title('Overall Feedback Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.axvline(overall_scores.mean(), color='red', linestyle='--',
                  label=f'Mean: {overall_scores.mean():.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feedback_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = eval_df[metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feedback Metrics Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'feedback_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Dataset-wise Performance
        if 'dataset' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            dataset_scores = df.groupby('dataset')['overall_score'].agg(['mean', 'std', 'count'])
            dataset_scores.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
            ax.set_title('Feedback Quality by Dataset', fontsize=14)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Mean Overall Score')
            ax.set_ylim([0, 1])
            
            # Add count labels
            for i, (idx, row) in enumerate(dataset_scores.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.02, f"n={int(row['count'])}", 
                       ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'feedback_by_dataset.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"Feedback analysis visualizations saved to {output_dir}")
    
    def create_detection_analysis(self, output_dir: Path = Path("outputs/visualizations")):
        """Create comprehensive AI detection analysis visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results or not self.results.get('detection_results'):
            print("No detection results to analyze")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results['detection_results'])
        
        # Filter valid predictions
        valid_df = df[df['predicted_label'].notna()].copy()
        
        # 1. Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standard confusion matrix
        cm = confusion_matrix(valid_df['true_label'], valid_df['predicted_label'],
                            labels=['Human', 'AI', 'Hybrid'])
        
        ax = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'AI', 'Hybrid'],
                   yticklabels=['Human', 'AI', 'Hybrid'], ax=ax)
        ax.set_title('Confusion Matrix - AI Detection (FSL)', fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax = axes[1]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=['Human', 'AI', 'Hybrid'],
                   yticklabels=['Human', 'AI', 'Hybrid'], ax=ax)
        ax.set_title('Normalized Confusion Matrix', fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confidence Analysis
        if 'confidence' in valid_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Confidence distribution
            ax = axes[0]
            confidence_counts = valid_df['confidence'].value_counts()
            confidence_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
            ax.set_title('Confidence Distribution', fontsize=14)
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            
            # Confidence vs Accuracy
            ax = axes[1]
            conf_accuracy = valid_df.groupby('confidence').apply(
                lambda x: (x['true_label'] == x['predicted_label']).mean()
            )
            conf_accuracy.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Accuracy by Confidence Level', fontsize=14)
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'detection_confidence.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Evaluation Metrics Distribution
        eval_df = pd.json_normalize(df['evaluation'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Few-Shot Learning - AI Detection Metrics', fontsize=16)
        
        metrics = ['detection_accuracy', 'detection_confidence', 
                  'rationale_quality', 'flag_relevance']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            ax.hist(eval_df[metric], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_title(metric.replace('detection_', '').replace('_', ' ').title())
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.axvline(eval_df[metric].mean(), color='red', linestyle='--',
                      label=f'Mean: {eval_df[metric].mean():.3f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Performance by Label Type
        fig, ax = plt.subplots(figsize=(10, 6))
        
        label_performance = valid_df.groupby('true_label').apply(
            lambda x: pd.Series({
                'accuracy': (x['true_label'] == x['predicted_label']).mean(),
                'count': len(x)
            })
        )
        
        label_performance['accuracy'].plot(kind='bar', ax=ax, color='teal')
        ax.set_title('Detection Accuracy by True Label Type', fontsize=14)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Add count annotations
        for i, (idx, row) in enumerate(label_performance.iterrows()):
            ax.text(i, row['accuracy'] + 0.02, f"n={int(row['count'])}", 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_by_label.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Detection analysis visualizations saved to {output_dir}")
    
    def create_interactive_dashboard(self, output_path: str = "outputs/dashboard.html"):
        """Create interactive Plotly dashboard"""
        if not self.results:
            print("No results loaded")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Feedback Scores Distribution', 'Detection Accuracy',
                          'Feedback Metrics Comparison', 'Detection Confusion Matrix',
                          'API Usage Statistics', 'Performance Timeline'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                  [{'type': 'box'}, {'type': 'heatmap'}],
                  [{'type': 'indicator'}, {'type': 'scatter'}]]
        )
        
        # 1. Feedback Scores Distribution
        if self.results.get('feedback_results'):
            feedback_scores = [r['overall_score'] for r in self.results['feedback_results']]
            fig.add_trace(
                go.Histogram(x=feedback_scores, name='Feedback Scores', nbinsx=20),
                row=1, col=1
            )
        
        # 2. Detection Accuracy by Label
        if self.results.get('detection_results'):
            detection_df = pd.DataFrame(self.results['detection_results'])
            valid_detection = detection_df[detection_df['predicted_label'].notna()]
            
            accuracy_by_label = valid_detection.groupby('true_label').apply(
                lambda x: (x['true_label'] == x['predicted_label']).mean()
            )
            
            fig.add_trace(
                go.Bar(x=accuracy_by_label.index, y=accuracy_by_label.values,
                      name='Accuracy by Label'),
                row=1, col=2
            )
        
        # 3. Feedback Metrics Box Plot
        if self.results.get('feedback_results'):
            eval_data = []
            metrics = ['feedback_relevance', 'feedback_specificity', 'feedback_alignment',
                      'feedback_constructiveness', 'feedback_completeness']
            
            for metric in metrics:
                values = [r['evaluation'][metric] for r in self.results['feedback_results']]
                for v in values:
                    eval_data.append({'metric': metric.replace('feedback_', ''), 'value': v})
            
            eval_df = pd.DataFrame(eval_data)
            
            for metric in eval_df['metric'].unique():
                metric_data = eval_df[eval_df['metric'] == metric]['value']
                fig.add_trace(
                    go.Box(y=metric_data, name=metric.title()),
                    row=2, col=1
                )
        
        # 4. Confusion Matrix Heatmap
        if self.results.get('detection_results'):
            cm = confusion_matrix(valid_detection['true_label'], 
                                valid_detection['predicted_label'],
                                labels=['Human', 'AI', 'Hybrid'])
            
            fig.add_trace(
                go.Heatmap(z=cm, x=['Human', 'AI', 'Hybrid'], 
                          y=['Human', 'AI', 'Hybrid'],
                          colorscale='Blues', showscale=True),
                row=2, col=2
            )
        
        # 5. API Usage Indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=self.results['metadata'].get('total_tokens', 0),
                title={"text": "Total Tokens Used"},
                delta={'reference': 100000, 'relative': True}
            ),
            row=3, col=1
        )
        
        # 6. Performance Timeline (if multiple runs)
        if self.results.get('feedback_results'):
            indices = list(range(len(self.results['feedback_results'])))
            feedback_timeline = [r['overall_score'] for r in self.results['feedback_results']]
            
            fig.add_trace(
                go.Scatter(x=indices, y=feedback_timeline, mode='lines+markers',
                          name='Feedback Score Timeline'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="ZSL/FSL Experiment Dashboard",
            title_font_size=20
        )
        
        # Save dashboard
        fig.write_html(output_path)
        print(f"Interactive dashboard saved to {output_path}")
        
        return fig
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        if not self.results:
            return {}
        
        summary = {
            "experiment_metadata": self.results['metadata'],
            "feedback_statistics": {},
            "detection_statistics": {}
        }
        
        # Feedback statistics
        if self.results.get('feedback_results'):
            feedback_scores = [r['overall_score'] for r in self.results['feedback_results']]
            summary['feedback_statistics'] = {
                "total_evaluations": len(feedback_scores),
                "mean_score": np.mean(feedback_scores),
                "std_score": np.std(feedback_scores),
                "min_score": np.min(feedback_scores),
                "max_score": np.max(feedback_scores),
                "median_score": np.median(feedback_scores),
                "quartiles": {
                    "Q1": np.percentile(feedback_scores, 25),
                    "Q2": np.percentile(feedback_scores, 50),
                    "Q3": np.percentile(feedback_scores, 75)
                }
            }
        
        # Detection statistics
        if self.results.get('detection_results'):
            detection_df = pd.DataFrame(self.results['detection_results'])
            valid_detection = detection_df[detection_df['predicted_label'].notna()]
            
            if len(valid_detection) > 0:
                accuracy = (valid_detection['true_label'] == valid_detection['predicted_label']).mean()
                
                summary['detection_statistics'] = {
                    "total_predictions": len(valid_detection),
                    "overall_accuracy": accuracy,
                    "label_distribution": valid_detection['true_label'].value_counts().to_dict(),
                    "prediction_distribution": valid_detection['predicted_label'].value_counts().to_dict(),
                    "confidence_distribution": valid_detection['confidence'].value_counts().to_dict() 
                                             if 'confidence' in valid_detection.columns else {}
                }
        
        return summary
    
    def export_to_excel(self, output_path: str = "outputs/experiment_results.xlsx"):
        """Export all results to Excel workbook"""
        if not self.results:
            print("No results to export")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Metadata
            metadata_df = pd.DataFrame([self.results['metadata']])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Feedback results
            if self.results.get('feedback_results'):
                feedback_df = pd.DataFrame(self.results['feedback_results'])
                # Flatten evaluation columns
                eval_cols = pd.json_normalize(feedback_df['evaluation'])
                feedback_df = pd.concat([feedback_df.drop('evaluation', axis=1), eval_cols], axis=1)
                feedback_df.to_excel(writer, sheet_name='Feedback Results', index=False)
            
            # Detection results
            if self.results.get('detection_results'):
                detection_df = pd.DataFrame(self.results['detection_results'])
                # Flatten evaluation columns
                eval_cols = pd.json_normalize(detection_df['evaluation'])
                detection_df = pd.concat([detection_df.drop('evaluation', axis=1), eval_cols], axis=1)
                detection_df.to_excel(writer, sheet_name='Detection Results', index=False)
            
            # Summary statistics
            summary = self.generate_summary_statistics()
            summary_df = pd.DataFrame([summary['feedback_statistics']])
            summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        print(f"Results exported to {output_path}")


def main():
    """Main function for standalone analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ZSL/FSL experiment results')
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--output-dir', default='outputs/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--dashboard', action='store_true',
                       help='Generate interactive dashboard')
    parser.add_argument('--excel', action='store_true',
                       help='Export results to Excel')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.results_file)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    analyzer.create_feedback_analysis(output_dir)
    analyzer.create_detection_analysis(output_dir)
    
    # Generate dashboard if requested
    if args.dashboard:
        analyzer.create_interactive_dashboard(str(output_dir / 'dashboard.html'))
    
    # Export to Excel if requested
    if args.excel:
        analyzer.export_to_excel(str(output_dir / 'results.xlsx'))
    
    # Print summary
    summary = analyzer.generate_summary_statistics()
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()