#!/usr/bin/env python3
"""
Main runner script for ZSL/FSL experiments with OpenAI GPT-4
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import os

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from openai_zsl_fsl_pipeline import OpenAIZSLFSLPipeline
from visualization_analysis import ExperimentAnalyzer
from config import (
    get_config, validate_config, create_directories,
    OPENAI_API_KEY, OPENAI_MODEL, DATASETS
)


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def run_full_experiment(args):
    """Run complete ZSL/FSL experiment"""
    logger = setup_logging(args.log_file)
    
    # Validate configuration
    logger.info("Validating configuration...")
    errors = validate_config()
    if errors:
        logger.error("Configuration errors found:")
        for error in errors:
            logger.error(f"  - {error}")
        if not args.force:
            logger.error("Exiting. Use --force to continue anyway.")
            return 1
    
    # Create directories
    create_directories()
    
    # Initialize pipeline
    logger.info(f"Initializing OpenAI pipeline with model: {args.model or OPENAI_MODEL}")
    
    try:
        pipeline = OpenAIZSLFSLPipeline(
            api_key=args.api_key or OPENAI_API_KEY,
            model=args.model or OPENAI_MODEL
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Prepare data paths
    if args.datasets:
        data_paths = [Path(d) for d in args.datasets]
    else:
        data_paths = [Path(d) for d in DATASETS]
    
    # Filter existing files
    existing_paths = [p for p in data_paths if p.exists()]
    if not existing_paths:
        logger.error("No data files found!")
        logger.info("Run 'python create_sample_data.py' to create sample data.")
        return 1
    
    logger.info(f"Found {len(existing_paths)} data files: {[str(p) for p in existing_paths]}")
    
    # Run experiment
    logger.info("Starting ZSL/FSL experiment...")
    try:
        results = pipeline.run_experiment(
            data_paths=existing_paths,
            output_dir=Path(args.output_dir)
        )
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Total API calls: {pipeline.api_calls}")
        logger.info(f"Total tokens used: {pipeline.total_tokens}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1
    
    # Generate report
    if not args.skip_report:
        logger.info("Generating evaluation report...")
        try:
            report_path = pipeline.generate_report(
                results=results,
                output_dir=Path(args.output_dir)
            )
            logger.info(f"Report generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    # Generate visualizations
    if args.visualize:
        logger.info("Creating visualizations...")
        try:
            # Save results for analyzer
            results_path = Path(args.output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Run analyzer
            analyzer = ExperimentAnalyzer(str(results_path))
            viz_dir = Path(args.output_dir) / "visualizations"
            
            analyzer.create_feedback_analysis(viz_dir)
            analyzer.create_detection_analysis(viz_dir)
            
            if args.dashboard:
                dashboard_path = viz_dir / "dashboard.html"
                analyzer.create_interactive_dashboard(str(dashboard_path))
                logger.info(f"Interactive dashboard created: {dashboard_path}")
            
            if args.excel:
                excel_path = viz_dir / "results.xlsx"
                analyzer.export_to_excel(str(excel_path))
                logger.info(f"Excel export created: {excel_path}")
            
            logger.info(f"Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if 'evaluation_scores' in results:
        if results['evaluation_scores'].get('feedback'):
            import numpy as np
            avg_feedback = np.mean(results['evaluation_scores']['feedback'])
            print(f"Average Feedback Score: {avg_feedback:.3f}")
        
        if results['evaluation_scores'].get('detection'):
            import numpy as np
            avg_detection = np.mean(results['evaluation_scores']['detection'])
            print(f"Average Detection Score: {avg_detection:.3f}")
    
    print(f"Output Directory: {args.output_dir}")
    print("="*60)
    
    return 0


def create_sample_data(args):
    """Create sample data files"""
    logger = setup_logging()
    logger.info("Creating sample data files...")
    
    try:
        from create_sample_data import main as create_data
        create_data()
        logger.info("Sample data created successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return 1


def analyze_results(args):
    """Analyze existing results"""
    logger = setup_logging()
    
    if not Path(args.results_file).exists():
        logger.error(f"Results file not found: {args.results_file}")
        return 1
    
    logger.info(f"Analyzing results from: {args.results_file}")
    
    try:
        analyzer = ExperimentAnalyzer(args.results_file)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        analyzer.create_feedback_analysis(output_dir)
        analyzer.create_detection_analysis(output_dir)
        
        # Generate dashboard if requested
        if args.dashboard:
            dashboard_path = output_dir / "dashboard.html"
            analyzer.create_interactive_dashboard(str(dashboard_path))
            logger.info(f"Dashboard created: {dashboard_path}")
        
        # Export to Excel if requested
        if args.excel:
            excel_path = output_dir / "results.xlsx"
            analyzer.export_to_excel(str(excel_path))
            logger.info(f"Excel export created: {excel_path}")
        
        # Print summary statistics
        summary = analyzer.generate_summary_statistics()
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Zero-Shot and Few-Shot Learning Experiments with OpenAI GPT-4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment with default settings
  python run_experiment.py
  
  # Create sample data first
  python run_experiment.py create-data
  
  # Run experiment with custom model
  python run_experiment.py --model gpt-4 --visualize
  
  # Analyze existing results
  python run_experiment.py analyze results.json --dashboard
  
  # Run with specific datasets
  python run_experiment.py --datasets accounting.json psychology.json
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Default command (run experiment)
    parser.add_argument('--api-key', help='OpenAI API key (overrides config/env)')
    parser.add_argument('--model', help='Model to use (default: from config)')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to process')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--skip-report', action='store_true', help='Skip report generation')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--dashboard', action='store_true', help='Create interactive dashboard')
    parser.add_argument('--excel', action='store_true', help='Export results to Excel')
    parser.add_argument('--force', action='store_true', help='Continue despite config errors')
    
    # Create data subcommand
    create_parser = subparsers.add_parser('create-data', help='Create sample data files')
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing results')
    analyze_parser.add_argument('results_file', help='Path to results JSON file')
    analyze_parser.add_argument('--output-dir', default='outputs/analysis', 
                               help='Output directory for visualizations')
    analyze_parser.add_argument('--dashboard', action='store_true', 
                               help='Create interactive dashboard')
    analyze_parser.add_argument('--excel', action='store_true', 
                               help='Export to Excel')
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.command == 'create-data':
        return create_sample_data(args)
    elif args.command == 'analyze':
        return analyze_results(args)
    else:
        # Default: run experiment
        return run_full_experiment(args)


if __name__ == "__main__":
    sys.exit(main())