#!/usr/bin/env python3
"""
Phi2 Model Evaluation Framework
==============================

This framework tests the Phi2 model's ability to:
1. Generate feedback using rubrics from training datasets
2. Rate the quality of generated feedback 
3. Detect AI in submissions

Author: AI Assistant
Date: 2024
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# ML Libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import pandas as pd

# Add the base prompts to path and import
sys.path.append('Model and Prompt Selection/Models/Base Prompts')
from base_prompts import build_detection_prompt, build_feedback_prompt, self_eval_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    domain: str
    submission_id: int
    generated_feedback: str
    feedback_rating: float
    ai_detection_result: Dict[str, Any]
    ground_truth_label: str
    processing_time: float
    errors: List[str]

class Phi2EvaluationFramework:
    """Main evaluation framework for Phi2 model testing"""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """Initialize the evaluation framework"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Training datasets
        self.training_datasets = {}
        self.results = []
        
    def load_model(self):
        """Load the Phi2 model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_training_datasets(self):
        """Load all training datasets"""
        dataset_dir = Path("Model and Prompt Selection/Models/Training Data")
        dataset_files = ["accounting.json", "psychology.json", "engineering.json", "it.json", "teaching.json"]
        
        for file_name in dataset_files:
            file_path = dataset_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        domain = data['domain']
                        self.training_datasets[domain] = data
                        logger.info(f"Loaded dataset: {domain}")
                except Exception as e:
                    logger.error(f"Error loading {file_name}: {str(e)}")
            else:
                logger.warning(f"Dataset file not found: {file_path}")
    
    def generate_feedback(self, domain: str, assignment_prompt: str, rubric: Dict, submission: str) -> str:
        """Generate feedback using the Phi2 model"""
        try:
            # Build rubric text from the rubric structure
            rubric_text = f"Rubric ID: {rubric['rubric_id']}\n\nCriteria:\n"
            for criterion in rubric['criteria']:
                rubric_text += f"\n{criterion['criterion_id']}: {criterion['name']}\n"
                rubric_text += f"Description: {criterion['description']}\n"
                rubric_text += "Performance Levels:\n"
                for level, desc in criterion['performance_descriptors'].items():
                    rubric_text += f"- {level.title()}: {desc}\n"
            
            # Build prompt using base_prompts.py
            messages = build_feedback_prompt(domain, assignment_prompt, rubric_text, submission)
            
            # Format for Phi2
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\nAssistant: "
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = response[0]['generated_text'].strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON content
                if '{' in generated_text and '}' in generated_text:
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    json_str = generated_text[start_idx:end_idx]
                    # Validate it's proper JSON
                    json.loads(json_str)
                    return json_str
                else:
                    return generated_text
            except:
                return generated_text
                
        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            return f"Error generating feedback: {str(e)}"
    
    def rate_feedback(self, rubric: Dict, submission: str, feedback: str) -> float:
        """Rate the quality of generated feedback using self-evaluation"""
        try:
            # Use the self_eval_prompt from base_prompts.py
            eval_prompt = self_eval_prompt(rubric, submission, feedback)
            
            # Generate rating
            response = self.pipeline(
                eval_prompt,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            rating_text = response[0]['generated_text'].strip()
            
            # Extract numeric rating
            import re
            rating_match = re.search(r'\b([1-5])\b', rating_text)
            if rating_match:
                return float(rating_match.group(1))
            else:
                logger.warning(f"Could not extract rating from: {rating_text}")
                return 3.0  # Default to average
                
        except Exception as e:
            logger.error(f"Error rating feedback: {str(e)}")
            return 3.0  # Default to average on error
    
    def detect_ai_in_submission(self, submission: str, few_shots: List[Dict]) -> Dict[str, Any]:
        """Detect if a submission is AI-generated, human, or hybrid"""
        try:
            # Build detection prompt using base_prompts.py
            messages = build_detection_prompt(submission, few_shots)
            
            # Format for Phi2
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\nAssistant: "
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = response[0]['generated_text'].strip()
            
            # Try to extract JSON from the response
            try:
                if '{' in generated_text and '}' in generated_text:
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    json_str = generated_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Validate required fields
                    if 'label' in result and 'rationale' in result:
                        return result
                
                # Fallback: try to extract label from text
                label_mapping = {"human": "Human", "ai": "AI", "hybrid": "Hybrid"}
                for key, value in label_mapping.items():
                    if key.lower() in generated_text.lower():
                        return {
                            "label": value,
                            "rationale": ["Extracted from model response"],
                            "flags": ["extraction_fallback"]
                        }
                
                # Default fallback
                return {
                    "label": "Unknown",
                    "rationale": ["Could not parse model response"],
                    "flags": ["parsing_error"],
                    "raw_response": generated_text
                }
                
            except Exception as e:
                logger.error(f"Error parsing AI detection response: {str(e)}")
                return {
                    "label": "Unknown",
                    "rationale": [f"Parsing error: {str(e)}"],
                    "flags": ["parsing_error"],
                    "raw_response": generated_text
                }
                
        except Exception as e:
            logger.error(f"Error in AI detection: {str(e)}")
            return {
                "label": "Error",
                "rationale": [f"Model error: {str(e)}"],
                "flags": ["model_error"]
            }
    
    def evaluate_submission(self, domain: str, submission_data: Dict, few_shots: List[Dict]) -> EvaluationResult:
        """Evaluate a single submission"""
        start_time = time.time()
        errors = []
        
        try:
            dataset = self.training_datasets[domain]
            submission = submission_data['final_submission']
            ground_truth_label = submission_data['label_type']
            
            # Generate feedback
            logger.info(f"Generating feedback for {domain} submission...")
            generated_feedback = self.generate_feedback(
                domain,
                dataset['prompt'],
                dataset['rubric'],
                submission
            )
            
            # Rate the feedback
            logger.info(f"Rating feedback quality...")
            feedback_rating = self.rate_feedback(
                dataset['rubric'],
                submission,
                generated_feedback
            )
            
            # Detect AI in submission
            logger.info(f"Detecting AI in submission...")
            ai_detection_result = self.detect_ai_in_submission(submission, few_shots)
            
            processing_time = time.time() - start_time
            
            return EvaluationResult(
                domain=domain,
                submission_id=len(self.results),
                generated_feedback=generated_feedback,
                feedback_rating=feedback_rating,
                ai_detection_result=ai_detection_result,
                ground_truth_label=ground_truth_label,
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Error evaluating submission: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return EvaluationResult(
                domain=domain,
                submission_id=len(self.results),
                generated_feedback="Error generating feedback",
                feedback_rating=0.0,
                ai_detection_result={"label": "Error", "rationale": [error_msg], "flags": ["error"]},
                ground_truth_label=submission_data.get('label_type', 'Unknown'),
                processing_time=time.time() - start_time,
                errors=errors
            )
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run evaluation across all datasets"""
        logger.info("Starting comprehensive evaluation...")
        
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
        
        all_results = []
        summary_stats = {
            'total_submissions': 0,
            'domains_tested': 0,
            'avg_feedback_rating': 0.0,
            'ai_detection_accuracy': 0.0,
            'avg_processing_time': 0.0,
            'domain_results': {}
        }
        
        for domain, dataset in self.training_datasets.items():
            logger.info(f"\n=== Testing domain: {domain} ===")
            domain_results = []
            
            # Use other submissions as few-shots for AI detection
            few_shots = [sub for sub in dataset['submissions']]
            
            for i, submission in enumerate(dataset['submissions']):
                logger.info(f"Processing submission {i+1}/{len(dataset['submissions'])}")
                
                # Use other submissions as few-shots (exclude current one)
                current_few_shots = [s for j, s in enumerate(few_shots) if j != i][:3]  # Use max 3 examples
                
                result = self.evaluate_submission(domain, submission, current_few_shots)
                domain_results.append(result)
                all_results.append(result)
            
            # Calculate domain-specific metrics
            domain_feedback_ratings = [r.feedback_rating for r in domain_results if r.feedback_rating > 0]
            domain_ai_accuracy = self._calculate_ai_detection_accuracy(domain_results)
            domain_processing_time = np.mean([r.processing_time for r in domain_results])
            
            summary_stats['domain_results'][domain] = {
                'submissions_count': len(domain_results),
                'avg_feedback_rating': np.mean(domain_feedback_ratings) if domain_feedback_ratings else 0.0,
                'ai_detection_accuracy': domain_ai_accuracy,
                'avg_processing_time': domain_processing_time
            }
        
        # Calculate overall statistics
        all_feedback_ratings = [r.feedback_rating for r in all_results if r.feedback_rating > 0]
        summary_stats['total_submissions'] = len(all_results)
        summary_stats['domains_tested'] = len(self.training_datasets)
        summary_stats['avg_feedback_rating'] = np.mean(all_feedback_ratings) if all_feedback_ratings else 0.0
        summary_stats['ai_detection_accuracy'] = self._calculate_ai_detection_accuracy(all_results)
        summary_stats['avg_processing_time'] = np.mean([r.processing_time for r in all_results])
        
        self.results = all_results
        return summary_stats
    
    def _calculate_ai_detection_accuracy(self, results: List[EvaluationResult]) -> float:
        """Calculate AI detection accuracy"""
        correct_predictions = 0
        total_predictions = 0
        
        for result in results:
            if result.ai_detection_result.get('label') != 'Error':
                predicted_label = result.ai_detection_result.get('label', 'Unknown')
                actual_label = result.ground_truth_label
                
                if predicted_label == actual_label:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed evaluation report"""
        if not self.results:
            return "No evaluation results available. Run evaluation first."
        
        report = []
        report.append("=" * 80)
        report.append("PHI2 MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Model: {self.model_name}")
        report.append(f"Device: {self.device}")
        report.append(f"Total submissions evaluated: {len(self.results)}")
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE METRICS")
        report.append("-" * 40)
        
        all_feedback_ratings = [r.feedback_rating for r in self.results if r.feedback_rating > 0]
        if all_feedback_ratings:
            report.append(f"Average Feedback Rating: {np.mean(all_feedback_ratings):.2f}/5.0")
            report.append(f"Feedback Rating Std Dev: {np.std(all_feedback_ratings):.2f}")
        
        ai_accuracy = self._calculate_ai_detection_accuracy(self.results)
        report.append(f"AI Detection Accuracy: {ai_accuracy:.2%}")
        
        avg_time = np.mean([r.processing_time for r in self.results])
        report.append(f"Average Processing Time: {avg_time:.2f} seconds")
        report.append("")
        
        # Domain-specific results
        domains = {}
        for result in self.results:
            if result.domain not in domains:
                domains[result.domain] = []
            domains[result.domain].append(result)
        
        for domain, domain_results in domains.items():
            report.append(f"DOMAIN: {domain.upper()}")
            report.append("-" * 40)
            report.append(f"Submissions: {len(domain_results)}")
            
            domain_ratings = [r.feedback_rating for r in domain_results if r.feedback_rating > 0]
            if domain_ratings:
                report.append(f"Avg Feedback Rating: {np.mean(domain_ratings):.2f}/5.0")
            
            domain_accuracy = self._calculate_ai_detection_accuracy(domain_results)
            report.append(f"AI Detection Accuracy: {domain_accuracy:.2%}")
            
            # AI Detection Breakdown
            label_counts = {}
            correct_counts = {}
            for r in domain_results:
                predicted = r.ai_detection_result.get('label', 'Unknown')
                actual = r.ground_truth_label
                
                if predicted not in label_counts:
                    label_counts[predicted] = 0
                    correct_counts[predicted] = 0
                
                label_counts[predicted] += 1
                if predicted == actual:
                    correct_counts[predicted] += 1
            
            report.append("AI Detection Breakdown:")
            for label, count in label_counts.items():
                accuracy = correct_counts[label] / count if count > 0 else 0
                report.append(f"  {label}: {count} predictions, {accuracy:.1%} accuracy")
            
            report.append("")
        
        # Error analysis
        all_errors = []
        for result in self.results:
            all_errors.extend(result.errors)
        
        if all_errors:
            report.append("ERRORS ENCOUNTERED")
            report.append("-" * 40)
            error_counts = {}
            for error in all_errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in error_counts.items():
                report.append(f"{count}x: {error}")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "/workspace/evaluation_results"):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = []
        for result in self.results:
            results_data.append({
                'domain': result.domain,
                'submission_id': result.submission_id,
                'generated_feedback': result.generated_feedback,
                'feedback_rating': result.feedback_rating,
                'ai_detection_result': result.ai_detection_result,
                'ground_truth_label': result.ground_truth_label,
                'processing_time': result.processing_time,
                'errors': result.errors
            })
        
        with open(f"{output_dir}/detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        report = self.generate_detailed_report()
        with open(f"{output_dir}/evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save CSV for analysis
        df_data = []
        for result in self.results:
            df_data.append({
                'domain': result.domain,
                'submission_id': result.submission_id,
                'feedback_rating': result.feedback_rating,
                'predicted_label': result.ai_detection_result.get('label', 'Unknown'),
                'actual_label': result.ground_truth_label,
                'detection_correct': result.ai_detection_result.get('label') == result.ground_truth_label,
                'processing_time': result.processing_time,
                'has_errors': len(result.errors) > 0
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
        
        logger.info(f"Results saved to {output_dir}")

def main():
    """Main execution function"""
    print("Phi2 Model Evaluation Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = Phi2EvaluationFramework()
    
    # Load training datasets
    print("Loading training datasets...")
    framework.load_training_datasets()
    
    if not framework.training_datasets:
        print("No training datasets found. Exiting.")
        return
    
    print(f"Loaded {len(framework.training_datasets)} datasets: {list(framework.training_datasets.keys())}")
    
    # Load model
    print("\nLoading Phi2 model...")
    if not framework.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    summary_stats = framework.run_comprehensive_evaluation()
    
    # Display results
    print("\n" + framework.generate_detailed_report())
    
    # Save results
    print("\nSaving results...")
    framework.save_results()
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()