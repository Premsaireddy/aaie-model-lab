#!/usr/bin/env python3
"""
Academic Submission Analyzer using Microsoft Phi-2 Model
Provides comprehensive feedback and AI detection capabilities
"""

import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline"""
    model_name: str = "microsoft/phi-2"
    max_tokens: int = 800
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    batch_size: int = 1

class Phi2ModelManager:
    """Manages Phi-2 model initialization and inference"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = self._get_optimal_device()
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available - Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def _initialize_model(self):
        """Initialize tokenizer and model with optimal settings"""
        logger.info(f"Loading {self.config.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Phi-2 model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return "Error: Unable to generate response"

class PromptEngineering:
    """Advanced prompt engineering for different analysis tasks"""
    
    @staticmethod
    def create_feedback_prompt(domain: str, assignment: str, rubric: str, submission: str) -> str:
        """Generate comprehensive feedback prompt"""
        return f"""<|im_start|>system
You are an expert academic evaluator specializing in {domain}. Your task is to provide constructive, detailed feedback that helps students improve their work.

Analysis Framework:
1. Identify key strengths and areas for development
2. Align assessment with provided rubric criteria
3. Offer specific, actionable recommendations
4. Maintain encouraging but honest tone
<|im_end|>

<|im_start|>user
Assignment Context: {assignment}

Evaluation Rubric:
{rubric}

Student Submission to Evaluate:
{submission}

Please provide structured feedback covering:
• Strengths: What the student did well
• Development Areas: Specific aspects needing improvement  
• Rubric Alignment: How the work meets each criterion
• Actionable Steps: Concrete next steps for improvement
• Overall Assessment: Summary evaluation

Respond with clear, structured analysis:
<|im_end|>

<|im_start|>assistant
"""

    @staticmethod
    def create_detection_prompt(submission: str, examples: List[Dict]) -> str:
        """Generate AI detection analysis prompt"""
        example_text = ""
        if examples:
            example_text = "\nReference Examples:\n"
            for idx, ex in enumerate(examples[:3], 1):
                example_text += f"Example {idx}: {ex.get('final_submission', '')[:200]}... [Label: {ex.get('label_type', 'Unknown')}]\n"
        
        return f"""<|im_start|>system
You are an advanced AI detection specialist for academic integrity. Analyze text characteristics to determine authorship patterns.

Detection Criteria:
- Writing style consistency and natural flow
- Vocabulary sophistication and domain-specific usage  
- Structural patterns and coherence markers
- Personal voice and subjective elements
- Technical accuracy vs. generic statements
<|im_end|>

<|im_start|>user
{example_text}

Analyze this submission for AI involvement:
{submission}

Provide analysis in this format:
Classification: [Human/AI/Hybrid]
Confidence Level: [High/Medium/Low]
Key Indicators:
• [Specific evidence point 1]
• [Specific evidence point 2] 
• [Specific evidence point 3]
Reasoning: [Brief explanation of decision]
<|im_end|>

<|im_start|>assistant
"""

class SubmissionProcessor:
    """Processes academic submissions with advanced analysis capabilities"""
    
    def __init__(self, model_manager: Phi2ModelManager):
        self.model_manager = model_manager
        self.prompt_engineer = PromptEngineering()
    
    def analyze_submission(self, submission_data: Dict, rubric: str, domain: str, 
                          assignment_prompt: str, examples: List[Dict]) -> Dict[str, Any]:
        """Comprehensive analysis of a single submission"""
        submission_text = submission_data.get('final_submission', '')
        
        # Generate feedback analysis
        feedback_prompt = self.prompt_engineer.create_feedback_prompt(
            domain, assignment_prompt, rubric, submission_text
        )
        feedback_result = self.model_manager.generate_response(feedback_prompt)
        
        # Generate detection analysis  
        detection_prompt = self.prompt_engineer.create_detection_prompt(
            submission_text, examples
        )
        detection_result = self.model_manager.generate_response(detection_prompt)
        
        return {
            'original_label': submission_data.get('label_type', 'Unknown'),
            'feedback_analysis': feedback_result,
            'detection_analysis': detection_result,
            'submission_length': len(submission_text),
            'submission_preview': submission_text[:200] + "..." if len(submission_text) > 200 else submission_text
        }
    
    def process_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Process entire dataset file"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Format rubric
            rubric_text = self._format_rubric_structure(data.get('rubric', {}))
            domain = data.get('domain', 'Academic')
            assignment = data.get('prompt', 'Assignment analysis and feedback')
            examples = data.get('few_shots', [])
            submissions = data.get('submissions', [])
            
            logger.info(f"Processing {len(submissions)} submissions from {dataset_path}")
            
            results = []
            for idx, submission in enumerate(submissions, 1):
                logger.info(f"Analyzing submission {idx}/{len(submissions)}")
                
                analysis = self.analyze_submission(
                    submission, rubric_text, domain, assignment, examples
                )
                analysis['submission_id'] = idx
                analysis['dataset'] = Path(dataset_path).stem
                results.append(analysis)
            
            return results
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in: {dataset_path}")
            return []
    
    def _format_rubric_structure(self, rubric: Dict) -> str:
        """Format rubric data into readable structure"""
        if not rubric:
            return "No rubric provided"
        
        formatted = f"Assessment Rubric: {rubric.get('rubric_id', 'N/A')}\n\n"
        
        for criterion in rubric.get('criteria', []):
            formatted += f"Criterion: {criterion.get('name', 'Unnamed')}\n"
            formatted += f"Description: {criterion.get('description', 'No description')}\n"
            
            descriptors = criterion.get('performance_descriptors', {})
            if descriptors:
                formatted += "Performance Levels:\n"
                for level, description in descriptors.items():
                    formatted += f"  • {level.title()}: {description}\n"
            formatted += "\n"
        
        return formatted.strip()

class ResultsFormatter:
    """Formats and displays analysis results"""
    
    @staticmethod
    def display_analysis(result: Dict[str, Any], show_preview: bool = True):
        """Display formatted analysis results"""
        print(f"\n{'='*80}")
        print(f"SUBMISSION {result['submission_id']} - {result['dataset'].upper()} DATASET")
        print(f"Original Label: {result['original_label']}")
        print(f"Length: {result['submission_length']} characters")
        print(f"{'='*80}")
        
        if show_preview:
            print(f"\nSubmission Preview:")
            print(f"{result['submission_preview']}")
        
        print(f"\n{'-'*50} ACADEMIC FEEDBACK {'-'*50}")
        print(result['feedback_analysis'])
        
        print(f"\n{'-'*50} AI DETECTION ANALYSIS {'-'*50}")
        print(result['detection_analysis'])
    
    @staticmethod
    def save_results(results: List[Dict], output_file: str):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def main():
    """Main execution pipeline"""
    # Initialize configuration
    config = AnalysisConfig()
    
    # Initialize model manager
    model_manager = Phi2ModelManager(config)
    
    # Initialize processor
    processor = SubmissionProcessor(model_manager)
    
    # Initialize formatter
    formatter = ResultsFormatter()
    
    # Dataset files to process
    datasets = ["psychology.json", "engineering.json"]
    all_results = []
    
    for dataset in datasets:
        if Path(dataset).exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING: {dataset.upper()}")
            logger.info(f"{'='*60}")
            
            # Process dataset
            results = processor.process_dataset(dataset)
            
            # Display results
            for result in results:
                formatter.display_analysis(result, show_preview=False)
            
            all_results.extend(results)
        else:
            logger.warning(f"Dataset file not found: {dataset}")
    
    # Save comprehensive results
    if all_results:
        formatter.save_results(all_results, "phi2_analysis_results.json")
        logger.info(f"\nAnalysis complete! Processed {len(all_results)} submissions total.")
    else:
        logger.warning("No submissions were processed.")

if __name__ == "__main__":
    main()