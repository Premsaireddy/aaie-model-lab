#!/usr/bin/env python3
"""
Phi-2 Model Testing - Improved Version

Enhanced implementation for AI detection and feedback generation using Microsoft's Phi-2 model.

Features:
- Optimized model loading with better memory management
- Enhanced AI detection with multiple metrics
- Batch processing capabilities
- Comprehensive error handling
- Data visualization and analysis
- Configuration management
- Performance monitoring
"""

import argparse
import json
import math
import os
import sys
import statistics
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ML imports
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration for model loading and inference"""
    model_name: str = "microsoft/phi-2"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: str = "auto"
    max_memory: Optional[Dict] = None
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True
    
@dataclass
class InferenceConfig:
    """Configuration for inference parameters"""
    max_length: int = 2048
    stride: int = 1024
    batch_size: int = 4
    max_new_tokens: int = 220
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
@dataclass
class DetectionResult:
    """Result of AI detection analysis"""
    perplexity: float
    prediction: int
    prediction_label: str
    confidence: float
    threshold: float
    additional_metrics: Dict[str, float]


def get_optimal_device() -> str:
    """Get the best available device with memory info"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        print(f"🚀 Using GPU: {gpu_name} ({memory_total:.1f}GB total memory)")
        print(f"📊 Available GPUs: {gpu_count}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("💻 Using CPU (consider using GPU for better performance)")
    
    return device


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")


class OptimizedPhi2Loader:
    """Optimized loader for Phi-2 model with better error handling and memory management"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load model with optimized settings and fallback options"""
        print(f"🔄 Loading {self.config.model_name}...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise
        
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
        }
        
        # Try quantization if requested
        if self.config.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                print("🔧 4-bit quantization enabled")
            except ImportError:
                print("⚠️ bitsandbytes not available, falling back to standard loading")
                model_kwargs["torch_dtype"] = self.config.torch_dtype
        elif self.config.load_in_8bit:
            try:
                model_kwargs["load_in_8bit"] = True
                print("🔧 8-bit quantization enabled")
            except ImportError:
                print("⚠️ bitsandbytes not available, falling back to standard loading")
                model_kwargs["torch_dtype"] = self.config.torch_dtype
        else:
            model_kwargs["torch_dtype"] = self.config.torch_dtype
        
        # Load model with fallback
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed with optimized settings: {e}")
            print("🔄 Trying fallback loading...")
            
            # Fallback to basic loading
            fallback_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch.float32 if get_optimal_device() == "cpu" else self.config.torch_dtype
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **fallback_kwargs
            )
            print("✅ Model loaded with fallback settings")
        
        # Configure model
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        
        # Move to device if not using device_map
        device = get_optimal_device()
        if self.config.device_map != "auto" and not self.config.load_in_4bit and not self.config.load_in_8bit:
            self.model = self.model.to(device)
        
        self._print_model_info()
        return self.tokenizer, self.model
    
    def _print_model_info(self):
        """Print model information"""
        if self.model is None:
            return
            
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model Parameters: {param_count:,}")
        print(f"🎯 Trainable Parameters: {trainable_params:,}")
        print(f"💾 Model Device: {next(self.model.parameters()).device}")
        print(f"🔢 Model Dtype: {next(self.model.parameters()).dtype}")


class PerplexityCalculator:
    """Enhanced perplexity calculator with multiple strategies"""
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
    
    def sliding_window_perplexity(self, 
                                  text: str,
                                  max_len: int = 2048,
                                  stride: int = 1024,
                                  show_progress: bool = False) -> Dict[str, float]:
        """Enhanced sliding window perplexity with additional metrics"""
        try:
            enc = self.tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = enc["input_ids"].to(self.device)
            
            if input_ids.size(1) == 0:
                return {"perplexity": float('inf'), "avg_loss": float('inf'), "token_count": 0}
            
            nlls = []
            token_losses = []
            seq_len = input_ids.size(1)
            
            windows = list(range(0, seq_len, stride))
            if show_progress:
                windows = tqdm(windows, desc="Computing perplexity")
            
            for i in windows:
                begin = i
                end = min(i + max_len, seq_len)
                trg_len = end - i
                
                input_ids_slice = input_ids[:, begin:end]
                
                with torch.no_grad():
                    outputs = self.model(input_ids_slice, labels=input_ids_slice)
                    loss = outputs.loss
                    
                    # Store individual token losses for analysis
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids_slice[..., 1:].contiguous()
                        
                        # Calculate per-token losses
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        token_losses_batch = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        token_losses.extend(token_losses_batch.cpu().tolist())
                    
                    nll = loss * trg_len
                    nlls.append(nll)
                
                if end == seq_len:
                    break
            
            # Calculate metrics
            total_nll = torch.stack(nlls).sum()
            perplexity = torch.exp(total_nll / seq_len)
            avg_loss = total_nll / seq_len
            
            # Additional metrics
            token_losses_array = np.array(token_losses) if token_losses else np.array([float('inf')])
            
            return {
                "perplexity": float(perplexity),
                "avg_loss": float(avg_loss),
                "token_count": seq_len,
                "min_token_loss": float(np.min(token_losses_array)),
                "max_token_loss": float(np.max(token_losses_array)),
                "std_token_loss": float(np.std(token_losses_array)),
                "median_token_loss": float(np.median(token_losses_array))
            }
            
        except Exception as e:
            print(f"❌ Error calculating perplexity: {e}")
            return {"perplexity": float('inf'), "avg_loss": float('inf'), "token_count": 0}
    
    def batch_perplexity(self, 
                        texts: List[str], 
                        max_len: int = 2048,
                        stride: int = 1024,
                        show_progress: bool = True) -> List[Dict[str, float]]:
        """Calculate perplexity for multiple texts with progress tracking"""
        results = []
        
        texts_iter = tqdm(texts, desc="Processing texts") if show_progress else texts
        
        for text in texts_iter:
            result = self.sliding_window_perplexity(text, max_len, stride, show_progress=False)
            results.append(result)
        
        return results


class EnhancedAIDetector:
    """Enhanced AI detector with multiple metrics and improved threshold selection"""
    
    def __init__(self, perplexity_calc: PerplexityCalculator):
        self.perplexity_calc = perplexity_calc
        self.threshold = None
        self.calibration_data = None
        
    def calibrate_threshold(self, 
                          texts: List[str], 
                          labels: List[int],
                          metric: str = 'f1',
                          show_analysis: bool = True) -> Dict[str, Any]:
        """Calibrate detection threshold with comprehensive analysis"""
        print(f"🎯 Calibrating threshold using {len(texts)} samples...")
        
        # Calculate perplexities
        perplexity_results = self.perplexity_calc.batch_perplexity(texts, show_progress=True)
        perplexities = [r['perplexity'] for r in perplexity_results]
        
        # Store calibration data
        self.calibration_data = {
            'perplexities': perplexities,
            'labels': labels,
            'results': perplexity_results
        }
        
        # Find optimal threshold
        threshold_results = self._optimize_threshold(perplexities, labels, metric)
        self.threshold = threshold_results['best_threshold']
        
        if show_analysis:
            self._show_calibration_analysis(perplexities, labels, threshold_results)
        
        return threshold_results
    
    def _optimize_threshold(self, 
                           perplexities: List[float], 
                           labels: List[int], 
                           metric: str) -> Dict[str, Any]:
        """Optimize threshold using various metrics"""
        # Filter out infinite perplexities
        valid_indices = [i for i, p in enumerate(perplexities) if not math.isinf(p)]
        valid_perplexities = [perplexities[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        
        if not valid_perplexities:
            return {'best_threshold': 10.0, 'best_scores': {}, 'all_results': []}
        
        # Generate threshold candidates
        candidates = sorted(set(valid_perplexities))
        # Add some intermediate values
        extended_candidates = candidates.copy()
        for i in range(len(candidates) - 1):
            mid = (candidates[i] + candidates[i + 1]) / 2
            extended_candidates.append(mid)
        
        candidates = sorted(extended_candidates)
        
        best_score = -1.0
        best_threshold = candidates[0] if candidates else 10.0
        best_scores = {}
        all_results = []
        
        for threshold in candidates:
            # Predict: AI if perplexity < threshold
            y_pred = [1 if p < threshold else 0 for p in valid_perplexities]
            
            # Calculate metrics
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    valid_labels, y_pred, average='binary', zero_division=0
                )
                accuracy = accuracy_score(valid_labels, y_pred)
                
                # Calculate AUC if possible
                try:
                    # Use negative perplexity as score (higher perplexity = more human-like)
                    scores = [-p for p in valid_perplexities]
                    auc = roc_auc_score(valid_labels, scores)
                except:
                    auc = 0.0
                
                result = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'auc': auc
                }
                all_results.append(result)
                
                # Select best based on chosen metric
                current_score = result[metric]
                if current_score > best_score:
                    best_score = current_score
                    best_threshold = threshold
                    best_scores = result
                    
            except Exception as e:
                print(f"⚠️ Error calculating metrics for threshold {threshold}: {e}")
                continue
        
        return {
            'best_threshold': best_threshold,
            'best_scores': best_scores,
            'all_results': all_results
        }
    
    def _show_calibration_analysis(self, 
                                  perplexities: List[float], 
                                  labels: List[int], 
                                  threshold_results: Dict[str, Any]):
        """Show calibration analysis with visualizations"""
        print(f"\n📊 Calibration Results:")
        print(f"🎯 Optimal Threshold: {threshold_results['best_threshold']:.3f}")
        
        best_scores = threshold_results['best_scores']
        print(f"📈 Performance Metrics:")
        print(f"  • Accuracy: {best_scores.get('accuracy', 0):.3f}")
        print(f"  • Precision: {best_scores.get('precision', 0):.3f}")
        print(f"  • Recall: {best_scores.get('recall', 0):.3f}")
        print(f"  • F1-Score: {best_scores.get('f1', 0):.3f}")
        print(f"  • AUC: {best_scores.get('auc', 0):.3f}")
        
        # Create visualizations
        self._plot_calibration_analysis(perplexities, labels, threshold_results)
    
    def _plot_calibration_analysis(self, 
                                  perplexities: List[float], 
                                  labels: List[int], 
                                  threshold_results: Dict[str, Any]):
        """Create calibration analysis plots"""
        # Filter valid perplexities for plotting
        valid_data = [(p, l) for p, l in zip(perplexities, labels) if not math.isinf(p)]
        if not valid_data:
            print("⚠️ No valid data for plotting")
            return
            
        valid_perplexities, valid_labels = zip(*valid_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Perplexity distribution by class
        ax1 = axes[0, 0]
        human_perp = [p for p, l in zip(valid_perplexities, valid_labels) if l == 0]
        ai_perp = [p for p, l in zip(valid_perplexities, valid_labels) if l == 1]
        
        ax1.hist(human_perp, alpha=0.7, label='Human', bins=30, color='blue')
        ax1.hist(ai_perp, alpha=0.7, label='AI/Hybrid', bins=30, color='red')
        ax1.axvline(threshold_results['best_threshold'], color='green', linestyle='--', 
                   label=f'Threshold: {threshold_results["best_threshold"]:.2f}')
        ax1.set_xlabel('Perplexity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Perplexity Distribution by Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Threshold optimization curve
        ax2 = axes[0, 1]
        all_results = threshold_results['all_results']
        if all_results:
            thresholds = [r['threshold'] for r in all_results]
            f1_scores = [r['f1'] for r in all_results]
            accuracies = [r['accuracy'] for r in all_results]
            
            ax2.plot(thresholds, f1_scores, label='F1-Score', marker='o', markersize=3)
            ax2.plot(thresholds, accuracies, label='Accuracy', marker='s', markersize=3)
            ax2.axvline(threshold_results['best_threshold'], color='green', linestyle='--', 
                       label='Best Threshold')
            ax2.set_xlabel('Threshold')
            ax2.set_ylabel('Score')
            ax2.set_title('Threshold Optimization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix
        ax3 = axes[1, 0]
        y_pred = [1 if p < threshold_results['best_threshold'] else 0 for p in valid_perplexities]
        cm = confusion_matrix(valid_labels, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Human', 'AI/Hybrid'],
                   yticklabels=['Human', 'AI/Hybrid'])
        ax3.set_title('Confusion Matrix')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')
        
        # Plot 4: ROC Curve (if possible)
        ax4 = axes[1, 1]
        try:
            scores = [-p for p in valid_perplexities]  # Negative perplexity as score
            fpr, tpr, _ = roc_curve(valid_labels, scores)
            auc_score = roc_auc_score(valid_labels, scores)
            
            ax4.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curve')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except Exception as e:
            ax4.text(0.5, 0.5, f'ROC curve unavailable\n{str(e)}', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('ROC Curve (Unavailable)')
        
        plt.tight_layout()
        plt.show()
    
    def detect(self, text: str) -> DetectionResult:
        """Detect if text is AI-generated"""
        if self.threshold is None:
            raise ValueError("Detector not calibrated. Call calibrate_threshold() first.")
        
        # Calculate perplexity and additional metrics
        metrics = self.perplexity_calc.sliding_window_perplexity(text)
        perplexity = metrics['perplexity']
        
        # Make prediction
        prediction = 1 if perplexity < self.threshold else 0
        prediction_label = "AI/Hybrid" if prediction == 1 else "Human"
        
        # Calculate confidence based on distance from threshold
        distance_from_threshold = abs(perplexity - self.threshold)
        max_distance = max(abs(min(self.calibration_data['perplexities']) - self.threshold),
                          abs(max(self.calibration_data['perplexities']) - self.threshold))
        confidence = min(distance_from_threshold / max_distance, 1.0) if max_distance > 0 else 0.5
        
        return DetectionResult(
            perplexity=perplexity,
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            threshold=self.threshold,
            additional_metrics=metrics
        )


class DataLoader:
    """Enhanced data loader with validation and preprocessing"""
    
    @staticmethod
    def load_json_dataset(path: str) -> Dict[str, Any]:
        """Load and validate JSON dataset"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded dataset from {path}")
            DataLoader._print_dataset_info(data)
            return data
            
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return {}
    
    @staticmethod
    def _print_dataset_info(data: Dict[str, Any]):
        """Print dataset information"""
        if isinstance(data, dict):
            print(f"📊 Dataset keys: {list(data.keys())}")
            
            # Check for submissions
            if 'submissions' in data:
                submissions = data['submissions']
                print(f"📝 Number of submissions: {len(submissions)}")
                
                # Analyze labels
                if submissions:
                    labels = [s.get('label', '').lower() for s in submissions]
                    label_counts = {}
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    
                    print(f"🏷️ Label distribution: {label_counts}")
                    
                    # Check text lengths
                    text_lengths = [len(s.get('text', '')) for s in submissions]
                    if text_lengths:
                        print(f"📏 Text length stats:")
                        print(f"  • Min: {min(text_lengths)} chars")
                        print(f"  • Max: {max(text_lengths)} chars")
                        print(f"  • Mean: {statistics.mean(text_lengths):.1f} chars")
                        print(f"  • Median: {statistics.median(text_lengths):.1f} chars")
    
    @staticmethod
    def map_label(raw_label: str) -> int:
        """Enhanced label mapping with validation"""
        if not isinstance(raw_label, str):
            print(f"⚠️ Invalid label type: {type(raw_label)}")
            return 0
        
        raw = raw_label.strip().lower()
        
        if raw in ('ai', 'hybrid'):
            return 1
        elif raw == 'human':
            return 0
        else:
            print(f"⚠️ Unknown label: '{raw_label}', defaulting to Human (0)")
            return 0
    
    @staticmethod
    def extract_texts_and_labels(data: Dict[str, Any]) -> Tuple[List[str], List[int]]:
        """Extract texts and labels from dataset with validation"""
        if 'submissions' not in data:
            print("❌ No 'submissions' key found in dataset")
            return [], []
        
        submissions = data['submissions']
        texts = []
        labels = []
        
        for i, submission in enumerate(submissions):
            # Validate submission structure
            if not isinstance(submission, dict):
                print(f"⚠️ Skipping invalid submission at index {i}")
                continue
            
            text = submission.get('text', '')
            raw_label = submission.get('label', '')
            
            # Validate text
            if not text or not isinstance(text, str):
                print(f"⚠️ Skipping submission {i} with invalid text")
                continue
            
            # Clean and validate text
            text = text.strip()
            if len(text) < 10:  # Skip very short texts
                print(f"⚠️ Skipping submission {i} with text too short ({len(text)} chars)")
                continue
            
            label = DataLoader.map_label(raw_label)
            
            texts.append(text)
            labels.append(label)
        
        print(f"✅ Extracted {len(texts)} valid samples")
        return texts, labels


class EnhancedFeedbackGenerator:
    """Enhanced feedback generator with better prompting and error handling"""
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, config: InferenceConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def generate_feedback(self, 
                         essay: str, 
                         rubric: Dict[str, Any],
                         detection_result: DetectionResult,
                         max_new_tokens: Optional[int] = None) -> str:
        """Generate enhanced feedback with context awareness"""
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        try:
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(essay, rubric, detection_result)
            
            # Generate feedback
            feedback = self._generate_text(prompt, max_tokens)
            
            # Post-process feedback
            feedback = self._post_process_feedback(feedback)
            
            return feedback
            
        except Exception as e:
            print(f"❌ Error generating feedback: {e}")
            return f"Error generating feedback: {str(e)}"
    
    def _build_enhanced_prompt(self, 
                              essay: str, 
                              rubric: Dict[str, Any],
                              detection_result: DetectionResult) -> str:
        """Build enhanced prompt with rubric and detection context"""
        # Extract rubric criteria
        criteria_lines = []
        criteria = rubric.get('criteria', [])
        
        for criterion in criteria:
            if isinstance(criterion, dict):
                name = criterion.get('name', 'Unknown')
                desc = criterion.get('description', 'No description')
                criteria_lines.append(f"• {name}: {desc}")
        
        # Build context-aware prompt
        rubric_id = rubric.get('rubric_id', 'Unknown')
        domain = rubric.get('domain', 'Academic Writing')
        
        # Add AI detection context if relevant
        ai_context = ""
        if detection_result.prediction == 1:  # AI detected
            ai_context = f"\n\nNOTE: This essay has been flagged as potentially AI-generated (confidence: {detection_result.confidence:.2f}). Please provide feedback that addresses both content quality and originality concerns."
        
        prompt = f"""You are an expert academic writing instructor providing detailed, constructive feedback. Analyze the student's essay according to the provided rubric and offer specific, actionable suggestions for improvement.

RUBRIC ({rubric_id} - {domain}):
{chr(10).join(criteria_lines)}

INSTRUCTIONS:
1. Evaluate the essay against each rubric criterion
2. Highlight specific strengths and areas for improvement
3. Provide concrete, actionable suggestions
4. Maintain a supportive and professional tone
5. Focus on helping the student develop their writing skills{ai_context}

STUDENT ESSAY:
{essay.strip()}

DETAILED FEEDBACK:
"""
        
        return prompt
    
    def _generate_text(self, prompt: str, max_new_tokens: int) -> str:
        """Generate text with optimized parameters"""
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens
        ).to(self.device)
        
        # Generate with enhanced parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        feedback = generated_text[len(prompt):].strip()
        
        return feedback
    
    def _post_process_feedback(self, feedback: str) -> str:
        """Post-process generated feedback"""
        # Remove any remaining prompt artifacts
        feedback = feedback.split("DETAILED FEEDBACK:")[-1].strip()
        
        # Clean up common generation artifacts
        lines = feedback.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines at the beginning
            if not cleaned_lines and not line:
                continue
            # Stop at certain end markers
            if line.startswith(('USER:', 'HUMAN:', 'ASSISTANT:', '<|')):
                break
            cleaned_lines.append(line)
        
        # Join and clean up
        feedback = '\n'.join(cleaned_lines).strip()
        
        # Ensure minimum feedback length
        if len(feedback) < 50:
            feedback += "\n\nPlease consider revising your essay with attention to the rubric criteria for improved quality."
        
        return feedback


class ComprehensiveAnalyzer:
    """Comprehensive analyzer combining AI detection and feedback generation"""
    
    def __init__(self, 
                 ai_detector: EnhancedAIDetector, 
                 feedback_generator: EnhancedFeedbackGenerator):
        self.ai_detector = ai_detector
        self.feedback_generator = feedback_generator
    
    def analyze_essay(self, 
                     essay: str, 
                     rubric: Dict[str, Any],
                     generate_feedback: bool = True,
                     show_details: bool = True) -> Dict[str, Any]:
        """Comprehensive essay analysis"""
        print("🔍 Starting comprehensive essay analysis...")
        
        # Validate inputs
        if not essay or not essay.strip():
            return {"error": "Empty essay provided"}
        
        if not rubric:
            print("⚠️ No rubric provided, using default")
            rubric = self._get_default_rubric()
        
        try:
            # Step 1: AI Detection
            if show_details:
                print("\n🤖 Performing AI detection...")
            
            detection_result = self.ai_detector.detect(essay)
            
            if show_details:
                print(f"📊 Detection Results:")
                print(f"  • Perplexity: {detection_result.perplexity:.3f}")
                print(f"  • Threshold: {detection_result.threshold:.3f}")
                print(f"  • Prediction: {detection_result.prediction_label}")
                print(f"  • Confidence: {detection_result.confidence:.3f}")
            
            # Step 2: Generate Feedback (if requested)
            feedback = ""
            if generate_feedback:
                if show_details:
                    print("\n📝 Generating feedback...")
                
                feedback = self.feedback_generator.generate_feedback(
                    essay, rubric, detection_result
                )
                
                if show_details:
                    print("✅ Feedback generated")
            
            # Step 3: Compile Results
            analysis_result = {
                "essay_stats": {
                    "character_count": len(essay),
                    "word_count": len(essay.split()),
                    "sentence_count": essay.count('.') + essay.count('!') + essay.count('?')
                },
                "ai_detection": {
                    "prediction": detection_result.prediction_label,
                    "confidence": detection_result.confidence,
                    "perplexity": detection_result.perplexity,
                    "threshold": detection_result.threshold,
                    "additional_metrics": detection_result.additional_metrics
                },
                "feedback": feedback,
                "rubric_used": rubric,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if show_details:
                print("\n✅ Analysis completed successfully")
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def _get_default_rubric(self) -> Dict[str, Any]:
        """Get default rubric for analysis"""
        return {
            "rubric_id": "default_academic",
            "domain": "Academic Writing",
            "criteria": [
                {
                    "name": "Content & Ideas",
                    "description": "Depth of understanding, relevance of ideas, and originality of thought"
                },
                {
                    "name": "Organization & Structure",
                    "description": "Logical flow, clear introduction/conclusion, effective transitions"
                },
                {
                    "name": "Evidence & Support",
                    "description": "Use of credible sources, integration of evidence, proper citations"
                },
                {
                    "name": "Writing Quality",
                    "description": "Grammar, syntax, vocabulary, clarity of expression"
                },
                {
                    "name": "Critical Thinking",
                    "description": "Analysis, evaluation, synthesis of ideas, argumentation"
                }
            ]
        }
    
    def batch_analyze(self, 
                     essays: List[str], 
                     rubric: Dict[str, Any],
                     generate_feedback: bool = True,
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """Analyze multiple essays in batch"""
        results = []
        
        essays_iter = tqdm(essays, desc="Analyzing essays") if show_progress else essays
        
        for i, essay in enumerate(essays_iter):
            print(f"\n📄 Analyzing essay {i+1}/{len(essays)}")
            result = self.analyze_essay(
                essay, rubric, generate_feedback, show_details=False
            )
            result["essay_id"] = i + 1
            results.append(result)
        
        return results
    
    def print_analysis_summary(self, analysis_result: Dict[str, Any]):
        """Print formatted analysis summary"""
        if "error" in analysis_result:
            print(f"❌ Analysis Error: {analysis_result['error']}")
            return
        
        print("\n" + "="*60)
        print("📊 ESSAY ANALYSIS SUMMARY")
        print("="*60)
        
        # Essay Stats
        stats = analysis_result.get("essay_stats", {})
        print(f"📏 Essay Statistics:")
        print(f"  • Characters: {stats.get('character_count', 0):,}")
        print(f"  • Words: {stats.get('word_count', 0):,}")
        print(f"  • Sentences: {stats.get('sentence_count', 0):,}")
        
        # AI Detection
        ai_detection = analysis_result.get("ai_detection", {})
        prediction = ai_detection.get("prediction", "Unknown")
        confidence = ai_detection.get("confidence", 0)
        perplexity = ai_detection.get("perplexity", 0)
        
        print(f"\n🤖 AI Detection:")
        print(f"  • Classification: {prediction}")
        print(f"  • Confidence: {confidence:.3f}")
        print(f"  • Perplexity: {perplexity:.3f}")
        
        # Feedback
        feedback = analysis_result.get("feedback", "")
        if feedback:
            print(f"\n📝 Generated Feedback:")
            print("-" * 40)
            print(feedback)
            print("-" * 40)
        
        print(f"\n⏰ Analysis completed at: {analysis_result.get('timestamp', 'Unknown')}")
        print("="*60)


def save_analysis_results(results: List[Dict[str, Any]], filename: str):
    """Save analysis results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Results saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def load_analysis_results(filename: str) -> List[Dict[str, Any]]:
    """Load analysis results from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✅ Results loaded from {filename}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return []


def export_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """Export analysis results to CSV format"""
    try:
        # Flatten results for CSV export
        flattened_results = []
        
        for result in results:
            if "error" in result:
                continue
                
            flat_result = {
                "essay_id": result.get("essay_id", "N/A"),
                "character_count": result.get("essay_stats", {}).get("character_count", 0),
                "word_count": result.get("essay_stats", {}).get("word_count", 0),
                "sentence_count": result.get("essay_stats", {}).get("sentence_count", 0),
                "ai_prediction": result.get("ai_detection", {}).get("prediction", "Unknown"),
                "ai_confidence": result.get("ai_detection", {}).get("confidence", 0),
                "perplexity": result.get("ai_detection", {}).get("perplexity", 0),
                "timestamp": result.get("timestamp", "Unknown")
            }
            flattened_results.append(flat_result)
        
        # Create DataFrame and save
        df = pd.DataFrame(flattened_results)
        df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")
        
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")


def monitor_performance():
    """Monitor system performance and provide optimization suggestions"""
    print("📊 PERFORMANCE MONITORING")
    print("="*50)
    
    # GPU Memory
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🖥️ GPU Memory Usage:")
        print(f"  • Current: {current_memory:.2f} GB")
        print(f"  • Peak: {max_memory:.2f} GB")
        print(f"  • Total Available: {total_memory:.2f} GB")
        print(f"  • Utilization: {(current_memory/total_memory)*100:.1f}%")
        
        if current_memory / total_memory > 0.8:
            print("⚠️ High GPU memory usage detected")
            print("💡 Suggestions:")
            print("  • Use smaller batch sizes")
            print("  • Enable gradient checkpointing")
            print("  • Use 8-bit or 4-bit quantization")
    
    print(f"\n💡 OPTIMIZATION TIPS:")
    print(f"  • Use 4-bit quantization for memory efficiency")
    print(f"  • Batch process multiple texts for better throughput")
    print(f"  • Cache model outputs when analyzing similar texts")
    print(f"  • Use gradient accumulation for larger effective batch sizes")
    print(f"  • Consider using mixed precision training/inference")
    
    print("="*50)


def cleanup_resources():
    """Clean up GPU memory and resources"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 Resources cleaned up")


def main():
    """Main execution function"""
    print("🚀 Phi-2 Model Testing - Improved Version")
    print("="*50)
    
    # Initialize configurations
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    
    # Get device
    device = get_optimal_device()
    
    # Load model
    print("\n📥 Loading Phi-2 model...")
    loader = OptimizedPhi2Loader(model_config)
    tokenizer, model = loader.load_model()
    
    # Initialize components
    perplexity_calc = PerplexityCalculator(tokenizer, model)
    ai_detector = EnhancedAIDetector(perplexity_calc)
    feedback_generator = EnhancedFeedbackGenerator(tokenizer, model, inference_config)
    
    # Load dataset
    dataset_path = "/workspace/Deprecated/rub_psychology_0011.json"
    print(f"\n📂 Loading dataset from {dataset_path}...")
    data = DataLoader.load_json_dataset(dataset_path)
    
    if data:
        texts, labels = DataLoader.extract_texts_and_labels(data)
    else:
        print("🔧 Using sample data for demonstration...")
        texts = [
            "Artificial intelligence is rapidly transforming various industries. Its impact on healthcare is particularly noteworthy, enabling faster and more accurate diagnoses. AI-powered tools are also revolutionizing the financial sector through algorithmic trading and fraud detection. The potential benefits are immense, but ethical considerations regarding job displacement and bias in algorithms must be addressed. Governments and organizations need to collaborate to ensure responsible development and deployment of AI technologies.",
            "I think that psychology is a really interesting field because it helps us understand how people think and feel. When I was in high school, I had this teacher who was really good at explaining complex concepts in simple ways. She made me realize that I wanted to study psychology in college. Now I'm learning about different theories and research methods. It's challenging but also very rewarding. I hope to become a therapist someday and help people work through their problems.",
            "The study of cognitive psychology encompasses various mental processes including perception, memory, attention, and problem-solving. Research in this field utilizes experimental methodologies to investigate how individuals process information. Contemporary theories emphasize the role of neural networks and computational models in understanding cognitive functions. The integration of neuroscience and psychology has led to significant advances in our comprehension of brain-behavior relationships.",
            "My experience with group therapy has been transformative. Initially, I was hesitant to share personal struggles with strangers, but the supportive environment fostered genuine connections. The facilitator skillfully guided discussions, ensuring everyone felt heard. Through listening to others' stories, I gained new perspectives on my own challenges. The group became a safe space for vulnerability and growth. I learned coping strategies that I continue to use today."
        ]
        labels = [1, 0, 1, 0]  # AI, Human, AI, Human
    
    if texts and labels:
        # Calibrate AI detector
        print("\n🎯 Calibrating AI detector...")
        calibration_results = ai_detector.calibrate_threshold(
            texts=texts,
            labels=labels,
            metric='f1',
            show_analysis=True
        )
        
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveAnalyzer(ai_detector, feedback_generator)
        
        # Test with sample essay
        test_essay = """
        The field of psychology has undergone significant transformations throughout its history, evolving from philosophical speculation to empirical science. Contemporary psychological research employs rigorous methodologies to investigate human behavior and mental processes. The integration of neuroscience and psychology has particularly enhanced our understanding of the biological bases of behavior.
        
        Cognitive psychology, as a major subdiscipline, focuses on mental processes such as perception, memory, attention, and problem-solving. Research in this area utilizes experimental designs to test hypotheses about how individuals process information. The development of cognitive models has provided frameworks for understanding complex mental operations.
        
        Furthermore, the application of psychological principles in clinical settings has demonstrated the practical value of psychological research. Evidence-based therapeutic interventions have shown efficacy in treating various mental health conditions. The continued advancement of psychological science promises further insights into human nature and behavior.
        """
        
        # Create sample rubric
        sample_rubric = {
            "rubric_id": "psychology_essay_v1",
            "domain": "Psychology",
            "criteria": [
                {
                    "name": "Theoretical Understanding",
                    "description": "Demonstrates clear understanding of psychological concepts and theories"
                },
                {
                    "name": "Critical Analysis",
                    "description": "Shows ability to analyze and evaluate psychological concepts and evidence"
                },
                {
                    "name": "Evidence Integration",
                    "description": "Effectively integrates credible psychological research to support arguments"
                },
                {
                    "name": "Academic Writing",
                    "description": "Uses appropriate academic style, terminology, and APA formatting"
                }
            ]
        }
        
        print("\n🔍 Running comprehensive analysis on test essay...")
        
        # Perform analysis
        analysis_result = analyzer.analyze_essay(
            essay=test_essay,
            rubric=sample_rubric,
            generate_feedback=True,
            show_details=True
        )
        
        # Display results
        analyzer.print_analysis_summary(analysis_result)
        
        # Monitor performance
        print("\n" + "="*50)
        monitor_performance()
        
        print("\n✅ Analysis completed successfully!")
        print("\n📋 Available Functions:")
        print("  • analyzer.analyze_essay() - Comprehensive essay analysis")
        print("  • analyzer.batch_analyze() - Batch processing multiple essays")
        print("  • save_analysis_results() - Save results to JSON")
        print("  • export_results_to_csv() - Export results to CSV")
        print("  • cleanup_resources() - Clean up GPU memory")
        print("  • monitor_performance() - Check system performance")
        
    else:
        print("❌ Cannot proceed without valid data")
    
    # Clean up resources
    cleanup_resources()


if __name__ == "__main__":
    main()