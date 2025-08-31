import os
import json
import time
from typing import List, Dict, Any, Optional

import requests


class ModelClient:
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        raise NotImplementedError


class HFInferenceClient(ModelClient):
    """
    Lightweight wrapper for Hugging Face Inference API.
    Requires env HF_API_TOKEN. Good for hosted models (fast start, no local weights).
    """

    def __init__(self, model_id: str = "microsoft/phi-2", timeout: int = 120):
        self.model_id = model_id
        self.api_token = os.environ.get("HF_API_TOKEN", "").strip()
        if not self.api_token:
            raise RuntimeError("HF_API_TOKEN env var is required for HFInferenceClient")
        self.url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        self.timeout = timeout

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True}
        }
        resp = requests.post(self.url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # HF returns list of dicts with 'generated_text'
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        # Some models return a dict with 'generated_text'
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # Fallback string
        return json.dumps(data)


class TransformersClient(ModelClient):
    """
    Local inference via transformers. Attempts CPU inference by default; supports device_map="auto" if available.
    Suitable for small evals. For larger runs, prefer HFInferenceClient or quantized local backends.
    """

    def __init__(self, model_id: str = "microsoft/phi-2", device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use float16 on CUDA if available, else bfloat16/float32
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        from transformers import GenerationConfig
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=generation_config)
        # Return only the newly generated tokens
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text


class OllamaClient(ModelClient):
    """
    Local inference via Ollama REST API.
    Start server: `ollama serve`. Pull model: `ollama pull phi:2` (or any compatible model name).
    """

    def __init__(self, model_id: str = "phi:2", host: str = "http://localhost:11434"):
        self.model_id = model_id
        self.url = f"{host}/api/generate"

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens
            },
            "stream": False
        }
        resp = requests.post(self.url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


class MockClient(ModelClient):
    """
    Deterministic stub for pipeline validation without real model calls.
    - If prompt contains 'Academic Integrity Detector', returns a simple JSON classification.
    - If prompt contains 'Rubric-Aligned Feedback', returns a structured feedback JSON.
    - If prompt contains 'Rate on a 1-5 scale', returns the string of an integer.
    """

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        p = prompt.lower()
        if "academic integrity detector" in p:
            # naive heuristic: if 'as a' appears, call it AI; else Human; else Hybrid
            label = "AI" if "as a" in p else "Human"
            return json.dumps({
                "label": label,
                "rationale": ["mock rationale 1", "mock rationale 2"],
                "flags": ["generic_phrasing"]
            })
        if "rubric-aligned feedback".lower() in p or "Provide actionable feedback aligned to the rubric".lower() in p:
            return json.dumps({
                "overall_summary": "Mock summary.",
                "criteria_feedback": [
                    {
                        "criterion_id": "c1",
                        "rating": "good",
                        "evidence": ["Example evidence 1"],
                        "improvement_tip": "Add more specifics."
                    }
                ],
                "suggested_grade": "B"
            })
        if "Rate on a 1-5 scale".lower() in p or "RATING (1-5):".lower() in p:
            return "4"
        return "{}"


def get_client(backend: str, model_id: str) -> ModelClient:
    backend = backend.lower()
    if backend == "hf":
        return HFInferenceClient(model_id=model_id)
    if backend == "transformers":
        return TransformersClient(model_id=model_id)
    if backend == "ollama":
        return OllamaClient(model_id=model_id)
    if backend == "mock":
        return MockClient()
    raise ValueError(f"Unknown backend: {backend}")

