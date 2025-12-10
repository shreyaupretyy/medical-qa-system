"""
Simple Ollama Model Wrapper for Medical Reasoning

This module provides a clean interface to Ollama for medical question answering.
Uses plain Ollama models (e.g., llama3.1:8b) without specialized medical models.
"""

import requests
import time
from typing import Optional, Dict
from pathlib import Path


class OllamaModel:
    """
    Simple Ollama model wrapper for medical reasoning.
    
    Uses Ollama API for inference. Supports any Ollama model.
    
    Parameters:
        model_name: Ollama model name (e.g., "llama3.1:8b")
        ollama_url: Ollama API endpoint (default: http://localhost:11434/api/generate)
        temperature: Sampling temperature (0.1 for consistent medical answers)
        max_tokens: Maximum tokens to generate
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434/api/generate",
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        """
        Initialize Ollama model.
        
        Args:
            model_name: Ollama model identifier (e.g., "llama3.1:8b")
            ollama_url: Ollama API endpoint
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Check if Ollama is running
        self._check_ollama_health()
    
    def _check_ollama_health(self):
        """Check if Ollama is running and model is available."""
        try:
            health_url = self.ollama_url.replace("/api/generate", "/api/tags")
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if model is available
                if any(self.model_name in name for name in model_names):
                    print(f"[OK] Model {self.model_name} is available")
                else:
                    print(f"[WARN] Model {self.model_name} not found in Ollama")
                    print(f"[INFO] Available models: {', '.join(model_names[:5])}")
                    print(f"[INFO] To install: ollama pull {self.model_name}")
            else:
                print(f"[WARN] Ollama health check failed (status {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Cannot connect to Ollama: {e}")
            print(f"[INFO] Make sure Ollama is running: ollama serve")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text response
        """
        # Use provided parameters or defaults
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        
        # Build full prompt with system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Prepare request
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_toks
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Ollama API request failed: {e}")
            raise
    
    def __repr__(self):
        return f"OllamaModel(model_name='{self.model_name}', url='{self.ollama_url}')"

