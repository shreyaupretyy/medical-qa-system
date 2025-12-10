"""
Configuration Loader for Multi-Stage RAG Pipeline

This module provides utilities for loading and managing pipeline configuration
from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import sys


class ConfigLoader:
    """Load and manage pipeline configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML config file (default: config/pipeline_config.yaml)
        """
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = str(base_dir / "config" / "pipeline_config.yaml")
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            print(f"[WARN] Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            return self.config
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'retrieval.stage1.k')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        retrieval = self.config.get('retrieval', {})
        return {
            'stage1_k': retrieval.get('stage1', {}).get('k', 20),
            'stage2_k': retrieval.get('stage2', {}).get('k', 10),
            'stage3_k': retrieval.get('stage3', {}).get('k', 5),
            'stage1_weight': retrieval.get('stage1', {}).get('weight', 0.3),
            'stage2_weight': retrieval.get('stage2', {}).get('weight', 0.3),
            'stage3_weight': retrieval.get('stage3', {}).get('weight', 0.4),
            'concept_first': retrieval.get('concept_first', False),
            'cross_encoder_name': retrieval.get('stage3', {}).get('model', 'cross-encoder/nli-deberta-v3-base'),
        }
    
    def get_reasoning_config(self) -> Dict[str, Any]:
        """Get reasoning configuration."""
        reasoning = self.config.get('reasoning', {})
        return {
            'evidence': reasoning.get('evidence', {}),
            'selection': reasoning.get('selection', {}),
            'medical_rules': reasoning.get('medical_rules', {})
        }
    
    def get_query_understanding_config(self) -> Dict[str, Any]:
        """Get query understanding configuration."""
        return self.config.get('query_understanding', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get('performance', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('models', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get data paths configuration."""
        return self.config.get('paths', {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'retrieval': {
                'stage1': {'k': 20, 'weight': 0.3},
                'stage2': {'k': 10, 'weight': 0.3},
                'stage3': {'k': 5, 'weight': 0.4}
            },
            'reasoning': {
                'evidence': {
                    'keyword_overlap_threshold': 0.3,
                    'strong_support_threshold': 0.6,
                    'weak_support_threshold': 0.1
                },
                'selection': {
                    'direct_match_boost': 1.2,
                    'contradiction_penalty': 0.7
                }
            },
            'paths': {
                'index_dir': 'data/indexes',
                'guidelines_path': 'data/raw/medical_guidelines.json',
                'clinical_cases_path': 'data/processed/clinical_cases.json'
            }
        }


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)

