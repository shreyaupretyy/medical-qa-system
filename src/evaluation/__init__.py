"""
Comprehensive Evaluation Framework for Medical QA System

This package provides:
- Metrics calculation (retrieval, reasoning, safety)
- Error analysis and categorization
- Performance visualization and reporting
- Complete evaluation pipeline
"""

# Use relative imports to avoid module path issues
from .metrics_calculator import MetricsCalculator, RetrievalMetrics, ReasoningMetrics, MedicalSafetyMetrics
from .ground_truth_processor import GroundTruthProcessor
from .analyzer import ErrorAnalyzer, ErrorCategory, ConfusionMatrixEntry
from .visualizer import EvaluationVisualizer
from .pipeline import MedicalQAEvaluator

__all__ = [
    'MetricsCalculator',
    'RetrievalMetrics',
    'ReasoningMetrics',
    'MedicalSafetyMetrics',
    'GroundTruthProcessor',
    'ErrorAnalyzer',
    'ErrorCategory',
    'ConfusionMatrixEntry',
    'EvaluationVisualizer',
    'MedicalQAEvaluator'
]
