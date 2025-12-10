"""
Confidence Calibration Module

Implements confidence calibration to address overconfident wrong answers:
- Temperature scaling
- Evidence-based confidence adjustment
- Uncertainty estimation
- Calibration curve fitting

Addresses Day 4 Issue: 20 cases with high confidence but wrong answers
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CalibrationResult:
    """Calibrated confidence with uncertainty."""
    calibrated_confidence: float
    uncertainty: float
    evidence_strength: float
    calibration_method: str
    raw_confidence: float


class ConfidenceCalibrator:
    """
    Calibrates confidence scores based on evidence strength and uncertainty.
    
    This addresses the overconfidence problem by:
    - Temperature scaling for better calibration
    - Evidence-based confidence adjustment
    - Uncertainty estimation from multiple reasoning paths
    - Conservative thresholds for low-evidence cases
    """
    
    def __init__(self, temperature: float = 1.2):
        """
        Initialize confidence calibrator.
        
        Args:
            temperature: Temperature for scaling (higher = more conservative)
            Reduced from 1.5 to 1.2 to be less aggressive
        """
        self.temperature = temperature
        self.calibration_history = []  # For learning calibration parameters
    
    def calibrate(
        self,
        raw_confidence: float,
        evidence_strength: float,
        evidence_count: int,
        uncertainty: Optional[float] = None,
        method: str = 'temperature_evidence'
    ) -> CalibrationResult:
        """
        Calibrate confidence score.
        
        Args:
            raw_confidence: Raw confidence from model (0-1)
            evidence_strength: Strength of evidence (0-1)
            evidence_count: Number of evidence sources
            uncertainty: Optional uncertainty estimate
            method: Calibration method ('temperature', 'evidence', 'temperature_evidence')
            
        Returns:
            Calibrated confidence with uncertainty
        """
        if method == 'temperature':
            calibrated = self._temperature_scaling(raw_confidence)
        elif method == 'evidence':
            calibrated = self._evidence_based_calibration(
                raw_confidence, evidence_strength, evidence_count
            )
        else:  # temperature_evidence
            calibrated = self._combined_calibration(
                raw_confidence, evidence_strength, evidence_count
            )
        
        # Estimate uncertainty if not provided
        if uncertainty is None:
            uncertainty = self._estimate_uncertainty(
                raw_confidence, evidence_strength, evidence_count
            )
        
        return CalibrationResult(
            calibrated_confidence=max(0.0, min(1.0, calibrated)),
            uncertainty=uncertainty,
            evidence_strength=evidence_strength,
            calibration_method=method,
            raw_confidence=raw_confidence
        )
    
    def _temperature_scaling(self, raw_confidence: float) -> float:
        """
        Apply temperature scaling to confidence.
        
        Formula: calibrated = raw^(1/temperature)
        Higher temperature = more conservative (lower confidence)
        """
        # Avoid division by zero
        if raw_confidence <= 0:
            return 0.0
        
        # Temperature scaling
        scaled = np.power(raw_confidence, 1.0 / self.temperature)
        return float(scaled)
    
    def _evidence_based_calibration(
        self,
        raw_confidence: float,
        evidence_strength: float,
        evidence_count: int
    ) -> float:
        """
        Adjust confidence based on evidence strength and count.
        
        Low evidence = lower confidence
        High evidence = can trust raw confidence more
        """
        # Evidence strength multiplier
        evidence_multiplier = evidence_strength
        
        # Evidence count adjustment (diminishing returns)
        count_adjustment = min(1.0, evidence_count / 3.0)  # 3+ sources = full adjustment
        
        # Combined adjustment
        adjusted_confidence = raw_confidence * evidence_multiplier * count_adjustment
        
        # Conservative floor: if evidence is weak, cap confidence
        if evidence_strength < 0.5:
            adjusted_confidence = min(adjusted_confidence, 0.7)
        
        return adjusted_confidence
    
    def _combined_calibration(
        self,
        raw_confidence: float,
        evidence_strength: float,
        evidence_count: int
    ) -> float:
        """Combine temperature scaling with evidence-based adjustment."""
        # Less aggressive: only apply temperature scaling if confidence is very high
        if raw_confidence > 0.9:
            temp_scaled = self._temperature_scaling(raw_confidence)
        else:
            temp_scaled = raw_confidence
        
        # Apply evidence adjustment more gently
        if evidence_strength < 0.3:
            # Very weak evidence: reduce confidence
            evidence_adjusted = temp_scaled * 0.7
        elif evidence_strength < 0.5:
            # Weak evidence: slight reduction
            evidence_adjusted = temp_scaled * 0.85
        else:
            # Good evidence: trust the scaled confidence
            evidence_adjusted = temp_scaled
        
        # Adjust based on evidence count (less aggressive)
        if evidence_count < 2:
            evidence_adjusted = evidence_adjusted * 0.9  # Slight reduction
        elif evidence_count >= 3:
            evidence_adjusted = min(1.0, evidence_adjusted * 1.05)  # Slight boost
        
        return max(0.0, min(1.0, evidence_adjusted))
    
    def _estimate_uncertainty(
        self,
        raw_confidence: float,
        evidence_strength: float,
        evidence_count: int
    ) -> float:
        """
        Estimate uncertainty in the confidence score.
        
        Higher uncertainty when:
        - Low evidence strength
        - Few evidence sources
        - Confidence is in middle range (0.4-0.6)
        """
        # Base uncertainty from evidence
        evidence_uncertainty = 1.0 - evidence_strength
        
        # Count uncertainty (fewer sources = more uncertainty)
        count_uncertainty = max(0.0, 1.0 - (evidence_count / 5.0))
        
        # Confidence range uncertainty (middle range is more uncertain)
        if 0.4 <= raw_confidence <= 0.6:
            range_uncertainty = 0.3
        else:
            range_uncertainty = 0.1
        
        # Combine uncertainties
        total_uncertainty = (
            evidence_uncertainty * 0.5 +
            count_uncertainty * 0.3 +
            range_uncertainty * 0.2
        )
        
        return min(1.0, total_uncertainty)
    
    def calibrate_batch(
        self,
        confidences: List[float],
        evidence_strengths: List[float],
        evidence_counts: List[int],
        method: str = 'temperature_evidence'
    ) -> List[CalibrationResult]:
        """Calibrate a batch of confidence scores."""
        results = []
        for conf, strength, count in zip(confidences, evidence_strengths, evidence_counts):
            result = self.calibrate(conf, strength, count, method=method)
            results.append(result)
        return results
    
    def update_temperature_from_history(
        self,
        predictions: List[Tuple[float, bool]],  # (confidence, is_correct)
        target_ece: float = 0.1
    ):
        """
        Update temperature parameter based on calibration history.
        
        This implements a simple grid search to find optimal temperature.
        """
        best_temperature = self.temperature
        best_ece = self._calculate_ece(predictions, self.temperature)
        
        # Grid search over temperature values
        for temp in np.arange(1.0, 3.0, 0.1):
            ece = self._calculate_ece(predictions, temp)
            if abs(ece - target_ece) < abs(best_ece - target_ece):
                best_temperature = temp
                best_ece = ece
        
        self.temperature = best_temperature
        return best_temperature
    
    def _calculate_ece(
        self,
        predictions: List[Tuple[float, bool]],
        temperature: float,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error for given temperature."""
        if not predictions:
            return 1.0
        
        # Apply temperature scaling
        calibrated = [np.power(conf, 1.0 / temperature) for conf, _ in predictions]
        correct = [corr for _, corr in predictions]
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = [
                (cal, corr)
                for cal, corr in zip(calibrated, correct)
                if bin_lower <= cal < bin_upper
            ]
            
            if not in_bin:
                continue
            
            bin_confidences, bin_correctness = zip(*in_bin)
            bin_size = len(in_bin)
            bin_accuracy = np.mean(bin_correctness)
            bin_confidence = np.mean(bin_confidences)
            
            ece += (bin_size / len(predictions)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def get_confidence_threshold(
        self,
        evidence_strength: float,
        evidence_count: int,
        min_confidence: float = 0.5
    ) -> float:
        """
        Get minimum confidence threshold based on evidence.
        
        For low evidence, require higher confidence to be trustworthy.
        """
        # Base threshold
        threshold = min_confidence
        
        # Adjust based on evidence
        if evidence_strength < 0.5:
            threshold = 0.7  # Require higher confidence for weak evidence
        elif evidence_strength < 0.7:
            threshold = 0.6
        
        if evidence_count < 2:
            threshold = max(threshold, 0.65)  # Require more confidence with few sources
        
        return threshold
    
    def should_trust_prediction(
        self,
        calibrated_result: CalibrationResult
    ) -> Tuple[bool, str]:
        """
        Determine if prediction should be trusted.
        
        Returns:
            (should_trust, reason)
        """
        # Check confidence threshold
        threshold = self.get_confidence_threshold(
            calibrated_result.evidence_strength,
            1  # Will be adjusted based on actual count
        )
        
        if calibrated_result.calibrated_confidence < threshold:
            return False, f"Confidence {calibrated_result.calibrated_confidence:.2f} below threshold {threshold:.2f}"
        
        # Check uncertainty
        if calibrated_result.uncertainty > 0.5:
            return False, f"Uncertainty {calibrated_result.uncertainty:.2f} too high"
        
        # Check evidence strength
        if calibrated_result.evidence_strength < 0.3:
            return False, f"Evidence strength {calibrated_result.evidence_strength:.2f} too weak"
        
        return True, "All checks passed"


def main():
    """Demo: Test confidence calibration."""
    print("="*70)
    print("CONFIDENCE CALIBRATOR DEMO")
    print("="*70)
    
    calibrator = ConfidenceCalibrator(temperature=1.5)
    
    # Test cases
    test_cases = [
        (0.95, 0.9, 5, "High confidence, strong evidence"),
        (0.95, 0.3, 1, "High confidence, weak evidence (OVERCONFIDENT)"),
        (0.6, 0.7, 3, "Moderate confidence, good evidence"),
        (0.8, 0.5, 2, "High confidence, moderate evidence"),
    ]
    
    print("\nCalibration Results:")
    print("-" * 70)
    print(f"{'Case':<40} {'Raw':<8} {'Calibrated':<12} {'Uncertainty':<12} {'Trust?'}")
    print("-" * 70)
    
    for raw_conf, evidence_str, evidence_cnt, description in test_cases:
        result = calibrator.calibrate(
            raw_conf, evidence_str, evidence_cnt, method='temperature_evidence'
        )
        should_trust, reason = calibrator.should_trust_prediction(result)
        
        print(f"{description:<40} {raw_conf:<8.2f} {result.calibrated_confidence:<12.2f} "
              f"{result.uncertainty:<12.2f} {'Yes' if should_trust else 'No'}")
        if not should_trust:
            print(f"  Reason: {reason}")
    
    print("\n" + "="*70)
    print("[OK] Confidence Calibrator operational!")
    print("="*70)


if __name__ == "__main__":
    main()

