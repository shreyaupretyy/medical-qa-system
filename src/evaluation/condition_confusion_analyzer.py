"""
Condition-Level Confusion Matrix Analyzer for Medical QA

This module provides confusion-matrix style error analysis that goes beyond 
simple A/B/C/D tracking to analyze medical condition confusion patterns.

Key Features:
- Standard answer confusion matrix (A/B/C/D)
- Condition-level confusion matrix (actual medical diagnoses)
- Medical similarity analysis (related vs unrelated errors)
- Top confusion pair identification
- Beautiful visualizations

This is an ANALYSIS-ONLY module - does not affect model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class ConditionConfusionAnalyzer:
    """
    Analyzes confusion patterns in medical QA predictions at both
    answer-level and condition-level.
    """
    
    def __init__(self, medical_similarity_groups: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the analyzer.
        
        Args:
            medical_similarity_groups: Optional dictionary grouping similar conditions
                Example: {
                    "respiratory": ["Pneumonia", "Bronchitis", "COPD"],
                    "cardiac": ["MI", "Angina", "Heart Failure"],
                }
        """
        self.medical_similarity_groups = medical_similarity_groups or self._default_similarity_groups()
        self.condition_to_group = self._build_condition_to_group_map()
    
    def _default_similarity_groups(self) -> Dict[str, List[str]]:
        """Default medical condition similarity groups."""
        return {
            "respiratory_infection": [
                "pneumonia", "bronchitis", "upper respiratory infection",
                "community acquired pneumonia", "viral pneumonia", "bacterial pneumonia"
            ],
            "respiratory_obstructive": [
                "copd", "asthma", "chronic bronchitis", "emphysema",
                "copd exacerbation", "asthma exacerbation", "acute asthma"
            ],
            "cardiac_ischemic": [
                "myocardial infarction", "acute mi", "stemi", "nstemi",
                "acute coronary syndrome", "unstable angina", "angina", "acs"
            ],
            "cardiac_failure": [
                "heart failure", "congestive heart failure", "chf",
                "acute decompensated heart failure", "pulmonary edema", "cardiogenic shock"
            ],
            "cardiac_arrhythmia": [
                "atrial fibrillation", "afib", "ventricular tachycardia",
                "supraventricular tachycardia", "bradycardia", "tachycardia"
            ],
            "stroke": [
                "ischemic stroke", "hemorrhagic stroke", "tia",
                "transient ischemic attack", "cva", "stroke", "cerebrovascular accident"
            ],
            "thrombotic": [
                "deep vein thrombosis", "dvt", "pulmonary embolism",
                "pe", "venous thromboembolism", "vte"
            ],
            "infection_systemic": [
                "sepsis", "septic shock", "bacteremia", "septicemia", "sirs"
            ],
            "infection_urinary": [
                "urinary tract infection", "uti", "cystitis",
                "pyelonephritis", "urethritis"
            ],
            "gastrointestinal_bleeding": [
                "gi bleeding", "upper gi bleed", "lower gi bleed",
                "peptic ulcer disease", "gastric ulcer", "variceal bleeding"
            ],
            "gastrointestinal_inflammatory": [
                "pancreatitis", "acute pancreatitis", "cholecystitis",
                "appendicitis", "diverticulitis"
            ],
            "renal": [
                "acute kidney injury", "aki", "chronic kidney disease",
                "ckd", "renal failure", "kidney failure"
            ],
            "metabolic_diabetes": [
                "diabetes", "type 2 diabetes", "diabetic ketoacidosis",
                "dka", "hypoglycemia", "hyperglycemia", "diabetes mellitus"
            ],
            "hepatic": [
                "liver cirrhosis", "cirrhosis", "hepatic encephalopathy",
                "hepatitis", "liver failure"
            ],
            "rheumatologic": [
                "rheumatoid arthritis", "ra", "osteoarthritis",
                "lupus", "sle", "osteoporosis"
            ],
            "hypertensive": [
                "hypertension", "hypertensive crisis", "htn",
                "hypertensive emergency", "high blood pressure"
            ],
            "psychiatric": [
                "depression", "major depressive disorder", "anxiety",
                "bipolar disorder", "schizophrenia"
            ]
        }
    
    def _build_condition_to_group_map(self) -> Dict[str, str]:
        """Build reverse mapping from condition to similarity group."""
        condition_map = {}
        for group_name, conditions in self.medical_similarity_groups.items():
            for condition in conditions:
                normalized = condition.lower().strip()
                condition_map[normalized] = group_name
        return condition_map
    
    def _normalize_condition(self, condition: str) -> str:
        """Normalize condition name for comparison."""
        if not condition:
            return "unknown"
        return condition.lower().strip()
    
    def _get_condition_group(self, condition: str) -> str:
        """Get similarity group for a condition."""
        normalized = self._normalize_condition(condition)
        return self.condition_to_group.get(normalized, "other")
    
    def _are_conditions_similar(self, condition1: str, condition2: str) -> bool:
        """Check if two conditions are medically similar (in same group)."""
        group1 = self._get_condition_group(condition1)
        group2 = self._get_condition_group(condition2)
        return group1 == group2 and group1 != "other"
    
    def analyze_confusion(
        self,
        gold_answers: List[str],
        predicted_answers: List[str],
        answer_to_condition: List[Dict[str, str]],
        question_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform comprehensive confusion analysis.
        
        Args:
            gold_answers: List of correct answers (e.g., ["A", "C", "B"])
            predicted_answers: List of model predictions (e.g., ["A", "B", "B"])
            answer_to_condition: List of dicts mapping answers to conditions
                Example: [
                    {"A": "Pneumonia", "B": "Bronchitis", "C": "COPD", "D": "Asthma"},
                    ...
                ]
            question_ids: Optional list of question IDs
            
        Returns:
            Dictionary with comprehensive confusion analysis
        """
        n = len(gold_answers)
        if len(predicted_answers) != n or len(answer_to_condition) != n:
            raise ValueError("All input lists must have the same length")
        
        if question_ids is None:
            question_ids = [f"Q_{i+1}" for i in range(n)]
        
        # 1. Build standard answer confusion matrix (A/B/C/D)
        answer_confusion = self._build_answer_confusion_matrix(
            gold_answers, predicted_answers
        )
        
        # 2. Build condition-level confusion matrix
        condition_confusion = self._build_condition_confusion_matrix(
            gold_answers, predicted_answers, answer_to_condition
        )
        
        # 3. Analyze medical similarity of errors
        similarity_analysis = self._analyze_medical_similarity(
            gold_answers, predicted_answers, answer_to_condition, question_ids
        )
        
        # 4. Identify top confusion pairs
        top_confusions = self._identify_top_confusion_pairs(
            condition_confusion, top_k=5
        )
        
        # 5. Calculate summary statistics
        summary = self._calculate_summary_statistics(
            gold_answers, predicted_answers, similarity_analysis
        )
        
        results = {
            "answer_confusion_matrix": answer_confusion,
            "condition_confusion_matrix": condition_confusion,
            "similarity_analysis": similarity_analysis,
            "top_confusion_pairs": top_confusions,
            "summary_statistics": summary,
            "total_questions": n
        }
        
        return results
    
    def _build_answer_confusion_matrix(
        self,
        gold_answers: List[str],
        predicted_answers: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Build standard confusion matrix for answer options."""
        all_answers = sorted(set(gold_answers + predicted_answers))
        
        confusion_matrix = {
            true_ans: {pred_ans: 0 for pred_ans in all_answers}
            for true_ans in all_answers
        }
        
        for true_ans, pred_ans in zip(gold_answers, predicted_answers):
            confusion_matrix[true_ans][pred_ans] += 1
        
        return confusion_matrix
    
    def _build_condition_confusion_matrix(
        self,
        gold_answers: List[str],
        predicted_answers: List[str],
        answer_to_condition: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix at condition level."""
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for gold_ans, pred_ans, mapping in zip(
            gold_answers, predicted_answers, answer_to_condition
        ):
            gold_condition = mapping.get(gold_ans, "Unknown")
            pred_condition = mapping.get(pred_ans, "Unknown")
            confusion_matrix[gold_condition][pred_condition] += 1
        
        return {
            true_cond: dict(pred_counts)
            for true_cond, pred_counts in confusion_matrix.items()
        }
    
    def _analyze_medical_similarity(
        self,
        gold_answers: List[str],
        predicted_answers: List[str],
        answer_to_condition: List[Dict[str, str]],
        question_ids: List[str]
    ) -> Dict:
        """Analyze whether errors are medically similar or unrelated."""
        similar_errors = []
        unrelated_errors = []
        correct_predictions = []
        
        for qid, gold_ans, pred_ans, mapping in zip(
            question_ids, gold_answers, predicted_answers, answer_to_condition
        ):
            gold_condition = mapping.get(gold_ans, "Unknown")
            pred_condition = mapping.get(pred_ans, "Unknown")
            
            if gold_ans == pred_ans:
                correct_predictions.append({
                    "question_id": qid,
                    "condition": gold_condition,
                    "answer": gold_ans
                })
            else:
                is_similar = self._are_conditions_similar(
                    gold_condition, pred_condition
                )
                
                error_info = {
                    "question_id": qid,
                    "true_condition": gold_condition,
                    "predicted_condition": pred_condition,
                    "true_answer": gold_ans,
                    "predicted_answer": pred_ans,
                    "condition_group_true": self._get_condition_group(gold_condition),
                    "condition_group_pred": self._get_condition_group(pred_condition)
                }
                
                if is_similar:
                    similar_errors.append(error_info)
                else:
                    unrelated_errors.append(error_info)
        
        total_errors = len(similar_errors) + len(unrelated_errors)
        
        return {
            "similar_condition_errors": similar_errors,
            "unrelated_condition_errors": unrelated_errors,
            "correct_predictions": correct_predictions,
            "similar_error_count": len(similar_errors),
            "unrelated_error_count": len(unrelated_errors),
            "total_errors": total_errors,
            "similar_error_rate": len(similar_errors) / total_errors if total_errors > 0 else 0,
            "unrelated_error_rate": len(unrelated_errors) / total_errors if total_errors > 0 else 0
        }
    
    def _identify_top_confusion_pairs(
        self,
        condition_confusion: Dict[str, Dict[str, int]],
        top_k: int = 5
    ) -> List[Tuple[str, str, int]]:
        """Identify most frequent confusion pairs."""
        confusion_pairs = []
        
        for true_cond, predictions in condition_confusion.items():
            for pred_cond, count in predictions.items():
                if true_cond != pred_cond and count > 0:
                    confusion_pairs.append((true_cond, pred_cond, count))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        return confusion_pairs[:top_k]
    
    def _calculate_summary_statistics(
        self,
        gold_answers: List[str],
        predicted_answers: List[str],
        similarity_analysis: Dict
    ) -> Dict:
        """Calculate summary statistics."""
        n = len(gold_answers)
        correct = sum(1 for g, p in zip(gold_answers, predicted_answers) if g == p)
        incorrect = n - correct
        
        return {
            "total_questions": n,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": correct / n if n > 0 else 0,
            "error_rate": incorrect / n if n > 0 else 0,
            "similar_condition_confusion_rate": similarity_analysis["similar_error_rate"],
            "unrelated_condition_confusion_rate": similarity_analysis["unrelated_error_rate"],
            "similar_condition_errors": similarity_analysis["similar_error_count"],
            "unrelated_condition_errors": similarity_analysis["unrelated_error_count"]
        }
    
    def visualize_confusion_matrix(
        self,
        results: Dict,
        output_path: str,
        show_condition_level: bool = True
    ) -> None:
        """
        Generate beautiful confusion matrix visualization.
        
        Args:
            results: Results dictionary from analyze_confusion()
            output_path: Path to save the PNG file
            show_condition_level: If True, show condition-level matrix; else answer-level
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Answer-level confusion matrix (A/B/C/D) - Predicted vs True
        answer_matrix = results["answer_confusion_matrix"]
        self._plot_matrix(
            ax,
            answer_matrix,
            "Confusion Matrix: Predicted vs True Answer",
            cmap='YlOrRd'
        )
        
        # Add title with summary stats
        summary = results["summary_statistics"]
        fig.suptitle(
            f"Medical QA Answer Confusion Matrix\n"
            f"Accuracy: {summary['accuracy']*100:.1f}% | "
            f"Total: {summary['total_questions']} | "
            f"Correct: {summary['correct_predictions']} | "
            f"Errors: {summary['incorrect_predictions']}",
            fontsize=16,
            fontweight='bold',
            y=0.96
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Confusion matrix saved to: {output_path}")
    
    def _plot_matrix(
        self,
        ax,
        matrix: Dict[str, Dict[str, int]],
        title: str,
        cmap: str = 'YlOrRd'
    ) -> None:
        """Plot confusion matrix heatmap."""
        if not matrix:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            return
        
        labels = sorted(set(matrix.keys()) | set(
            pred for preds in matrix.values() for pred in preds.keys()
        ))
        
        # Build matrix array
        matrix_array = np.zeros((len(labels), len(labels)))
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                matrix_array[i, j] = matrix.get(true_label, {}).get(pred_label, 0)
        
        # Plot heatmap
        sns.heatmap(
            matrix_array,
            annot=True,
            fmt='g',
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            ax=ax,
            linewidths=2,
            linecolor='#333333',
            square=True,
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        
        # Highlight diagonal (correct predictions) with green border
        for i in range(len(labels)):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1,
                fill=False,
                edgecolor='#2ecc71',
                lw=4,
                linestyle='-'
            ))
        
        ax.set_xlabel('Predicted Answer', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Answer', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    def _get_top_confused_conditions(
        self,
        condition_matrix: Dict[str, Dict[str, int]],
        top_k: int = 10
    ) -> List[Tuple[str, str, int]]:
        """Get top K most confused condition pairs."""
        confusions = []
        for true_cond, predictions in condition_matrix.items():
            for pred_cond, count in predictions.items():
                if true_cond != pred_cond and count > 0:
                    confusions.append((true_cond, pred_cond, count))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        return confusions[:top_k]
    
    def _plot_condition_confusions(
        self,
        ax,
        top_confusions: List[Tuple[str, str, int]],
        summary: Dict
    ) -> None:
        """Plot top condition confusion pairs as horizontal bar chart."""
        if not top_confusions:
            ax.text(0.5, 0.5, 'No condition confusions found', ha='center', va='center')
            ax.set_title('Condition-Level Confusion Analysis')
            return
        
        # Prepare data
        labels = [f"{true_c[:20]} → {pred_c[:20]}" for true_c, pred_c, _ in top_confusions]
        counts = [count for _, _, count in top_confusions]
        
        # Create horizontal bar chart
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(labels)))
        bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(
                count + 0.1,
                i,
                f'{count}',
                va='center',
                fontsize=11,
                fontweight='bold'
            )
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Confusion Count', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Top {len(labels)} Condition Confusions\n'
            f'Similar Errors: {summary["similar_condition_errors"]} | '
            f'Unrelated: {summary["unrelated_condition_errors"]}',
            fontsize=14,
            fontweight='bold',
            pad=10
        )
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    def print_analysis(self, results: Dict) -> None:
        """Print comprehensive analysis results."""
        print("=" * 80)
        print("MEDICAL QA CONDITION CONFUSION ANALYSIS")
        print("=" * 80)
        
        summary = results["summary_statistics"]
        print("\n[SUMMARY STATISTICS]")
        print("-" * 80)
        print(f"Total Questions:        {summary['total_questions']}")
        print(f"Correct Predictions:    {summary['correct_predictions']} ({summary['accuracy']*100:.1f}%)")
        print(f"Incorrect Predictions:  {summary['incorrect_predictions']} ({summary['error_rate']*100:.1f}%)")
        print(f"\nError Breakdown:")
        print(f"  Similar Conditions:   {summary['similar_condition_errors']} ({summary['similar_condition_confusion_rate']*100:.1f}% of errors)")
        print(f"  Unrelated Conditions: {summary['unrelated_condition_errors']} ({summary['unrelated_condition_confusion_rate']*100:.1f}% of errors)")
        
        print("\n[TOP 5 CONDITION CONFUSION PAIRS]")
        print("-" * 80)
        for i, (true_cond, pred_cond, count) in enumerate(results["top_confusion_pairs"], 1):
            is_similar = self._are_conditions_similar(true_cond, pred_cond)
            similar_tag = " [SIMILAR]" if is_similar else " [UNRELATED]"
            print(f"{i}. {true_cond:35s} → {pred_cond:35s} ({count} times){similar_tag}")
        
        print("\n" + "=" * 80)
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save analysis results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Condition analysis saved to: {output_path}")
