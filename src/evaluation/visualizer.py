"""
Visualization and Report Generation

This module creates:
- Performance dashboards
- Confusion matrix heatmaps
- Calibration plots
- Error analysis charts
- Comprehensive evaluation reports
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class EvaluationVisualizer:
    """
    Create visualizations and reports for evaluation results.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save reports and charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
    
    def plot_performance_summary(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create performance summary dashboard.
        
        Args:
            metrics: Dictionary with evaluation metrics
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall Accuracy
        ax1 = axes[0, 0]
        accuracy = metrics.get('reasoning', {}).get('exact_match_accuracy', 0.0)
        ax1.bar(['Overall Accuracy'], [accuracy], color='#2ecc71' if accuracy > 0.7 else '#e74c3c')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Answer Accuracy')
        ax1.text(0, accuracy + 0.02, f'{accuracy:.1%}', ha='center', fontweight='bold')
        
        # 2. Retrieval vs Reasoning Breakdown
        ax2 = axes[0, 1]
        retrieval_map = metrics.get('retrieval', {}).get('map_score', 0.0)
        reasoning_acc = metrics.get('reasoning', {}).get('exact_match_accuracy', 0.0)
        
        categories = ['Retrieval\n(MAP)', 'Reasoning\n(Accuracy)']
        values = [retrieval_map, reasoning_acc]
        colors = ['#3498db', '#9b59b6']
        
        bars = ax2.bar(categories, values, color=colors)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score')
        ax2.set_title('Retrieval vs Reasoning Performance')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.1%}', ha='center', fontweight='bold')
        
        # 3. Precision-Recall at different k
        ax3 = axes[1, 0]
        precision = metrics.get('retrieval', {}).get('precision_at_k', {})
        recall = metrics.get('retrieval', {}).get('recall_at_k', {})
        
        k_values = sorted([k for k in precision.keys()])
        prec_values = [precision.get(k, 0.0) for k in k_values]
        rec_values = [recall.get(k, 0.0) for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        ax3.bar(x - width/2, prec_values, width, label='Precision@k', color='#3498db')
        ax3.bar(x + width/2, rec_values, width, label='Recall@k', color='#e74c3c')
        ax3.set_xlabel('k (Number of Results)')
        ax3.set_ylabel('Score')
        ax3.set_title('Retrieval Precision and Recall')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'k={k}' for k in k_values])
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Confidence Distribution (DISABLED)
        ax4 = axes[1, 1]
        # conf_dist = metrics.get('reasoning', {}).get('confidence_distribution', {})
        # 
        # if conf_dist:
        #     bins = sorted(conf_dist.keys())
        #     counts = [conf_dist.get(bin_name, 0) for bin_name in bins]
        #     
        #     ax4.bar(range(len(bins)), counts, color='#f39c12')
        #     ax4.set_xticks(range(len(bins)))
        #     ax4.set_xticklabels(bins, rotation=45, ha='right')
        #     ax4.set_ylabel('Number of Questions')
        #     ax4.set_title('Confidence Score Distribution')
        ax4.axis('off')  # Hide the empty subplot
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.charts_dir / "performance_summary.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: Dict[str, Dict[str, int]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create professional confusion matrix for answer options (A, B, C, D).
        
        Args:
            confusion_matrix: {actual_answer: {predicted_answer: count}}
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        # Get all answer options (A, B, C, D)
        all_answers = set()
        for actual, predictions in confusion_matrix.items():
            if actual:
                all_answers.add(actual)
            for pred in predictions.keys():
                if pred:
                    all_answers.add(pred)
        
        # Sort answers alphabetically
        all_answers = sorted([ans for ans in all_answers if ans])
        
        if not all_answers:
            # Empty confusion matrix, create placeholder
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
            ax.axis('off')
            if save_path is None:
                save_path = self.charts_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        
        # Build matrix
        n = len(all_answers)
        matrix = np.zeros((n, n), dtype=int)
        for i, actual in enumerate(all_answers):
            for j, predicted in enumerate(all_answers):
                matrix[i, j] = confusion_matrix.get(actual, {}).get(predicted, 0)
        
        # Create figure with better sizing
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate percentages for each row (actual answer)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_pct = (matrix / row_sums * 100).astype(float)
        
        # Create annotations with counts and percentages
        annot_matrix = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                count = matrix[i, j]
                pct = matrix_pct[i, j]
                if count > 0:
                    annot_matrix[i, j] = f'{count}\n({pct:.1f}%)'
                else:
                    annot_matrix[i, j] = ''
        
        # Create heatmap with professional styling
        sns.heatmap(
            matrix_pct,
            annot=annot_matrix,
            fmt='',
            cmap='RdYlGn_r',  # Red for wrong, green for correct
            xticklabels=all_answers,
            yticklabels=all_answers,
            cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
            ax=ax,
            linewidths=2,
            linecolor='white',
            vmin=0,
            vmax=100,
            square=True,
            annot_kws={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # Styling
        ax.set_xlabel('Predicted Answer', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Correct Answer', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Answer Prediction Confusion Matrix', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold', rotation=0)
        
        # Add diagonal line to highlight correct predictions
        for i in range(n):
            rect = plt.Rectangle((i, i), 1, 1, fill=False, 
                                edgecolor='blue', linewidth=3, linestyle='--')
            ax.add_patch(rect)
        
        # Add summary statistics below the matrix
        total = matrix.sum()
        correct = np.trace(matrix)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        fig.text(0.5, 0.02, 
                f'Overall Accuracy: {correct}/{total} ({accuracy:.1f}%) | '
                f'Correct predictions are highlighted with dashed blue border',
                ha='center', fontsize=11, style='italic', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        if save_path is None:
            save_path = self.charts_dir / "confusion_matrix.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    def plot_calibration_curve(
        self,
        calibration_data: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create confidence calibration plot.
        
        Args:
            calibration_data: Dictionary with calibration bins
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        bins = sorted(calibration_data.keys())
        predicted_conf = [calibration_data[b]['predicted_confidence'] for b in bins]
        actual_acc = [calibration_data[b]['actual_accuracy'] for b in bins]
        counts = [calibration_data[b]['count'] for b in bins]
        
        # Plot calibration curve
        ax.plot(predicted_conf, actual_acc, 'o-', linewidth=2, markersize=8, 
               label='Calibration Curve', color='#3498db')
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], '--', linewidth=2, color='#e74c3c', 
               label='Perfect Calibration', alpha=0.7)
        
        # Add count annotations
        for i, (pred, acc, count) in enumerate(zip(predicted_conf, actual_acc, counts)):
            ax.annotate(f'n={count}', (pred, acc), 
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        ax.set_xlabel('Predicted Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Calibration', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.charts_dir / "calibration_curve.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_error_analysis(
        self,
        error_categories: Dict,
        pitfalls: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create error analysis charts.
        
        Args:
            error_categories: Dictionary of error categories
            pitfalls: List of identified pitfalls
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        
        # 1. Error type distribution
        ax1 = axes[0]
        error_types = list(error_categories.keys())
        # Handle both ErrorCategory objects and dict format
        error_counts = []
        for et in error_types:
            if isinstance(error_categories[et], dict):
                error_counts.append(error_categories[et].get('count', 0))
            else:
                error_counts.append(error_categories[et].count)
        colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
        
        wedges, texts, autotexts = ax1.pie(
            error_counts,
            labels=error_types,
            autopct='%1.1f%%',
            colors=colors[:len(error_types)],
            startangle=90
        )
        
        ax1.set_title('Error Type Distribution', fontsize=12, fontweight='bold')
        
        # 2. Pitfall severity
        ax2 = axes[1]
        if pitfalls:
            pitfall_names = [p['pitfall'] for p in pitfalls]
            pitfall_counts = [p['count'] for p in pitfalls]
            severity_colors = {
                'high': '#e74c3c',
                'medium': '#f39c12',
                'low': '#f1c40f'
            }
            colors_list = [severity_colors.get(p.get('severity', 'medium'), '#95a5a6') 
                          for p in pitfalls]
            
            bars = ax2.barh(pitfall_names, pitfall_counts, color=colors_list)
            ax2.set_xlabel('Number of Occurrences', fontsize=11)
            ax2.set_title('Identified Pitfalls', fontsize=12, fontweight='bold')
            ax2.invert_yaxis()
            
            # Add count labels
            for bar, count in zip(bars, pitfall_counts):
                ax2.text(count + 0.5, bar.get_y() + bar.get_height()/2, 
                        str(count), va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.charts_dir / "error_analysis.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_retrieval_quality(
        self,
        retrieval_metrics: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create retrieval quality charts.
        
        Args:
            retrieval_metrics: Retrieval metrics dictionary
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Retrieval Quality Metrics', fontsize=16, fontweight='bold')
        
        # 1. MAP and NDCG
        ax1 = axes[0, 0]
        map_score = retrieval_metrics.get('map_score', 0.0)
        ndcg_scores = retrieval_metrics.get('ndcg_at_k', {})
        
        metrics_names = ['MAP'] + [f'NDCG@{k}' for k in sorted(ndcg_scores.keys())]
        metrics_values = [map_score] + [ndcg_scores[k] for k in sorted(ndcg_scores.keys())]
        
        bars = ax1.bar(metrics_names, metrics_values, color='#3498db')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Mean Average Precision and NDCG', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, metrics_values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', fontweight='bold')
        
        # 2. Precision-Recall Curve
        ax2 = axes[0, 1]
        precision = retrieval_metrics.get('precision_at_k', {})
        recall = retrieval_metrics.get('recall_at_k', {})
        
        k_values = sorted([k for k in precision.keys()])
        prec_values = [precision.get(k, 0.0) for k in k_values]
        rec_values = [recall.get(k, 0.0) for k in k_values]
        
        ax2.plot(rec_values, prec_values, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Annotate k values
        for k, rec, prec in zip(k_values, rec_values, prec_values):
            ax2.annotate(f'k={k}', (rec, prec), textcoords="offset points", 
                        xytext=(5,5), fontsize=8)
        
        # 3. Context Relevance Distribution
        ax3 = axes[1, 0]
        relevance_scores = retrieval_metrics.get('context_relevance_scores', [])
        
        if relevance_scores:
            # Count relevance levels (0, 1, 2)
            relevance_counts = defaultdict(int)
            for score in relevance_scores:
                if score >= 1.5:
                    relevance_counts['Highly Relevant (2)'] += 1
                elif score >= 0.5:
                    relevance_counts['Somewhat Relevant (1)'] += 1
                else:
                    relevance_counts['Not Relevant (0)'] += 1
            
            labels = list(relevance_counts.keys())
            counts = list(relevance_counts.values())
            colors_list = ['#2ecc71', '#f39c12', '#e74c3c']
            
            ax3.bar(labels, counts, color=colors_list[:len(labels)])
            ax3.set_ylabel('Count', fontsize=11)
            ax3.set_title('Context Relevance Distribution', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Medical Concept Coverage
        ax4 = axes[1, 1]
        concept_coverage = retrieval_metrics.get('medical_concept_coverage', 0.0)
        guideline_coverage = retrieval_metrics.get('guideline_coverage', 0.0)
        
        categories = ['Medical\nConcepts', 'Guidelines']
        values = [concept_coverage, guideline_coverage]
        colors_list = ['#9b59b6', '#3498db']
        
        bars = ax4.bar(categories, values, color=colors_list)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Coverage', fontsize=11)
        ax4.set_title('Medical Knowledge Coverage', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.charts_dir / "retrieval_quality.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_html_report(
        self,
        evaluation_results: Dict,
        chart_paths: Dict[str, str],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate HTML evaluation report.
        
        Args:
            evaluation_results: Complete evaluation results dictionary
            chart_paths: Dictionary mapping chart names to file paths
            save_path: Optional path to save HTML
            
        Returns:
            Path to saved HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "evaluation_report.html"
        else:
            save_path = Path(save_path)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Medical QA System - Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric-box {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value {{
            font-size: 24px;
            color: #27ae60;
            margin-top: 5px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .error-type {{
            margin: 20px 0;
            padding: 15px;
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .pitfall {{
            margin: 15px 0;
            padding: 15px;
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Medical QA System - Comprehensive Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Executive Summary</h2>
        <div class="metric-box">
            <div class="metric-label">Overall Answer Accuracy</div>
            <div class="metric-value">
                {evaluation_results.get('reasoning', {}).get('exact_match_accuracy', 0.0):.1%}
            </div>
        </div>
        
        <div class="metric-box">
            <div class="metric-label">Retrieval MAP Score</div>
            <div class="metric-value">
                {evaluation_results.get('retrieval', {}).get('map_score', 0.0):.3f}
            </div>
        </div>
        
        <h2>Performance Visualizations</h2>
        {self._generate_chart_html(chart_paths)}
        
        <h2>Detailed Metrics</h2>
        {self._generate_metrics_html(evaluation_results)}
        
        <h2>Error Analysis</h2>
        {self._generate_error_analysis_html(evaluation_results)}
        
        <h2>Identified Pitfalls</h2>
        {self._generate_pitfalls_html(evaluation_results)}
        
        <h2>Recommendations</h2>
        {self._generate_recommendations_html(evaluation_results)}
    </div>
</body>
</html>
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(save_path)
    
    def _generate_chart_html(self, chart_paths: Dict[str, str]) -> str:
        """Generate HTML for chart images."""
        html = ""
        for chart_name, chart_path in chart_paths.items():
            # Convert to relative path
            rel_path = Path(chart_path).relative_to(self.output_dir)
            html += f"""
        <div class="chart-container">
            <h3>{chart_name.replace('_', ' ').title()}</h3>
            <img src="{rel_path}" alt="{chart_name}">
        </div>
"""
        return html
    
    def _generate_metrics_html(self, results: Dict) -> str:
        """Generate HTML for metrics table."""
        html = "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        # Retrieval metrics
        retrieval = results.get('retrieval', {})
        html += f"<tr><td>Mean Average Precision (MAP)</td><td>{retrieval.get('map_score', 0.0):.3f}</td></tr>"
        
        for k, prec in retrieval.get('precision_at_k', {}).items():
            html += f"<tr><td>Precision@{k}</td><td>{prec:.3f}</td></tr>"
        
        for k, rec in retrieval.get('recall_at_k', {}).items():
            html += f"<tr><td>Recall@{k}</td><td>{rec:.3f}</td></tr>"
        
        # Reasoning metrics
        reasoning = results.get('reasoning', {})
        html += f"<tr><td>Exact Match Accuracy</td><td>{reasoning.get('exact_match_accuracy', 0.0):.1%}</td></tr>"
        html += f"<tr><td>Brier Score</td><td>{reasoning.get('brier_score', 0.0):.3f}</td></tr>"
        html += f"<tr><td>Expected Calibration Error</td><td>{reasoning.get('expected_calibration_error', 0.0):.3f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_error_analysis_html(self, results: Dict) -> str:
        """Generate HTML for error analysis."""
        error_categories = results.get('error_analysis', {}).get('error_categories', {})
        
        html = ""
        for error_type, error_data in error_categories.items():
            html += f"""
        <div class="error-type">
            <h3>{error_type.title()} Errors ({error_data.get('count', 0)} occurrences)</h3>
            <p><strong>Description:</strong> {error_data.get('description', '')}</p>
            <p><strong>Root Causes:</strong></p>
            <ul>
"""
            for cause in error_data.get('root_causes', []):
                html += f"<li>{cause}</li>"
            html += """
            </ul>
        </div>
"""
        return html
    
    def _generate_pitfalls_html(self, results: Dict) -> str:
        """Generate HTML for pitfalls."""
        pitfalls = results.get('error_analysis', {}).get('pitfalls', [])
        
        html = ""
        for pitfall in pitfalls:
            html += f"""
        <div class="pitfall">
            <h3>{pitfall.get('pitfall', 'Unknown')} ({pitfall.get('count', 0)} occurrences)</h3>
            <p><strong>Description:</strong> {pitfall.get('description', '')}</p>
            <p><strong>Severity:</strong> {pitfall.get('severity', 'unknown').upper()}</p>
            <p><strong>Proposed Solution:</strong> {pitfall.get('solution', '')}</p>
        </div>
"""
        return html
    
    def _generate_recommendations_html(self, results: Dict) -> str:
        """Generate HTML for recommendations."""
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            recommendations = [
                "Improve retrieval precision for medical terminology",
                "Enhance reasoning chain completeness",
                "Add confidence calibration",
                "Expand medical knowledge base coverage"
            ]
        
        html = "<ol>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ol>"
        
        return html

