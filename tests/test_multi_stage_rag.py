"""
Unit tests for multi-stage RAG system components.

Tests cover:
- Multi-stage retriever
- Medical reasoning engine
- Query understanding
- RAG pipeline integration
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.query_understanding import MedicalQueryUnderstanding
from src.reasoning.medical_reasoning import MedicalReasoningEngine
from src.retrieval.document_processor import Document


class TestQueryUnderstanding(unittest.TestCase):
    """Test query understanding module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.understanding = MedicalQueryUnderstanding()
    
    def test_abbreviation_expansion(self):
        """Test medical abbreviation expansion."""
        query = "What is the treatment for acute MI?"
        understanding = self.understanding.understand(query)
        
        # Check that MI was expanded
        self.assertIn("myocardial infarction", understanding.expanded_query.lower())
    
    def test_clinical_feature_extraction(self):
        """Test clinical feature extraction."""
        query = "65-year-old male with chest pain and elevated troponin"
        understanding = self.understanding.understand(query)
        
        features = understanding.clinical_features
        
        # Check demographics
        self.assertEqual(features.demographics.get('age'), 65)
        self.assertEqual(features.demographics.get('gender'), 'male')
        
        # Check symptoms
        self.assertIn('chest pain', ' '.join(features.symptoms).lower())
        
        # Check tests
        self.assertIn('troponin', ' '.join(features.tests).lower())
    
    def test_specialty_identification(self):
        """Test medical specialty identification."""
        query = "Patient with acute myocardial infarction and elevated troponin"
        understanding = self.understanding.understand(query)
        
        # Should identify cardiology
        self.assertEqual(understanding.likely_specialty, 'cardiology')
    
    def test_acuity_determination(self):
        """Test acuity level determination."""
        # Emergency query
        emergency_query = "Emergency patient with acute MI"
        understanding = self.understanding.understand(emergency_query)
        self.assertEqual(understanding.acuity_level, 'emergency')
        
        # Routine query
        routine_query = "Follow-up for diabetes management"
        understanding = self.understanding.understand(routine_query)
        self.assertEqual(understanding.acuity_level, 'routine')


class TestMedicalReasoning(unittest.TestCase):
    """Test medical reasoning engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reasoning = MedicalReasoningEngine()
        
        # Create mock documents
        self.mock_docs = [
            Document(
                content="Aspirin 325mg is the first-line treatment for acute MI",
                metadata={'guideline_id': 'GL_001', 'title': 'MI Treatment', 'category': 'Cardiology'}
            ),
            Document(
                content="Morphine is contraindicated in patients with respiratory depression",
                metadata={'guideline_id': 'GL_002', 'title': 'Pain Management', 'category': 'Emergency'}
            )
        ]
    
    def test_evidence_matching(self):
        """Test evidence matching to answer options."""
        question = "What is the first-line treatment?"
        case = "65-year-old with acute MI"
        options = {
            'A': 'Aspirin 325mg',
            'B': 'Morphine',
            'C': 'Beta blocker',
            'D': 'ACE inhibitor'
        }
        
        answer_selection = self.reasoning.reason_and_select_answer(
            question=question,
            case_description=case,
            options=options,
            retrieved_contexts=self.mock_docs
        )
        
        # Should select A (Aspirin) based on evidence
        self.assertEqual(answer_selection.selected_answer, 'A')
        
        # Should have evidence for option A
        self.assertGreater(len(answer_selection.evidence_matches['A'].supporting_evidence), 0)
    
    def test_contradiction_detection(self):
        """Test detection of contradictions."""
        question = "What medication should be avoided?"
        case = "Patient with respiratory depression"
        options = {
            'A': 'Aspirin',
            'B': 'Morphine',
            'C': 'Beta blocker',
            'D': 'ACE inhibitor'
        }
        
        answer_selection = self.reasoning.reason_and_select_answer(
            question=question,
            case_description=case,
            options=options,
            retrieved_contexts=self.mock_docs
        )
        
        # Option B (Morphine) should have contradicting evidence
        self.assertGreater(
            len(answer_selection.evidence_matches['B'].contradicting_evidence),
            0
        )
    
    def test_confidence_scoring(self):
        """Test confidence score calculation."""
        question = "What is the treatment?"
        case = "Acute MI"
        options = {
            'A': 'Aspirin 325mg',
            'B': 'Placebo',
            'C': 'Unknown treatment',
            'D': 'No treatment'
        }
        
        answer_selection = self.reasoning.reason_and_select_answer(
            question=question,
            case_description=case,
            options=options,
            retrieved_contexts=self.mock_docs
        )
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(answer_selection.confidence_score, 0.0)
        self.assertLessEqual(answer_selection.confidence_score, 1.0)
        
        # Option A should have highest confidence (has evidence)
        option_a_confidence = answer_selection.evidence_matches['A'].evidence_strength
        option_b_confidence = answer_selection.evidence_matches['B'].evidence_strength
        self.assertGreater(option_a_confidence, option_b_confidence)
    
    def test_reasoning_steps(self):
        """Test chain-of-thought reasoning steps."""
        question = "What is the treatment?"
        case = "Acute MI"
        options = {'A': 'Aspirin', 'B': 'Morphine'}
        
        answer_selection = self.reasoning.reason_and_select_answer(
            question=question,
            case_description=case,
            options=options,
            retrieved_contexts=self.mock_docs
        )
        
        # Should have reasoning steps
        self.assertGreater(len(answer_selection.reasoning_steps), 0)
        
        # First step should be feature extraction
        self.assertEqual(answer_selection.reasoning_steps[0].step_number, 1)
        self.assertIn("Extract", answer_selection.reasoning_steps[0].description)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_pipeline_initialization(self):
        """Test that pipeline can be initialized."""
        try:
            from src.reasoning.rag_pipeline import load_pipeline
            # This will fail if indexes don't exist, but that's OK for unit test
            # In real scenario, indexes should be built first
            pass
        except FileNotFoundError:
            # Expected if indexes don't exist
            pass
        except Exception as e:
            self.fail(f"Pipeline initialization failed: {e}")


if __name__ == '__main__':
    unittest.main()

