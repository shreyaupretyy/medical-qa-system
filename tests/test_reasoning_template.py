import pytest
from pathlib import Path
from src.reasoning.medical_reasoning import MedicalReasoningEngine, ReasoningStep, EvidenceMatch
from src.retrieval.document_processor import Document


def fake_doc(text: str, gid: str = "GL_FAKE"):
    return Document(content=text, metadata={"guideline_id": gid, "title": "Fake Guideline", "category": "Cardio"})


def test_structured_reasoning_template_enforced():
    engine = MedicalReasoningEngine(llm_model=None)
    options = {"A": "Give aspirin", "B": "Give morphine", "C": "Order ECG", "D": "Do nothing"}
    docs = [
        fake_doc("ACS management: give aspirin 300 mg stat. Red flags include hypotension."),
        fake_doc("Diagnosis: chest pain with st elevation. Investigations: ECG, troponin.")
    ]
    # Build minimal evidence matches
    matches = {
        "A": EvidenceMatch("A", options["A"], [(docs[0], "give aspirin 300 mg stat", 0.9)], [], 0.9, "direct"),
        "B": EvidenceMatch("B", options["B"], [], [], 0.0, "none"),
        "C": EvidenceMatch("C", options["C"], [(docs[1], "Order ECG", 0.6)], [], 0.6, "direct"),
        "D": EvidenceMatch("D", options["D"], [], [], 0.0, "none"),
    }
    steps = engine._build_structured_reasoning(
        case_description="Chest pain radiating to left arm",
        question="What is the next best step?",
        clinical_features=engine.query_understanding.understand("Chest pain radiating to left arm").clinical_features,
        evidence_matches=matches,
        retrieved_contexts=docs,
    )
    titles = [s.description for s in steps]
    assert "Summary of key symptoms & findings" in titles
    assert "Differential diagnosis (>=3)" in titles
    assert "Probable diagnosis + rationale (guideline-first)" in titles
    assert "Recommended next steps / investigations" in titles


def test_rationale_cites_evidence():
    engine = MedicalReasoningEngine(llm_model=None)
    option = "Give aspirin"
    match = EvidenceMatch("A", option, [(fake_doc("Aspirin 300 mg"), "Aspirin 300 mg", 0.9)], [], 0.9, "direct")
    rationale = engine._generate_rationale("A", match, engine.query_understanding.understand("Chest pain").clinical_features, None)
    assert "Aspirin 300 mg" in rationale

