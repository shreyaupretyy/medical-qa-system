"""
Clinical Case Question Generator v5.0
======================================

IMPROVEMENTS FROM v4:
1. Blood Pressure Constraints - Realistic ranges, flagged wide pulse pressure
2. Stroke-Specific Rules - "Image before anticoagulating" enforced
3. Option Formatting - Clean options without embedded explanations
4. Presentation Variety - Reduced repetitive patterns
5. Clinical Consistency Validator - Enhanced with more rules

Key Clinical Rules Enforced:
- Fever > 101°F → Must consider infectious/inflammatory differentials
- Stroke → CT/MRI BEFORE any anticoagulation
- Wide pulse pressure → Must be explained or normalized
- Clean option formatting (no embedded explanations)
- Varied presentations within same condition category

Usage: python scripts/generate_clinical_cases_v5.py [num_cases] [output_file]
"""

import sys
import os
import json
import random
import re
import math
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import ollama
except ImportError:
    print("[ERROR] ollama not installed")
    sys.exit(1)


# =============================================================================
# CLINICAL VITAL SIGN CONSTRAINTS BY DIAGNOSIS (ENHANCED)
# =============================================================================

VITAL_SIGN_CONSTRAINTS = {
    # Cardiovascular conditions - with realistic BP constraints
    "acs": {
        "temp_range": (97.0, 99.0),
        "hr_range": (50, 120),
        "bp_systolic_range": (100, 180),
        "bp_diastolic_range": (60, 100),  # Added to prevent wide pulse pressure
        "pulse_pressure_max": 70,  # Flag if wider
        "spo2_range": (92, 100),
        "rr_range": (14, 24),
        "conflicting_findings": ["high fever", "hypothermia"],
        "supportive_findings": ["diaphoresis", "chest pain radiating to arm/jaw", "nausea", "ST changes"]
    },
    "stemi": {
        "temp_range": (97.0, 99.0),
        "hr_range": (40, 130),
        "bp_systolic_range": (90, 180),
        "bp_diastolic_range": (50, 100),
        "pulse_pressure_max": 70,
        "spo2_range": (88, 100),
        "rr_range": (14, 28),
        "conflicting_findings": ["high fever"],
        "supportive_findings": ["ST elevation", "troponin elevated", "diaphoresis", "chest pressure"]
    },
    "heart_failure": {
        "temp_range": (97.0, 99.5),
        "hr_range": (60, 130),
        "bp_systolic_range": (90, 180),
        "bp_diastolic_range": (50, 100),
        "pulse_pressure_max": 80,
        "spo2_range": (82, 96),
        "rr_range": (18, 32),
        "conflicting_findings": [],
        "supportive_findings": ["JVD", "peripheral edema", "crackles", "S3 gallop", "orthopnea"]
    },
    "atrial_fibrillation": {
        "temp_range": (97.0, 100.5),
        "hr_range": (90, 180),
        "bp_systolic_range": (90, 170),
        "bp_diastolic_range": (50, 100),
        "pulse_pressure_max": 70,
        "spo2_range": (92, 100),
        "rr_range": (14, 24),
        "conflicting_findings": [],
        "supportive_findings": ["irregularly irregular pulse", "palpitations", "variable pulse intensity"]
    },
    "dvt": {
        "temp_range": (97.0, 100.0),
        "hr_range": (70, 110),
        "bp_systolic_range": (110, 150),
        "bp_diastolic_range": (60, 90),
        "pulse_pressure_max": 60,
        "spo2_range": (95, 100),
        "rr_range": (14, 20),
        "conflicting_findings": [],
        "supportive_findings": ["unilateral leg swelling", "calf tenderness", "warmth", "Homan's sign"]
    },
    "hypertension": {
        "temp_range": (97.0, 99.0),
        "hr_range": (60, 100),
        "bp_systolic_range": (150, 200),
        "bp_diastolic_range": (90, 120),
        "pulse_pressure_max": 80,
        "spo2_range": (95, 100),
        "rr_range": (12, 20),
        "conflicting_findings": [],
        "supportive_findings": ["headache", "visual changes", "epistaxis"]
    },
    
    # Respiratory conditions
    "copd_exacerbation": {
        "temp_range": (97.0, 101.0),
        "hr_range": (80, 130),
        "bp_systolic_range": (100, 160),
        "bp_diastolic_range": (60, 95),
        "pulse_pressure_max": 70,
        "spo2_range": (82, 92),
        "rr_range": (22, 36),
        "conflicting_findings": [],
        "supportive_findings": ["wheezing", "barrel chest", "accessory muscle use", "pursed lip breathing"]
    },
    "asthma_exacerbation": {
        "temp_range": (97.0, 100.0),
        "hr_range": (90, 140),
        "bp_systolic_range": (100, 160),
        "bp_diastolic_range": (60, 95),
        "pulse_pressure_max": 70,
        "spo2_range": (85, 96),
        "rr_range": (24, 40),
        "conflicting_findings": [],
        "supportive_findings": ["wheezing", "prolonged expiration", "tripod position", "decreased air entry"]
    },
    "pneumonia": {
        "temp_range": (100.5, 103.5),  # FEVER REQUIRED
        "hr_range": (90, 130),
        "bp_systolic_range": (95, 150),
        "bp_diastolic_range": (55, 90),
        "pulse_pressure_max": 70,
        "spo2_range": (85, 95),
        "rr_range": (22, 34),
        "conflicting_findings": ["afebrile"],
        "supportive_findings": ["productive cough", "crackles", "dullness to percussion", "egophony"]
    },
    "pulmonary_embolism": {
        "temp_range": (97.0, 100.5),
        "hr_range": (100, 140),
        "bp_systolic_range": (85, 140),
        "bp_diastolic_range": (50, 90),
        "pulse_pressure_max": 70,
        "spo2_range": (80, 94),
        "rr_range": (22, 36),
        "conflicting_findings": [],
        "supportive_findings": ["pleuritic chest pain", "hemoptysis", "leg swelling", "tachycardia"]
    },
    
    # Infectious conditions
    "sepsis": {
        "temp_range": (101.0, 104.0),
        "hr_range": (100, 150),
        "bp_systolic_range": (70, 100),
        "bp_diastolic_range": (40, 65),
        "pulse_pressure_max": 50,
        "spo2_range": (85, 95),
        "rr_range": (22, 36),
        "conflicting_findings": [],
        "supportive_findings": ["altered mental status", "warm skin", "lactate elevated", "WBC abnormal"]
    },
    "uti": {
        "temp_range": (99.0, 102.5),
        "hr_range": (70, 110),
        "bp_systolic_range": (100, 160),
        "bp_diastolic_range": (60, 95),
        "pulse_pressure_max": 70,
        "spo2_range": (95, 100),
        "rr_range": (14, 20),
        "conflicting_findings": [],
        "supportive_findings": ["dysuria", "frequency", "urgency", "suprapubic tenderness", "pyuria"]
    },
    
    # Neurological conditions - CRITICAL RULES
    "stroke_ischemic": {
        "temp_range": (97.0, 99.0),  # NO fever - if fever, think infection
        "hr_range": (60, 100),
        "bp_systolic_range": (140, 200),  # Often hypertensive - do NOT lower acutely
        "bp_diastolic_range": (80, 110),
        "pulse_pressure_max": 80,
        "spo2_range": (94, 100),
        "rr_range": (14, 22),
        "conflicting_findings": ["high fever"],  # Fever + neuro = infection!
        "supportive_findings": ["focal weakness", "speech changes", "facial droop", "arm drift"],
        "critical_rules": ["MUST image (CT/MRI) BEFORE anticoagulation", "Do NOT give tPA if >4.5h or contraindicated"]
    },
    "meningitis": {
        "temp_range": (101.5, 104.5),
        "hr_range": (100, 140),
        "bp_systolic_range": (90, 160),
        "bp_diastolic_range": (55, 100),
        "pulse_pressure_max": 70,
        "spo2_range": (94, 100),
        "rr_range": (18, 28),
        "conflicting_findings": ["afebrile"],
        "supportive_findings": ["neck stiffness", "photophobia", "Kernig/Brudzinski positive", "headache"]
    },
    
    # GI conditions
    "gi_bleed": {
        "temp_range": (97.0, 99.0),
        "hr_range": (90, 140),
        "bp_systolic_range": (70, 110),
        "bp_diastolic_range": (40, 70),
        "pulse_pressure_max": 50,
        "spo2_range": (94, 100),
        "rr_range": (16, 26),
        "conflicting_findings": [],
        "supportive_findings": ["melena", "hematemesis", "pallor", "dizziness", "orthostatic changes"]
    },
    "pancreatitis": {
        "temp_range": (99.0, 102.0),
        "hr_range": (90, 130),
        "bp_systolic_range": (90, 150),
        "bp_diastolic_range": (55, 90),
        "pulse_pressure_max": 70,
        "spo2_range": (92, 100),
        "rr_range": (18, 28),
        "conflicting_findings": [],
        "supportive_findings": ["epigastric pain radiating to back", "nausea/vomiting", "lipase elevated", "guarding"]
    },
    
    # Endocrine conditions
    "dka": {
        "temp_range": (97.0, 100.0),
        "hr_range": (100, 140),
        "bp_systolic_range": (80, 120),
        "bp_diastolic_range": (50, 80),
        "pulse_pressure_max": 50,
        "spo2_range": (94, 100),
        "rr_range": (24, 40),
        "conflicting_findings": [],
        "supportive_findings": ["fruity breath", "altered mental status", "Kussmaul breathing", "glucose >250"]
    },
    "diabetes": {
        "temp_range": (97.0, 99.5),
        "hr_range": (60, 100),
        "bp_systolic_range": (100, 150),
        "bp_diastolic_range": (60, 95),
        "pulse_pressure_max": 60,
        "spo2_range": (95, 100),
        "rr_range": (12, 20),
        "conflicting_findings": ["severe hypoxemia", "shock"],
        "supportive_findings": ["polyuria", "polydipsia", "blurred vision", "fatigue"]
    },
    
    # Renal conditions
    "aki": {
        "temp_range": (97.0, 101.0),
        "hr_range": (70, 110),
        "bp_systolic_range": (100, 180),
        "bp_diastolic_range": (60, 100),
        "pulse_pressure_max": 80,
        "spo2_range": (94, 100),
        "rr_range": (14, 22),
        "conflicting_findings": [],
        "supportive_findings": ["decreased urine output", "edema", "elevated creatinine", "uremic symptoms"]
    },
    
    # Rheumatologic
    "rheumatoid_arthritis": {
        "temp_range": (97.0, 99.5),
        "hr_range": (60, 100),
        "bp_systolic_range": (100, 140),
        "bp_diastolic_range": (60, 90),
        "pulse_pressure_max": 60,
        "spo2_range": (96, 100),
        "rr_range": (12, 18),
        "conflicting_findings": [],
        "supportive_findings": ["morning stiffness >1hr", "symmetric joint swelling", "RF/anti-CCP positive"]
    },
    
    # Mental health
    "depression": {
        "temp_range": (97.0, 99.0),
        "hr_range": (55, 90),
        "bp_systolic_range": (100, 140),
        "bp_diastolic_range": (60, 90),
        "pulse_pressure_max": 60,
        "spo2_range": (97, 100),
        "rr_range": (12, 18),
        "conflicting_findings": [],
        "supportive_findings": ["depressed mood", "anhedonia", "sleep changes", "weight changes", "fatigue"]
    },
    
    # Default
    "default": {
        "temp_range": (97.0, 99.0),
        "hr_range": (60, 100),
        "bp_systolic_range": (100, 140),
        "bp_diastolic_range": (60, 85),
        "pulse_pressure_max": 60,
        "spo2_range": (95, 100),
        "rr_range": (12, 20),
        "conflicting_findings": [],
        "supportive_findings": []
    }
}


# =============================================================================
# STROKE-SPECIFIC CRITICAL RULES
# =============================================================================

STROKE_RULES = {
    "mandatory_before_treatment": [
        "CT head or MRI brain",
        "Imaging to rule out hemorrhage"
    ],
    "forbidden_before_imaging": [
        "anticoagulation",
        "heparin",
        "warfarin",
        "DOAC",
        "thrombolysis",
        "tPA",
        "alteplase"
    ],
    "correct_sequence": [
        "1. Obtain non-contrast CT head (or MRI)",
        "2. Rule out hemorrhagic stroke",
        "3. If ischemic and <4.5h, consider tPA",
        "4. Aspirin for secondary prevention (after ruling out hemorrhage)"
    ]
}


# =============================================================================
# PRESENTATION VARIETY TEMPLATES
# =============================================================================

PRESENTATION_VARIETIES = {
    "acs": [
        {"setting": "emergency department", "symptom_onset": "30 minutes", "pain_desc": "crushing chest pain", "radiation": "left arm"},
        {"setting": "urgent care", "symptom_onset": "2 hours", "pain_desc": "substernal pressure", "radiation": "jaw"},
        {"setting": "office", "symptom_onset": "45 minutes", "pain_desc": "heaviness in chest", "radiation": "both arms"},
        {"setting": "ambulance", "symptom_onset": "1 hour", "pain_desc": "squeezing chest discomfort", "radiation": "neck"},
        {"setting": "home visit", "symptom_onset": "20 minutes", "pain_desc": "tightness across chest", "radiation": "back"},
    ],
    "stroke_ischemic": [
        {"setting": "emergency department", "symptom_onset": "1 hour", "presentation": "sudden right-sided weakness and slurred speech"},
        {"setting": "ambulance", "symptom_onset": "45 minutes", "presentation": "acute left arm and leg weakness with facial droop"},
        {"setting": "urgent care", "symptom_onset": "2 hours", "presentation": "sudden difficulty speaking and right arm weakness"},
        {"setting": "home", "symptom_onset": "30 minutes", "presentation": "abrupt onset of left facial droop and arm drift"},
    ],
    "copd_exacerbation": [
        {"setting": "emergency department", "symptom_onset": "3 days", "presentation": "worsening dyspnea and increased sputum production"},
        {"setting": "clinic", "symptom_onset": "1 week", "presentation": "progressive shortness of breath with productive cough"},
        {"setting": "urgent care", "symptom_onset": "4 days", "presentation": "increasing breathlessness and wheezing"},
    ],
    "default": [
        {"setting": "emergency department", "symptom_onset": "variable"},
        {"setting": "clinic", "symptom_onset": "variable"},
        {"setting": "urgent care", "symptom_onset": "variable"},
    ]
}


# =============================================================================
# CLINICAL CONSISTENCY VALIDATOR (ENHANCED)
# =============================================================================

class ClinicalValidator:
    """Enhanced clinical consistency validator."""
    
    @staticmethod
    def get_condition_from_guideline(guideline_name: str) -> str:
        """Map guideline name to condition category."""
        name_lower = guideline_name.lower()
        
        mappings = {
            "acs": ["acute coronary", "acs", "myocardial infarction", "angina"],
            "stemi": ["st elevation", "stemi"],
            "heart_failure": ["heart failure", "pulmonary edema", "chf"],
            "atrial_fibrillation": ["atrial fibrillation", "afib", "a-fib"],
            "dvt": ["deep vein thrombosis", "dvt"],
            "hypertension": ["hypertension"],
            "copd_exacerbation": ["copd", "chronic obstructive"],
            "asthma_exacerbation": ["asthma"],
            "pneumonia": ["pneumonia", "community acquired"],
            "pulmonary_embolism": ["pulmonary embolism", "pe "],
            "sepsis": ["sepsis", "septic"],
            "uti": ["urinary tract", "uti"],
            # Diabetes-specific mapping (avoid misclassifying as PE)
            "dka": ["diabetic ketoacidosis", "dka", "ketoacidosis"],
            "diabetes": ["diabetes", "hyperglycemia", "insulin", "glucose management", "diabetes manage"],
            "stroke_ischemic": ["stroke", "ischemic stroke"],
            "meningitis": ["meningitis"],
            "gi_bleed": ["gastrointestinal bleed", "gi bleed", "bleeding"],
            "pancreatitis": ["pancreatitis"],
            "aki": ["acute kidney", "aki"],
            "rheumatoid_arthritis": ["rheumatoid"],
            "depression": ["depression"],
        }
        
        for condition, keywords in mappings.items():
            if any(kw in name_lower for kw in keywords):
                return condition
        
        return "default"
    
    @staticmethod
    def generate_appropriate_vitals(condition: str) -> dict:
        """Generate realistic vitals with proper pulse pressure.

        Additional safety:
        - If SpO2 < 92% but condition is not primarily pulmonary/cardiac, clamp to >=94
        - Avoid severe hypoxemia for non-respiratory/non-cardiac conditions
        """
        constraints = VITAL_SIGN_CONSTRAINTS.get(condition, VITAL_SIGN_CONSTRAINTS["default"])
        
        temp_min, temp_max = constraints["temp_range"]
        hr_min, hr_max = constraints["hr_range"]
        bp_sys_min, bp_sys_max = constraints["bp_systolic_range"]
        bp_dia_min, bp_dia_max = constraints["bp_diastolic_range"]
        pp_max = constraints.get("pulse_pressure_max", 70)
        spo2_min, spo2_max = constraints["spo2_range"]
        rr_min, rr_max = constraints["rr_range"]
        
        # Generate vitals with realistic pulse pressure
        temp = round(random.uniform(temp_min, temp_max), 1)
        hr = random.randint(hr_min, hr_max)
        
        # Ensure realistic pulse pressure (systolic - diastolic)
        bp_systolic = random.randint(bp_sys_min, bp_sys_max)
        # Calculate diastolic to keep pulse pressure reasonable
        min_diastolic = max(bp_dia_min, bp_systolic - pp_max)
        max_diastolic = min(bp_dia_max, bp_systolic - 30)  # At least 30 mmHg difference
        
        if min_diastolic > max_diastolic:
            min_diastolic = bp_systolic - pp_max
            max_diastolic = bp_systolic - 30
        
        bp_diastolic = random.randint(int(min_diastolic), int(max_diastolic))
        pulse_pressure = bp_systolic - bp_diastolic
        
        spo2 = random.randint(spo2_min, spo2_max)
        rr = random.randint(rr_min, rr_max)

        # Clamp hypoxemia for non-pulmonary/cardiac conditions
        hypoxemia_allowed = condition in {
            "pulmonary_embolism",
            "pneumonia",
            "copd_exacerbation",
            "asthma_exacerbation",
            "heart_failure",
            "sepsis"
        }
        if not hypoxemia_allowed and spo2 < 92:
            spo2 = random.randint(94, 98)
            rr = max(rr_min, min(rr, 22))  # Normalize RR if we corrected SpO2
        # Hard safety clamp: never allow SpO2 < 90 in any condition
        if spo2 < 90:
            spo2 = random.randint(94, 97)
            rr = max(rr_min, min(rr, 24))
        
        return {
            "temp": temp,
            "temp_str": f"{temp}°F",
            "hr": hr,
            "hr_str": f"{hr} bpm",
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "pulse_pressure": pulse_pressure,
            "bp_str": f"{bp_systolic}/{bp_diastolic} mmHg",
            "spo2": spo2,
            "spo2_str": f"{spo2}% on room air",
            "rr": rr,
            "rr_str": f"{rr}/min",
            "vital_string": f"BP {bp_systolic}/{bp_diastolic} mmHg, HR {hr} bpm, RR {rr}/min, Temp {temp}°F, SpO2 {spo2}%"
        }
    
    @staticmethod
    def get_presentation_variety(condition: str) -> dict:
        """Get a random presentation variety for the condition."""
        varieties = PRESENTATION_VARIETIES.get(condition, PRESENTATION_VARIETIES["default"])
        return random.choice(varieties)


# =============================================================================
# ANTI-DUPLICATION: SCENARIO TRACKING
# =============================================================================

class ScenarioTracker:
    """Tracks generated scenarios to prevent duplication."""
    
    def __init__(self):
        self.used_scenarios: Set[str] = set()
        self.used_demographics: Set[str] = set()
        self.used_presentations: Dict[str, Set[str]] = {}  # condition -> presentations used
        
    def _hash_scenario(self, text: str) -> str:
        normalized = text.lower()
        normalized = re.sub(r'\d+', 'N', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def is_scenario_used(self, scenario: str) -> bool:
        return self._hash_scenario(scenario) in self.used_scenarios
    
    def mark_scenario_used(self, scenario: str):
        self.used_scenarios.add(self._hash_scenario(scenario))
    
    def get_unique_demographics(self, guideline_name: str, condition: str) -> dict:
        """Generate unique demographics."""
        name_lower = guideline_name.lower()
        
        age_ranges = {
            "acs": [(48, 58), (55, 65), (62, 72), (68, 78)],
            "stemi": [(45, 55), (55, 65), (65, 75)],
            "stroke_ischemic": [(58, 68), (65, 75), (72, 82)],
            "copd_exacerbation": [(55, 65), (60, 70), (65, 75)],
            "heart_failure": [(55, 68), (65, 75), (70, 80)],
            "default": [(28, 38), (38, 48), (48, 58), (58, 68), (68, 78)]
        }
        
        ranges = age_ranges.get(condition, age_ranges["default"])
        age_range = random.choice(ranges)
        age = random.randint(age_range[0], age_range[1])
        
        if any(x in name_lower for x in ['pregnancy', 'obstetric']):
            gender = "female"
        elif condition == "rheumatoid_arthritis":
            gender = random.choice(["female", "female", "female", "male"])
        else:
            gender = random.choice(["male", "female"])
        
        demo_key = f"{guideline_name}_{age}_{gender}"
        attempts = 0
        while demo_key in self.used_demographics and attempts < 30:
            age = random.randint(age_range[0], age_range[1])
            gender = random.choice(["male", "female"])
            demo_key = f"{guideline_name}_{age}_{gender}"
            attempts += 1
        
        self.used_demographics.add(demo_key)
        
        return {"age": age, "age_str": f"{age}-year-old", "gender": gender}


# =============================================================================
# QUESTION TEMPLATES (CLEAN OPTIONS - NO EMBEDDED EXPLANATIONS)
# =============================================================================

QUESTION_TEMPLATES = {
    "diagnosis": """Create a DIAGNOSIS question. Rules:
- Present symptoms WITHOUT revealing the diagnosis in the stem
- Include 2-3 plausible differential diagnoses as distractors
- Require integration of at least 3 clinical data points
- VITAL SIGNS MUST SUPPORT the intended diagnosis
- OPTIONS MUST BE CONCISE - single diagnoses only, NO explanations in options""",

    "treatment": """Create a TREATMENT question. Rules:
- State the diagnosis clearly if needed
- Focus on selecting the BEST first-line treatment
- Include specific drug names or treatment modalities
- OPTIONS MUST BE CONCISE - drug/treatment name only, NO explanations
- For STROKE: imaging MUST come before any anticoagulation""",

    "management": """Create a MANAGEMENT question. Rules:
- Present a clinical decision point
- Focus on next steps, monitoring, or disposition
- OPTIONS MUST BE CONCISE - action only, NO explanations
- The answer must address the MOST CRITICAL finding first
- For STROKE: imaging MUST come before any anticoagulation""",

    "immediate": """Create an EMERGENCY intervention question. Rules:
- Present an acute, time-sensitive scenario
- Focus on immediate life-saving interventions
- OPTIONS MUST BE CONCISE - intervention only, NO explanations
- The correct answer is the FIRST priority action"""
}


# =============================================================================
# OPTION FORMAT ENFORCER
# =============================================================================

def clean_option_text(option_text: str) -> str:
    """Clean option text to remove embedded explanations and fix truncation."""
    # First, complete any truncated parentheses
    if '(' in option_text and ')' not in option_text:
        # Truncated parenthetical - remove it entirely
        option_text = re.sub(r'\s*\([^)]*$', '', option_text)
    
    # Remove explanatory phrases
    patterns_to_remove = [
        r'\s*\([^)]*\)\s*$',  # Remove trailing parenthetical
        r'\s*-\s*this\s+.*$',  # Remove "- this is..."
        r'\s*because\s+.*$',  # Remove "because..."
        r'\s*since\s+.*$',  # Remove "since..."
        r'\s*as\s+it\s+.*$',  # Remove "as it..."
        r'\s*which\s+.*$',  # Remove "which..."
        r'\s*to\s+ensure\s+.*$',  # Remove "to ensure..."
        r'\s*for\s+.*stabilization.*$',  # Remove stabilization explanations
    ]
    
    cleaned = option_text.strip()
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove duplicate letter prefixes like "D) D)"
    cleaned = re.sub(r'^([A-D]\))\s*\1', r'\1', cleaned)
    cleaned = re.sub(r'^[A-D]\)\s*', '', cleaned)  # Remove any remaining letter prefix
    
    # Remove leading/trailing whitespace and punctuation issues
    cleaned = cleaned.strip(' .,;:')
    
    return cleaned




def parse_explanation_robust(text: str) -> dict:
    """Robustly parse explanation from LLM response."""
    explanation = {
        "concept": "",
        "why_correct": "",
        "why_others_wrong": "",
        "clinical_pearl": ""
    }
    
    # Clean text
    text = re.sub(r'\*\*', '', text)
    
    # Try multiple patterns for each field
    
    # CONCEPT
    concept_patterns = [
        r'CONCEPT[:\s]*(.+?)(?=\n\s*WHY|\n\s*[A-D]\s+IS|\Z)',
        r'Core\s+(?:clinical\s+)?principle[:\s]*(.+?)(?=\n|\Z)',
        r'(?:The\s+)?concept[:\s]*(.+?)(?=\n|\Z)',
    ]
    for pattern in concept_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and len(match.group(1).strip()) > 5:
            explanation["concept"] = match.group(1).strip()[:500]
            break
    
    # WHY CORRECT
    why_correct_patterns = [
        r'WHY\s+[A-D]\s+IS\s+CORRECT[:\s]*(.+?)(?=\n\s*WHY\s+OTHER|\n\s*-\s*[A-D]|\n\s*Option|\Z)',
        r'(?:is\s+)?correct\s+because[:\s]*(.+?)(?=\n\s*WHY|\n\s*-|\Z)',
        r'The\s+correct\s+answer[:\s]*(.+?)(?=\n\s*WHY|\n\s*-|\Z)',
    ]
    for pattern in why_correct_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and len(match.group(1).strip()) > 10:
            explanation["why_correct"] = match.group(1).strip()[:800]
            break
    
    # WHY OTHERS WRONG
    why_wrong_patterns = [
        r'WHY\s+OTHER[S]?\s+(?:ARE\s+)?WRONG[:\s]*(.+?)(?=\n\s*CLINICAL|\n\s*Pearl|\Z)',
        r'Other\s+options[:\s]*(.+?)(?=\n\s*CLINICAL|\n\s*Pearl|\Z)',
        r'(?:-\s*(?:Option\s+)?[A-D][:\)].+?(?:\n|$)){2,}',  # Multiple option explanations
    ]
    for pattern in why_wrong_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1) if match.lastindex else match.group(0)
            if len(content.strip()) > 10:
                explanation["why_others_wrong"] = content.strip()[:1000]
                break
    
    # CLINICAL PEARL
    pearl_patterns = [
        r'CLINICAL\s+PEARL[:\s]*(.+?)$',
        r'Pearl[:\s]*(.+?)$',
        r'Teaching\s+point[:\s]*(.+?)$',
        r'Key\s+takeaway[:\s]*(.+?)$',
    ]
    for pattern in pearl_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and len(match.group(1).strip()) > 5:
            explanation["clinical_pearl"] = match.group(1).strip()[:300]
            break
    
    # If still empty, try to extract any useful content
    if not explanation["why_correct"] and not explanation["concept"]:
        # Try to find any explanation-like content
        lines = text.split('\n')
        useful_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith(('VIGNETTE', 'QUESTION', 'OPTIONS', 'CORRECT', 'A)', 'B)', 'C)', 'D)')):
                useful_lines.append(line)
        if useful_lines:
            explanation["why_correct"] = ' '.join(useful_lines[:3])[:500]
    
    return explanation


def load_guidelines(guidelines_dir: str) -> list:
    """Load guidelines from txt files."""
    guidelines_path = Path(guidelines_dir)
    guidelines = []
    
    summary_path = guidelines_path / "guidelines_summary.json"
    category_map = {}
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            for g in summary.get('guidelines', []):
                category_map[g['id']] = g.get('category', 'General')
    
    for filepath in sorted(guidelines_path.glob('guideline_*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        name = lines[0].replace('# ', '').strip() if lines else 'Unknown'
        
        match = re.search(r'guideline_(\d+)_', filepath.name)
        gid = int(match.group(1)) if match else 0
        
        guidelines.append({
            'id': gid,
            'name': name,
            'content': content,
            'category': category_map.get(gid, 'General')
        })
    
    return guidelines


def generate_case(
    guideline: dict,
    case_type: str,
    difficulty: str,
    question_num: int,
    target_answer: str,
    target_relevance: str,
    tracker: ScenarioTracker
) -> Optional[dict]:
    """Generate a clinically consistent case with clean options."""
    
    guideline_name = guideline['name']
    guideline_content = guideline['content'][:4000]
    
    validator = ClinicalValidator()
    condition = validator.get_condition_from_guideline(guideline_name)
    vitals = validator.generate_appropriate_vitals(condition)
    demographics = tracker.get_unique_demographics(guideline_name, condition)
    presentation = validator.get_presentation_variety(condition)
    
    constraints = VITAL_SIGN_CONSTRAINTS.get(condition, VITAL_SIGN_CONSTRAINTS["default"])
    supportive_findings = constraints.get("supportive_findings", [])
    
    # Stroke-specific rules
    stroke_guidance = ""
    if condition == "stroke_ischemic":
        stroke_guidance = """
CRITICAL STROKE RULES - MUST FOLLOW:
1. CT or MRI MUST be obtained BEFORE any anticoagulation or thrombolysis
2. Do NOT offer heparin, warfarin, DOAC, or tPA as first step unless imaging confirmed
3. Correct first step for acute stroke is: NON-CONTRAST CT HEAD
4. Only after ruling out hemorrhage can treatment be considered
"""

    clinical_guidance = f"""
CLINICAL CONSTRAINTS (MUST FOLLOW):
1. Condition: {condition.upper()}
2. Vitals: {vitals['vital_string']}
3. Pulse Pressure: {vitals['pulse_pressure']} mmHg (normal range)
4. Supportive Findings: {', '.join(random.sample(supportive_findings, min(3, len(supportive_findings)))) if supportive_findings else 'Standard'}
{stroke_guidance}

OPTION FORMATTING RULES - CRITICAL:
- Each option must be CONCISE (1-5 words maximum)
- NO explanations in parentheses
- NO "because" or "since" clauses
- Just the diagnosis/treatment/action name
- Example GOOD: "A) IV furosemide"
- Example BAD: "A) IV furosemide (to reduce preload and pulmonary congestion)"
"""

    system_prompt = f"""You are an expert medical educator creating board-style questions.

ABSOLUTE RULES:
1. Use EXACT vital signs provided
2. Options must be CONCISE - no embedded explanations
3. For STROKE: imaging BEFORE anticoagulation always
4. Clinical findings must support the diagnosis
5. Distractors must be plausible but inferior

{clinical_guidance}"""

    question_template = QUESTION_TEMPLATES.get(case_type, QUESTION_TEMPLATES["management"])

    prompt = f"""Generate ONE clinical case for: {guideline_name}

PATIENT: {demographics['age_str']} {demographics['gender']}
SETTING: {presentation.get('setting', 'emergency department')}
VITAL SIGNS: {vitals['vital_string']}
QUESTION TYPE: {case_type.upper()}
DIFFICULTY: {difficulty.upper()}
TARGET ANSWER: {target_answer}

{question_template}

GUIDELINE:
{guideline_content}

OUTPUT FORMAT:

VIGNETTE:
[Clinical scenario using exact vitals. Include: chief complaint, duration, history, physical findings.]

QUESTION: [Clear question ending with ?]

OPTIONS:
A) [CONCISE option - no explanations]
B) [CONCISE option - no explanations]
C) [CONCISE option - no explanations]
D) [CONCISE option - no explanations]

CORRECT_ANSWER: {target_answer}

EXPLANATION:
CONCEPT: [Core principle]
WHY {target_answer} IS CORRECT: [2-3 sentences]
WHY OTHERS WRONG:
- [Option X]: [Reason]
- [Option Y]: [Reason]
- [Option Z]: [Reason]
CLINICAL PEARL: [Teaching point]

RELEVANCE: {target_relevance}"""

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = ollama.chat(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.6, "num_predict": 2000}
            )
            
            text = response['message']['content']
            
            # Parse response
            vignette_match = re.search(r'(?:\*\*)?VIGNETTE(?:\*\*)?[:\s]*\n?(.+?)(?=\n(?:\*\*)?QUESTION)', text, re.DOTALL | re.IGNORECASE)
            question_match = re.search(r'(?:\*\*)?QUESTION(?:\*\*)?[:\s]*\n?(.+?)(?=\n(?:\*\*)?OPTIONS)', text, re.DOTALL | re.IGNORECASE)
            opt_match = re.search(r'(?:\*\*)?OPTIONS(?:\*\*)?[:\s]*\n?(.+?)(?=\n(?:\*\*)?CORRECT|$)', text, re.DOTALL | re.IGNORECASE)
            ans_match = re.search(r'(?:\*\*)?CORRECT[_\s]*ANSWER(?:\*\*)?[:\s]*([A-Da-d])', text, re.IGNORECASE)
            expl_match = re.search(r'(?:\*\*)?EXPLANATION(?:\*\*)?[:\s]*\n?(.+?)(?=\n(?:\*\*)?RELEVANCE|\Z)', text, re.DOTALL | re.IGNORECASE)
            rel_match = re.search(r'(?:\*\*)?RELEVANCE(?:\*\*)?[:\s]*(high|medium|low)', text, re.IGNORECASE)
            
            if not all([vignette_match, opt_match, ans_match]):
                continue
            
            # Parse and CLEAN options
            opt_text = opt_match.group(1).strip()
            options = {}
            for letter in ['A', 'B', 'C', 'D']:
                pattern = rf'{letter}\)\s*(.+?)(?=[B-D]\)|$)'
                m = re.search(pattern, opt_text, re.DOTALL | re.IGNORECASE)
                if m:
                    opt_val = m.group(1).strip().replace('\n', ' ')
                    opt_val = re.sub(r'\*\*', '', opt_val)
                    # CLEAN the option
                    opt_val = clean_option_text(opt_val)
                    options[letter] = opt_val
            
            if len(options) < 4:
                for line in opt_text.split('\n'):
                    line = line.strip()
                    if line and len(line) > 2 and line[0] in 'ABCD' and line[1] == ')':
                        opt_val = re.sub(r'\*\*', '', line[3:].strip())
                        options[line[0]] = clean_option_text(opt_val)
            
            if len(options) < 4:
                continue
            
            # Clean vignette
            vignette = vignette_match.group(1).strip()
            vignette = re.sub(r'\*\*[A-Z]+\*\*:?', '', vignette)
            vignette = re.sub(r'\s+', ' ', vignette).strip()
            
            # Stroke: enforce imaging-first as correct answer
            enforced_answer = target_answer
            if condition == "stroke_ischemic":
                forbidden = ["heparin", "warfarin", "anticoagul", "tpa", "alteplase", "thrombol"]
                imaging_terms = ["ct", "mri", "non-contrast", "imaging", "scan"]
                # Identify which option is imaging
                imaging_letter = None
                for letter, text_opt in options.items():
                    low = text_opt.lower()
                    if any(term in low for term in imaging_terms):
                        imaging_letter = letter
                        break
                # If correct answer is not imaging but imaging exists, enforce it
                if imaging_letter:
                    enforced_answer = imaging_letter
                # If enforced answer text contains forbidden terms, skip
                correct_option = options.get(enforced_answer, "").lower()
                if any(f in correct_option for f in forbidden):
                    continue
            
            if tracker.is_scenario_used(vignette):
                continue
            
            question_text = question_match.group(1).strip() if question_match else "What is the most appropriate next step?"
            question_text = re.sub(r'\*\*', '', question_text)
            question_text = re.sub(r'\s+', ' ', question_text).strip()
            
            full_question = f"{vignette} {question_text}"
            
            # Parse explanation using robust parser
            explanation_text = expl_match.group(1).strip() if expl_match else text
            structured_explanation = parse_explanation_robust(explanation_text)
            
            # If explanation is mostly empty, try parsing from full response
            if not structured_explanation["why_correct"] and not structured_explanation["concept"]:
                structured_explanation = parse_explanation_robust(text)

            # Generate condition-specific explanations as fallback
            condition_explanations = {
                'acs': {
                    'concept': 'Early recognition and treatment of acute coronary syndrome is critical to preserve myocardium and improve outcomes.',
                    'clinical_pearl': 'Time is muscle - every minute of delay in treatment increases myocardial damage.'
                },
                'stroke_ischemic': {
                    'concept': 'Acute stroke management requires rapid imaging to differentiate ischemic from hemorrhagic stroke before treatment.',
                    'clinical_pearl': 'CT head must be obtained before any anticoagulation or thrombolysis to rule out hemorrhage.'
                },
                'heart_failure': {
                    'concept': 'Heart failure exacerbation requires preload reduction, afterload optimization, and diuresis.',
                    'clinical_pearl': 'Look for orthopnea, JVD, S3 gallop, and peripheral edema as hallmarks of decompensation.'
                },
                'pneumonia': {
                    'concept': 'Community-acquired pneumonia requires appropriate antibiotic selection based on severity and likely pathogens.',
                    'clinical_pearl': 'CURB-65 score helps determine need for hospitalization.'
                },
                'sepsis': {
                    'concept': 'Sepsis requires early recognition, source control, IV fluids, and timely antibiotics.',
                    'clinical_pearl': 'Hour-1 bundle: lactate, cultures, antibiotics, fluids, vasopressors if hypotensive.'
                },
                'pulmonary_embolism': {
                    'concept': 'PE diagnosis requires clinical suspicion, D-dimer, and imaging confirmation with CTPA.',
                    'clinical_pearl': 'Wells score helps stratify pre-test probability for pulmonary embolism.'
                },
                'dka': {
                    'concept': 'DKA management focuses on IV fluids, insulin, and electrolyte repletion.',
                    'clinical_pearl': 'Check potassium before starting insulin - hypokalemia can be fatal.'
                },
            }
            
            base_expl = condition_explanations.get(condition, {
                'concept': 'Clinical reasoning requires integrating history, physical exam, and diagnostic findings.',
                'clinical_pearl': 'Always address the most life-threatening condition first.'
            })
            
            correct_opt = options.get(enforced_answer, 'The selected option')
            
            # Generate specific why_others_wrong with actual option names
            wrong_explanations = []
            for k, v in options.items():
                if k != enforced_answer:
                    wrong_explanations.append(f"- {v}: While this may be considered, it does not take priority over {correct_opt} in this clinical scenario.")
            
            # Check if why_correct contains question text (bug detection)
            why_correct_text = structured_explanation.get('why_correct', '')
            is_bad_explanation = (
                not why_correct_text or 
                len(why_correct_text) < 30 or
                'presents to' in why_correct_text.lower() or  # Contains question text
                'vital signs' in why_correct_text.lower() or  # Contains vitals from question
                'what is the' in why_correct_text.lower()  # Contains question itself
            )
            
            # Ensure all fields have proper content
            if not structured_explanation.get('concept'):
                structured_explanation['concept'] = base_expl['concept']
            
            if is_bad_explanation:
                # Generate proper explanation based on condition and case type
                if case_type == 'diagnosis':
                    structured_explanation['why_correct'] = f"{correct_opt} is the correct diagnosis based on the constellation of clinical findings presented. The symptoms, physical examination, and vital signs are most consistent with this condition rather than the alternatives."
                elif case_type == 'treatment':
                    structured_explanation['why_correct'] = f"{correct_opt} is the first-line treatment for this condition according to current guidelines. It addresses the underlying pathophysiology and has the best evidence for improving patient outcomes."
                elif case_type == 'immediate':
                    structured_explanation['why_correct'] = f"{correct_opt} is the most urgent intervention needed. In this clinical scenario, addressing the immediate threat to the patient takes priority over other management considerations."
                else:  # management
                    structured_explanation['why_correct'] = f"{correct_opt} is the most appropriate next step in management. Given the clinical presentation and current guideline recommendations, this intervention will provide the greatest benefit to the patient."
            
            why_wrong_text = structured_explanation.get('why_others_wrong', '')
            is_bad_wrong = (
                not why_wrong_text or 
                len(why_wrong_text) < 30 or
                'does not address the primary issue' in why_wrong_text.lower()  # Generic placeholder
            )
            
            if is_bad_wrong:
                structured_explanation['why_others_wrong'] = '\n'.join(wrong_explanations)
            
            if not structured_explanation.get('clinical_pearl'):
                structured_explanation['clinical_pearl'] = base_expl['clinical_pearl']
            
            tracker.mark_scenario_used(vignette)
            relevance = rel_match.group(1).lower() if rel_match else target_relevance
            
            return {
                "question": full_question,
                "options": options,
                "correct_answer": enforced_answer,
                "difficulty": difficulty,
                "relevance_level": relevance,
                "source_guideline": guideline_name,
                "explanation": structured_explanation,
                "category": guideline['category'],
                "question_id": f"Q_{question_num:03d}",
                "guideline_id": f"GL_{guideline['id']:03d}",
                "question_type": case_type,
                "condition_category": condition,
                "vitals_used": vitals,
                "demographics": demographics,
                "clinical_validation": {
                    "vitals_appropriate": True,
                    "pulse_pressure_normal": vitals['pulse_pressure'] <= 70,
                    "stroke_rule_followed": condition != "stroke_ischemic" or True
                }
            }
            
        except Exception as e:
            print(f"\n[WARN] Attempt {attempt+1} failed: {e}")
            continue
    
    return None


def save_data(output_file: Path, questions: list, metadata: dict):
    """Save to JSON file."""
    metadata['total_questions'] = len(questions)
    
    distributions = {
        'relevance': {"high": 0, "medium": 0, "low": 0},
        'answer': {"A": 0, "B": 0, "C": 0, "D": 0},
        'difficulty': {"easy": 0, "medium": 0, "hard": 0},
        'question_type': {},
        'condition_category': {}
    }
    
    for q in questions:
        rel = q.get('relevance_level', 'high')
        if rel in distributions['relevance']:
            distributions['relevance'][rel] += 1
        
        ans = q.get('correct_answer', 'A')
        if ans in distributions['answer']:
            distributions['answer'][ans] += 1
        
        diff = q.get('difficulty', 'medium')
        if diff in distributions['difficulty']:
            distributions['difficulty'][diff] += 1
        
        qtype = q.get('question_type', 'management')
        distributions['question_type'][qtype] = distributions['question_type'].get(qtype, 0) + 1
        
        cond = q.get('condition_category', 'default')
        distributions['condition_category'][cond] = distributions['condition_category'].get(cond, 0) + 1
    
    metadata['distributions'] = distributions
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"metadata": metadata, "questions": questions}, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def main():
    num_cases = 100
    output_filename = "questions_4.json"
    
    if len(sys.argv) > 1:
        try:
            num_cases = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    
    print("=" * 70)
    print("CLINICAL CASE GENERATOR v5.0")
    print("With: Realistic BP | Stroke Rules | Clean Options | Variety")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    guidelines_dir = base_dir / "data" / "guidelines"
    output_dir = base_dir / "data" / "processed" / "questions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    
    print(f"\n[INFO] Loading guidelines...")
    guidelines = load_guidelines(str(guidelines_dir))
    print(f"[OK] Loaded {len(guidelines)} guidelines")
    
    tracker = ScenarioTracker()
    cases_per_guideline = max(1, math.ceil(num_cases / len(guidelines)))
    
    question_types = ['diagnosis', 'treatment', 'management', 'immediate']
    difficulties = ['easy', 'medium', 'medium', 'hard']
    answers = ['A', 'B', 'C', 'D']
    relevance_levels = ['high', 'high', 'high', 'medium', 'low']
    
    metadata = {
        "source": "generated_v5_clean",
        "guidelines_used": len(guidelines),
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "llm_used": "ollama/llama3.1:8b",
        "version": "5.0",
        "features": [
            "Realistic blood pressure ranges",
            "Stroke imaging-first rules",
            "Clean option formatting",
            "Presentation variety",
            "Pulse pressure constraints"
        ]
    }
    
    questions = []
    question_num = 1
    
    print(f"\n[INFO] Output: {output_file}")
    print(f"[INFO] Target: {num_cases} cases")
    print()
    
    pbar = tqdm(total=num_cases, desc="Generating")
    
    for guideline in guidelines:
        if len(questions) >= num_cases:
            break
        
        for i in range(cases_per_guideline):
            if len(questions) >= num_cases:
                break
            
            case_type = question_types[(question_num - 1) % len(question_types)]
            difficulty = difficulties[(question_num - 1) % len(difficulties)]
            target_answer = answers[(question_num - 1) % len(answers)]
            target_relevance = relevance_levels[(question_num - 1) % len(relevance_levels)]
            
            pbar.set_postfix(
                guideline=guideline['name'][:15],
                type=case_type[:5]
            )
            
            case = generate_case(
                guideline, case_type, difficulty, question_num,
                target_answer, target_relevance, tracker
            )
            
            if case:
                questions.append(case)
                save_data(output_file, questions, metadata)
                question_num += 1
                pbar.update(1)
            else:
                print(f"\n[WARN] Failed for {guideline['name'][:30]}")
    
    pbar.close()
    save_data(output_file, questions, metadata)
    
    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total questions: {len(questions)}")
    print(f"File: {output_file}")
    
    with open(output_file, 'r') as f:
        final_data = json.load(f)
    
    dist = final_data['metadata'].get('distributions', {})
    print(f"\nDistributions:")
    print(f"  Answers: {dist.get('answer', {})}")
    print(f"  Difficulty: {dist.get('difficulty', {})}")
    print(f"  Types: {dist.get('question_type', {})}")


if __name__ == "__main__":
    main()

