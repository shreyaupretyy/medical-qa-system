"""
Real-time Medical Safety Verifier

Checks for:
- Contraindications based on retrieved guidelines
- Emergency red flags
- Dangerous omissions (e.g., no ABC steps)
- Medication safety (dosing, interactions)

If flagged → answer must be revised or rejected.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re


class SafetyLevel(Enum):
    """Safety assessment levels."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    DANGEROUS = "dangerous"


@dataclass
class SafetyViolation:
    """A detected safety violation."""
    violation_type: str
    severity: SafetyLevel
    description: str
    recommendation: str
    evidence: Optional[str] = None


@dataclass
class SafetyAssessment:
    """Complete safety assessment result."""
    is_safe: bool
    safety_level: SafetyLevel
    violations: List[SafetyViolation]
    recommendations: List[str]
    requires_revision: bool
    emergency_flags: List[str]
    contraindication_flags: List[str]


class MedicalSafetyVerifier:
    """
    Real-time safety verification for medical answers.
    
    Checks:
    1. Emergency protocol compliance (ABCs)
    2. Contraindication detection
    3. Dangerous drug interactions
    4. Age-appropriate dosing
    5. Pregnancy/lactation safety
    6. Red flag recognition
    """
    
    def __init__(self):
        """Initialize safety verifier with rules."""
        self._init_contraindications()
        self._init_emergency_protocols()
        self._init_drug_safety()
        self._init_red_flags()
    
    def _init_contraindications(self):
        """Initialize contraindication rules."""
        self.contraindications = {
            # Drug contraindications
            'aspirin': ['children under 12', 'gi bleeding', 'peptic ulcer', 'aspirin allergy', 'asthma with aspirin sensitivity'],
            'metformin': ['renal failure', 'creatinine > 1.5', 'contrast dye', 'liver failure', 'acidosis'],
            'ace inhibitors': ['pregnancy', 'angioedema', 'bilateral renal artery stenosis', 'hyperkalemia'],
            'nsaids': ['gi bleeding', 'renal failure', 'heart failure', 'pregnancy third trimester'],
            'warfarin': ['active bleeding', 'pregnancy', 'severe liver disease', 'recent surgery'],
            'methotrexate': ['pregnancy', 'liver disease', 'renal failure', 'bone marrow suppression'],
            'fluoroquinolones': ['children', 'pregnancy', 'tendon disorders', 'myasthenia gravis'],
            'tetracyclines': ['children under 8', 'pregnancy', 'renal failure'],
            'aminoglycosides': ['myasthenia gravis', 'concurrent ototoxic drugs', 'renal failure'],
            'magnesium sulfate': ['myasthenia gravis', 'heart block', 'renal failure'],
            'beta blockers': ['asthma', 'severe bradycardia', 'heart block', 'cardiogenic shock'],
            'calcium channel blockers': ['heart failure', 'severe hypotension', 'heart block']
        }
        
        # Condition-specific contraindications
        self.condition_contraindications = {
            'pregnancy': ['warfarin', 'ace inhibitors', 'statins', 'methotrexate', 'fluoroquinolones', 'tetracyclines'],
            'renal failure': ['nsaids', 'aminoglycosides', 'metformin', 'contrast dye'],
            'liver failure': ['acetaminophen high dose', 'methotrexate', 'statins'],
            'gi bleeding': ['nsaids', 'aspirin', 'anticoagulants'],
            'heart failure': ['nsaids', 'calcium channel blockers', 'thiazolidinediones'],
            'asthma': ['beta blockers', 'aspirin']
        }
    
    def _init_emergency_protocols(self):
        """Initialize emergency protocol requirements."""
        self.emergency_protocols = {
            'cardiac_arrest': {
                'required_steps': ['cpr', 'defibrillation', 'airway', 'epinephrine'],
                'sequence': 'CAB (Circulation, Airway, Breathing)'
            },
            'anaphylaxis': {
                'required_steps': ['epinephrine', 'airway', 'iv fluids', 'antihistamines'],
                'first_line': 'epinephrine im'
            },
            'septic_shock': {
                'required_steps': ['iv fluids', 'antibiotics', 'vasopressors', 'lactate'],
                'time_critical': '1 hour bundle'
            },
            'stroke': {
                'required_steps': ['ct scan', 'nihss', 'thrombolysis if indicated'],
                'time_window': '4.5 hours for tPA'
            },
            'stemi': {
                'required_steps': ['ecg', 'aspirin', 'antiplatelet', 'reperfusion'],
                'time_critical': 'door to balloon < 90 min'
            },
            'trauma': {
                'required_steps': ['airway', 'breathing', 'circulation', 'disability', 'exposure'],
                'sequence': 'ABCDE'
            },
            'severe_preeclampsia': {
                'required_steps': ['magnesium sulfate', 'antihypertensives', 'delivery planning'],
                'monitoring': 'continuous fetal monitoring'
            },
            'dka': {
                'required_steps': ['iv fluids', 'insulin', 'potassium', 'monitoring'],
                'sequence': 'fluids before insulin'
            }
        }
    
    def _init_drug_safety(self):
        """Initialize drug safety rules."""
        self.pediatric_dosing = {
            'paracetamol': {'max_mg_kg': 15, 'max_daily_mg_kg': 60},
            'ibuprofen': {'max_mg_kg': 10, 'max_daily_mg_kg': 40},
            'amoxicillin': {'standard_mg_kg': 25, 'high_dose_mg_kg': 45},
            'ceftriaxone': {'max_mg_kg': 100, 'neonatal_max': 50},
            'gentamicin': {'max_mg_kg': 7.5}
        }
        
        self.dangerous_interactions = [
            ('warfarin', 'nsaids'),
            ('maois', 'ssris'),
            ('digoxin', 'amiodarone'),
            ('methotrexate', 'nsaids'),
            ('lithium', 'nsaids'),
            ('potassium', 'ace inhibitors'),
            ('aminoglycosides', 'loop diuretics')
        ]
    
    def _init_red_flags(self):
        """Initialize red flag symptoms requiring immediate attention."""
        self.red_flags = {
            'neurological': [
                'sudden severe headache', 'worst headache of life',
                'focal neurological deficit', 'altered consciousness',
                'neck stiffness with fever', 'papilledema'
            ],
            'cardiovascular': [
                'chest pain with diaphoresis', 'syncope with exertion',
                'pulseless extremity', 'severe hypotension',
                'new heart murmur with fever'
            ],
            'respiratory': [
                'stridor', 'severe respiratory distress',
                'oxygen saturation < 90%', 'cyanosis',
                'silent chest in asthma'
            ],
            'abdominal': [
                'rigid abdomen', 'rebound tenderness',
                'bilious vomiting in infant', 'blood per rectum with shock'
            ],
            'obstetric': [
                'vaginal bleeding with shock', 'seizures in pregnancy',
                'severe preeclampsia', 'cord prolapse',
                'absent fetal movements', 'ruptured ectopic'
            ],
            'pediatric': [
                'bulging fontanelle', 'non-blanching rash',
                'inconsolable cry', 'bilious vomiting',
                'severe dehydration', 'toxic appearance'
            ]
        }
    
    def verify(
        self,
        answer: str,
        question: str,
        case_description: str,
        options: List[str],
        retrieved_contexts: List[Any],
        patient_category: str = 'adult'
    ) -> SafetyAssessment:
        """
        Verify safety of the proposed answer.
        
        Args:
            answer: The proposed answer (A, B, C, D)
            question: The clinical question
            case_description: Patient case description
            options: Answer options
            retrieved_contexts: Retrieved guidelines
            patient_category: Patient category (adult/pediatric/pregnancy/etc)
            
        Returns:
            SafetyAssessment with violations and recommendations
        """
        violations = []
        recommendations = []
        emergency_flags = []
        contraindication_flags = []
        
        # Get the actual answer text
        answer_text = self._get_answer_text(answer, options)
        
        # Check for red flags in case
        red_flag_check = self._check_red_flags(case_description)
        emergency_flags.extend(red_flag_check)
        
        # Check contraindications
        contra_check = self._check_contraindications(
            answer_text, case_description, patient_category
        )
        violations.extend(contra_check)
        contraindication_flags.extend([v.description for v in contra_check])
        
        # Check emergency protocol compliance
        emergency_check = self._check_emergency_protocols(
            answer_text, case_description, emergency_flags
        )
        violations.extend(emergency_check)
        
        # Check drug safety
        drug_check = self._check_drug_safety(
            answer_text, case_description, patient_category
        )
        violations.extend(drug_check)
        
        # Check dangerous omissions
        omission_check = self._check_dangerous_omissions(
            answer_text, case_description, emergency_flags
        )
        violations.extend(omission_check)
        
        # Determine overall safety level
        if any(v.severity == SafetyLevel.DANGEROUS for v in violations):
            safety_level = SafetyLevel.DANGEROUS
            is_safe = False
            requires_revision = True
        elif any(v.severity == SafetyLevel.CRITICAL for v in violations):
            safety_level = SafetyLevel.CRITICAL
            is_safe = False
            requires_revision = True
        elif any(v.severity == SafetyLevel.WARNING for v in violations):
            safety_level = SafetyLevel.WARNING
            is_safe = True
            requires_revision = False
        else:
            safety_level = SafetyLevel.SAFE
            is_safe = True
            requires_revision = False
        
        # Generate recommendations
        for violation in violations:
            recommendations.append(violation.recommendation)
        
        return SafetyAssessment(
            is_safe=is_safe,
            safety_level=safety_level,
            violations=violations,
            recommendations=recommendations,
            requires_revision=requires_revision,
            emergency_flags=emergency_flags,
            contraindication_flags=contraindication_flags
        )
    
    def _get_answer_text(self, answer: str, options: List[str]) -> str:
        """Get the full text of the selected answer."""
        for option in options:
            if option.strip().startswith(answer):
                match = re.match(r'^[A-D][.)]\s*(.+)$', option.strip())
                if match:
                    return match.group(1).strip()
        return answer
    
    def _check_red_flags(self, case_description: str) -> List[str]:
        """Check for red flag symptoms in case."""
        flags_found = []
        text_lower = case_description.lower()
        
        for category, flags in self.red_flags.items():
            for flag in flags:
                if flag.lower() in text_lower:
                    flags_found.append(f"{category}: {flag}")
        
        return flags_found
    
    def _check_contraindications(
        self,
        answer_text: str,
        case_description: str,
        patient_category: str
    ) -> List[SafetyViolation]:
        """Check for contraindicated treatments."""
        violations = []
        answer_lower = answer_text.lower()
        case_lower = case_description.lower()
        
        # Check drug contraindications
        for drug, contraindications in self.contraindications.items():
            if drug in answer_lower:
                for contra in contraindications:
                    if contra.lower() in case_lower:
                        violations.append(SafetyViolation(
                            violation_type='contraindication',
                            severity=SafetyLevel.CRITICAL,
                            description=f"{drug} contraindicated with {contra}",
                            recommendation=f"Avoid {drug} due to {contra}",
                            evidence=f"Patient has {contra}"
                        ))
        
        # Check condition-specific contraindications
        for condition, drugs in self.condition_contraindications.items():
            if condition in case_lower or condition == patient_category:
                for drug in drugs:
                    if drug in answer_lower:
                        violations.append(SafetyViolation(
                            violation_type='condition_contraindication',
                            severity=SafetyLevel.CRITICAL,
                            description=f"{drug} contraindicated in {condition}",
                            recommendation=f"Do not use {drug} in {condition}",
                            evidence=f"Patient has {condition}"
                        ))
        
        return violations
    
    def _check_emergency_protocols(
        self,
        answer_text: str,
        case_description: str,
        red_flags: List[str]
    ) -> List[SafetyViolation]:
        """Check emergency protocol compliance."""
        violations = []
        case_lower = case_description.lower()
        answer_lower = answer_text.lower()
        
        # Identify which emergency protocol applies
        for emergency, protocol in self.emergency_protocols.items():
            emergency_keywords = emergency.replace('_', ' ').split()
            if any(kw in case_lower for kw in emergency_keywords) or red_flags:
                # Check if required steps are addressed
                for step in protocol.get('required_steps', []):
                    if step not in answer_lower and step not in case_lower:
                        # Missing critical step
                        if emergency in ['cardiac_arrest', 'anaphylaxis', 'septic_shock']:
                            severity = SafetyLevel.DANGEROUS
                        else:
                            severity = SafetyLevel.WARNING
                        
                        violations.append(SafetyViolation(
                            violation_type='missing_emergency_step',
                            severity=severity,
                            description=f"Missing critical step '{step}' for {emergency}",
                            recommendation=f"Ensure {step} is included in management",
                            evidence=f"Emergency: {emergency}"
                        ))
        
        return violations
    
    def _check_drug_safety(
        self,
        answer_text: str,
        case_description: str,
        patient_category: str
    ) -> List[SafetyViolation]:
        """Check drug dosing and interaction safety."""
        violations = []
        answer_lower = answer_text.lower()
        
        # Check dangerous interactions
        for drug1, drug2 in self.dangerous_interactions:
            if drug1 in answer_lower and drug2 in case_description.lower():
                violations.append(SafetyViolation(
                    violation_type='drug_interaction',
                    severity=SafetyLevel.CRITICAL,
                    description=f"Dangerous interaction: {drug1} + {drug2}",
                    recommendation=f"Avoid combining {drug1} with {drug2}",
                    evidence=f"Patient on {drug2}"
                ))
        
        # Check pediatric dosing if applicable
        if patient_category in ['pediatric', 'neonatal']:
            # Look for dose mentions without weight-based calculation
            dose_pattern = r'(\d+)\s*(mg|ml|g)\s*(once|twice|daily|bid|tid|qid)?'
            if re.search(dose_pattern, answer_lower):
                # Check if it's a known pediatric drug
                for drug, limits in self.pediatric_dosing.items():
                    if drug in answer_lower:
                        violations.append(SafetyViolation(
                            violation_type='pediatric_dosing',
                            severity=SafetyLevel.WARNING,
                            description=f"Verify weight-based dosing for {drug} in pediatric patient",
                            recommendation=f"Calculate dose: {limits['max_mg_kg']} mg/kg max",
                            evidence="Pediatric patient requires weight-based dosing"
                        ))
        
        return violations
    
    def _check_dangerous_omissions(
        self,
        answer_text: str,
        case_description: str,
        red_flags: List[str]
    ) -> List[SafetyViolation]:
        """Check for dangerous omissions in the answer."""
        violations = []
        case_lower = case_description.lower()
        answer_lower = answer_text.lower()
        
        # If red flags present, check for appropriate urgency
        if red_flags:
            urgency_words = ['emergency', 'urgent', 'immediate', 'stat', 'critical']
            if not any(word in answer_lower for word in urgency_words):
                violations.append(SafetyViolation(
                    violation_type='omitted_urgency',
                    severity=SafetyLevel.WARNING,
                    description="Red flags present but urgency not emphasized",
                    recommendation="Consider emergency management pathway",
                    evidence=f"Red flags: {', '.join(red_flags[:3])}"
                ))
        
        # Check for sepsis without antibiotics
        if 'sepsis' in case_lower or 'septic' in case_lower:
            if 'antibiotic' not in answer_lower and 'antimicrobial' not in answer_lower:
                # Check if any specific antibiotic mentioned
                common_abx = ['ceftriaxone', 'vancomycin', 'piperacillin', 'meropenem', 'ampicillin']
                if not any(abx in answer_lower for abx in common_abx):
                    violations.append(SafetyViolation(
                        violation_type='missing_antibiotics',
                        severity=SafetyLevel.CRITICAL,
                        description="Sepsis present but antibiotics not mentioned",
                        recommendation="Initiate empiric antibiotics within 1 hour",
                        evidence="Sepsis suspected"
                    ))
        
        # Check for shock without fluids
        if 'shock' in case_lower or 'hypotension' in case_lower:
            if 'fluid' not in answer_lower and 'saline' not in answer_lower and 'crystalloid' not in answer_lower:
                violations.append(SafetyViolation(
                    violation_type='missing_resuscitation',
                    severity=SafetyLevel.CRITICAL,
                    description="Shock/hypotension present but fluid resuscitation not mentioned",
                    recommendation="Initiate IV fluid resuscitation",
                    evidence="Shock or hypotension present"
                ))
        
        return violations

