# Data Creation Documentation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [PDF Extractor](#pdf-extractor)
3. [Guideline Generator](#guideline-generator)
4. [Question Generator](#question-generator)
5. [Complete Pipeline](#complete-pipeline)
6. [Quality Controls](#quality-controls)

---

## Overview

The `src/data_creation/` module provides tools for building high-quality medical question-answering datasets from clinical guidelines. The pipeline converts PDF treatment guidelines into structured formats and generates realistic clinical cases with quality controls.

**Components:**
- `pdf_extractor.py`: Extracts text from PDF medical guidelines
- `guideline_generator.py`: Structures raw text into standardized guideline format
- `question_generator.py`: Generates clinical MCQ cases from guidelines

**Pipeline Flow:**
```
PDF → Raw Text → Structured Guidelines → Clinical Cases → Quality Validation
```

---

## PDF Extractor

**File:** `src/data_creation/pdf_extractor.py`

### Purpose

Extracts clean, structured text from medical guideline PDFs while preserving section hierarchy and removing noise (headers, footers, page numbers).

### Key Classes

#### `PDFExtractor`

**Constructor:**
```python
class PDFExtractor:
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor.
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path)
```

**Methods:**

##### `extract_text() -> str`

Extracts all text from PDF with cleaning.

```python
def extract_text(self) -> str:
    """
    Extract text from all pages of PDF.
    
    Returns:
        Cleaned text content
        
    Process:
        1. Iterate through all pages
        2. Extract text with pdfplumber
        3. Remove headers/footers
        4. Clean whitespace
        5. Concatenate pages
    """
```

**Example Usage:**
```python
from src.data_creation.pdf_extractor import PDFExtractor

extractor = PDFExtractor("data/standard-treatment-guidelines.pdf")
raw_text = extractor.extract_text()

# Output: ~200,000 characters of clinical guidelines
print(f"Extracted {len(raw_text)} characters")
```

##### `extract_by_sections() -> Dict[str, str]`

Extracts text organized by detected sections.

```python
def extract_by_sections(self) -> Dict[str, str]:
    """
    Extract text with section detection.
    
    Returns:
        Dictionary mapping section titles to content
        
    Section Detection:
        - Identifies headings by font size, boldness
        - Tracks section hierarchy
        - Preserves subsection relationships
    """
```

**Example Output:**
```python
{
    "Cardiovascular Emergencies": "...",
    "Acute Coronary Syndrome": "...",
    "Diagnosis": "...",
    "Treatment": "..."
}
```

---

### Text Cleaning Pipeline

**Process:**
```python
def clean_text(text: str) -> str:
    """
    Clean extracted text.
    
    Steps:
        1. Remove page numbers (e.g., "Page 45")
        2. Remove headers/footers (repeated text)
        3. Fix hyphenation across lines
        4. Normalize whitespace
        5. Remove non-ASCII characters
    """
    # Remove page numbers
    text = re.sub(r'Page \d+', '', text)
    
    # Fix hyphenation
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

---

### Advanced Features

#### Table Extraction

```python
def extract_tables(self) -> List[pd.DataFrame]:
    """
    Extract tables from PDF.
    
    Returns:
        List of DataFrames, one per table
        
    Use Case:
        - Medication dosing tables
        - Diagnostic criteria tables
        - Reference ranges
    """
    tables = []
    for page in self.pdf.pages:
        page_tables = page.extract_tables()
        for table in page_tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            tables.append(df)
    return tables
```

#### Image Extraction

```python
def extract_images(self, output_dir: str):
    """
    Extract images from PDF (e.g., ECGs, X-rays).
    
    Args:
        output_dir: Directory to save images
        
    Process:
        1. Iterate through pages
        2. Extract image objects
        3. Save as PNG files
        4. Generate metadata JSON
    """
```

---

## Guideline Generator

**File:** `src/data_creation/guideline_generator.py`

### Purpose

Converts raw extracted text into structured clinical guidelines using LLM-based organization and template application.

### Key Classes

#### `GuidelineGenerator`

**Constructor:**
```python
class GuidelineGenerator:
    def __init__(self, ollama_model: str = "llama3.1:8b"):
        """
        Initialize guideline generator.
        
        Args:
            ollama_model: Ollama model name for structuring
        """
        self.model = ollama_model
        self.client = ollama.Client()
```

**Methods:**

##### `generate_from_text(raw_text: str, topics: List[str]) -> List[Dict]`

Generates structured guidelines from raw text.

```python
def generate_from_text(
    self,
    raw_text: str,
    topics: List[str]
) -> List[Dict]:
    """
    Generate structured guidelines from raw text.
    
    Args:
        raw_text: Extracted PDF text
        topics: List of clinical topics to extract
        
    Returns:
        List of structured guideline dictionaries
        
    Process:
        1. For each topic:
           a. Extract relevant text sections
           b. Use LLM to structure into template
           c. Validate completeness
           d. Post-process formatting
        2. Return structured guidelines
    """
```

**Example Usage:**
```python
from src.data_creation.guideline_generator import GuidelineGenerator

generator = GuidelineGenerator()

topics = [
    "Acute Coronary Syndrome",
    "Stroke Management",
    "Diabetes Management"
]

guidelines = generator.generate_from_text(raw_text, topics)

for guideline in guidelines:
    print(f"Generated: {guideline['title']}")
```

---

### Guideline Template

**Structure:**
```
GUIDELINE: {Title}
CATEGORY: {Medical Category}

DEFINITION:
{Comprehensive definition, epidemiology, pathophysiology}

DIAGNOSIS:
{Diagnostic criteria, clinical presentation, laboratory findings, imaging}

TREATMENT:
{Immediate management, medication protocols, procedures, dosing}

MANAGEMENT:
{Long-term management, monitoring, complications, patient education}
```

**Template Enforcement:**
```python
def validate_guideline_structure(guideline: Dict) -> bool:
    """
    Validate guideline has all required sections.
    
    Required Sections:
        - GUIDELINE (title)
        - CATEGORY
        - DEFINITION
        - DIAGNOSIS
        - TREATMENT
        - MANAGEMENT
    
    Returns:
        True if valid, False otherwise
    """
    required_sections = ["GUIDELINE", "CATEGORY", "DEFINITION", 
                        "DIAGNOSIS", "TREATMENT", "MANAGEMENT"]
    
    for section in required_sections:
        if section not in guideline:
            print(f"Missing section: {section}")
            return False
        
        if len(guideline[section]) < 50:  # Minimum content length
            print(f"Section too short: {section}")
            return False
    
    return True
```

---

### LLM Structuring Prompt

**Template:**
```python
STRUCTURING_PROMPT = """
You are a medical expert tasked with structuring clinical guideline content.

Topic: {topic}
Category: {category}

Raw Content:
{raw_content}

Instructions:
1. Extract information relevant to {topic}
2. Organize into the following sections:
   - DEFINITION: Comprehensive definition, epidemiology, pathophysiology
   - DIAGNOSIS: Diagnostic criteria, clinical presentation, tests
   - TREATMENT: Immediate management, medications, procedures
   - MANAGEMENT: Long-term care, monitoring, patient education

3. Use evidence-based information only
4. Include specific protocols, dosages, and timelines
5. Cite guidelines where appropriate (e.g., AHA, ACC, ESC)

Output format:
GUIDELINE: {topic}
CATEGORY: {category}

DEFINITION:
[Your definition here]

DIAGNOSIS:
[Your diagnostic criteria here]

TREATMENT:
[Your treatment protocols here]

MANAGEMENT:
[Your management strategies here]
"""
```

**Example LLM Call:**
```python
def structure_guideline(self, topic: str, raw_content: str) -> str:
    """
    Use LLM to structure raw content into guideline format.
    
    Args:
        topic: Clinical topic
        raw_content: Raw text about the topic
        
    Returns:
        Structured guideline text
    """
    prompt = STRUCTURING_PROMPT.format(
        topic=topic,
        category=self.detect_category(topic),
        raw_content=raw_content
    )
    
    response = self.client.generate(
        model=self.model,
        prompt=prompt,
        options={
            "temperature": 0.0,  # Deterministic
            "num_predict": 2048   # Long output
        }
    )
    
    return response['response']
```

---

### Category Detection

**Method:**
```python
def detect_category(self, topic: str) -> str:
    """
    Detect medical category from topic.
    
    Categories:
        - Cardiovascular Emergencies
        - Respiratory Conditions
        - Infectious Diseases
        - Gastrointestinal Disorders
        - Endocrine Disorders
        - Neurological Emergencies
        - Renal/Urinary Conditions
        - Rheumatologic Disorders
        - Psychiatric Conditions
        - General Emergency Medicine
    
    Args:
        topic: Clinical topic name
        
    Returns:
        Detected category
    """
    category_keywords = {
        "Cardiovascular Emergencies": [
            "coronary", "heart", "cardiac", "stroke", "hypertension",
            "atrial", "thrombosis", "embolism"
        ],
        "Respiratory Conditions": [
            "asthma", "copd", "pneumonia", "respiratory", "pulmonary"
        ],
        "Infectious Diseases": [
            "sepsis", "infection", "uti", "pneumonia"
        ],
        # ... more categories
    }
    
    topic_lower = topic.lower()
    for category, keywords in category_keywords.items():
        if any(keyword in topic_lower for keyword in keywords):
            return category
    
    return "General Emergency Medicine"
```

---

### Post-Processing

**Formatting:**
```python
def post_process_guideline(self, guideline_text: str) -> str:
    """
    Clean and format generated guideline.
    
    Steps:
        1. Remove LLM artifacts (e.g., "Here is the guideline:")
        2. Ensure section headers are properly formatted
        3. Remove excessive whitespace
        4. Validate section order
        5. Add blank lines between sections
    """
    # Remove LLM artifacts
    guideline_text = re.sub(r'^(Here is|Below is).*?\n', '', guideline_text)
    
    # Ensure section headers end with colon
    guideline_text = re.sub(r'^(DEFINITION|DIAGNOSIS|TREATMENT|MANAGEMENT)$', 
                            r'\1:', guideline_text, flags=re.MULTILINE)
    
    # Add blank lines between sections
    guideline_text = re.sub(r'(DEFINITION:|DIAGNOSIS:|TREATMENT:|MANAGEMENT:)', 
                            r'\n\1', guideline_text)
    
    return guideline_text.strip()
```

---

## Question Generator

**File:** `src/data_creation/question_generator.py`

### Purpose

Generates realistic clinical multiple-choice questions from structured guidelines with quality controls and balanced answer distribution.

### Key Classes

#### `QuestionGenerator`

**Constructor:**
```python
class QuestionGenerator:
    def __init__(
        self,
        ollama_model: str = "llama3.1:8b",
        guidelines_dir: str = "data/guidelines"
    ):
        """
        Initialize question generator.
        
        Args:
            ollama_model: Model for question generation
            guidelines_dir: Directory containing guideline files
        """
        self.model = ollama_model
        self.guidelines_dir = guidelines_dir
        self.guidelines = self.load_guidelines()
```

**Methods:**

##### `generate_questions(num_questions: int, seed: int = 42) -> List[Dict]`

Generates clinical MCQ cases.

```python
def generate_questions(
    self,
    num_questions: int,
    seed: int = 42
) -> List[Dict]:
    """
    Generate clinical MCQ cases from guidelines.
    
    Args:
        num_questions: Number of questions to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
        
    Process:
        1. Select guidelines randomly
        2. Generate case description with realistic vitals
        3. Generate question and 4 options
        4. Apply quality checks
        5. Ensure answer distribution balance
        6. Add metadata
    """
```

**Example Usage:**
```python
from src.data_creation.question_generator import QuestionGenerator

generator = QuestionGenerator()
questions = generator.generate_questions(num_questions=100, seed=42)

# Save to JSON
import json
with open("data/processed/questions/questions_1.json", "w") as f:
    json.dump({"questions": questions}, f, indent=2)
```

---

### Case Generation Pipeline

**Step 1: Guideline Selection**
```python
def select_guideline(self, rng: np.random.Generator) -> Dict:
    """
    Select random guideline for question generation.
    
    Strategy:
        - Weighted selection (prioritize underrepresented topics)
        - Track usage to ensure balanced coverage
    """
    # Weighted selection
    weights = [1 / (usage + 1) for usage in self.guideline_usage.values()]
    guideline = rng.choice(self.guidelines, p=weights/sum(weights))
    
    return guideline
```

**Step 2: Vital Sign Generation**
```python
def generate_realistic_vitals(
    self,
    age: int,
    condition: str,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Generate realistic vital signs for patient.
    
    Args:
        age: Patient age
        condition: Medical condition
        rng: Random generator
        
    Returns:
        Dictionary of vital signs
        
    Vitals Generated:
        - BP (systolic/diastolic)
        - HR (heart rate)
        - RR (respiratory rate)
        - SpO2 (oxygen saturation)
        - Temperature
    
    Constraints:
        - Age-appropriate ranges
        - Condition-appropriate abnormalities
        - Physiologically plausible combinations
    """
    vitals = {}
    
    # Blood Pressure
    if "hypertension" in condition.lower():
        vitals["BP_systolic"] = rng.integers(160, 190)
        vitals["BP_diastolic"] = rng.integers(95, 115)
    elif "shock" in condition.lower():
        vitals["BP_systolic"] = rng.integers(70, 90)
        vitals["BP_diastolic"] = rng.integers(40, 60)
    else:
        # Normal range with age adjustment
        vitals["BP_systolic"] = rng.integers(110, 140) + (age - 50) // 10
        vitals["BP_diastolic"] = rng.integers(70, 90)
    
    # Heart Rate
    if "tachycardia" in condition.lower():
        vitals["HR"] = rng.integers(100, 150)
    elif "bradycardia" in condition.lower():
        vitals["HR"] = rng.integers(40, 60)
    else:
        vitals["HR"] = rng.integers(60, 100)
    
    # Respiratory Rate
    if "respiratory" in condition.lower() or "pneumonia" in condition.lower():
        vitals["RR"] = rng.integers(22, 35)
    else:
        vitals["RR"] = rng.integers(12, 20)
    
    # SpO2
    if "hypoxia" in condition.lower():
        vitals["SpO2"] = rng.integers(85, 93)
    else:
        vitals["SpO2"] = rng.integers(95, 100)
    
    # Temperature
    if "fever" in condition.lower() or "infection" in condition.lower():
        vitals["temperature"] = rng.uniform(100.5, 103.5)
    else:
        vitals["temperature"] = rng.uniform(97.5, 99.0)
    
    return vitals
```

**Step 3: LLM Question Generation**
```python
QUESTION_GENERATION_PROMPT = """
You are a medical educator creating clinical MCQ cases.

Guideline:
{guideline_content}

Patient Demographics:
- Age: {age}
- Gender: {gender}

Vital Signs:
- BP: {bp}
- HR: {hr} bpm
- RR: {rr}/min
- SpO2: {spo2}%
- Temp: {temp}°F

Task:
Create a realistic clinical case with:
1. Case Description (2-3 sentences):
   - Chief complaint
   - Duration of symptoms
   - Relevant history
   - Physical exam findings

2. Question (one of these types):
   - Diagnosis: "What is the most likely diagnosis?"
   - Treatment: "What is the most appropriate treatment?"
   - Management: "What is the best next step in management?"
   - Immediate Action: "What is the most appropriate immediate action?"

3. Options (A/B/C/D):
   - One correct answer (based on guideline)
   - Three plausible distractors
   - All options should be properly formatted medical interventions

4. Explanation: Brief rationale for correct answer

Requirements:
- Base case on provided guideline content
- Make vitals consistent with condition
- Include relevant risk factors
- All options should be medically plausible
- Distractors should be common mistakes

Output JSON format:
{
  "case_description": "...",
  "question": "...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "correct_answer": "A|B|C|D",
  "explanation": "..."
}
"""
```

---

### Quality Controls

#### Quality Check 1: Fever Consistency

```python
def check_fever_consistency(self, case: Dict, vitals: Dict) -> bool:
    """
    If patient has fever (>101°F), ensure infectious differential considered.
    
    Args:
        case: Generated case
        vitals: Patient vital signs
        
    Returns:
        True if consistent, False if needs regeneration
    """
    if vitals.get("temperature", 98.0) > 101.0:
        options_text = " ".join(case["options"].values()).lower()
        
        infectious_terms = [
            "antibiotic", "infection", "sepsis", "culture",
            "infectious", "bacterial", "viral"
        ]
        
        if not any(term in options_text for term in infectious_terms):
            print(f"Warning: Fever present but no infectious options")
            return False
    
    return True
```

#### Quality Check 2: Stroke Protocol

```python
def check_stroke_protocol(self, case: Dict, condition: str) -> bool:
    """
    If stroke case, ensure imaging before anticoagulation.
    
    Args:
        case: Generated case
        condition: Medical condition
        
    Returns:
        True if protocol followed, False otherwise
    """
    if "stroke" in condition.lower():
        correct_option = case["options"][case["correct_answer"]].lower()
        
        # Check if anticoagulation is given before imaging
        if "anticoagul" in correct_option or "heparin" in correct_option:
            if "ct" not in case["case_description"].lower() and \
               "mri" not in case["case_description"].lower():
                print(f"Warning: Anticoagulation before imaging in stroke")
                return False
    
    return True
```

#### Quality Check 3: Clean Options

```python
def check_clean_options(self, case: Dict) -> bool:
    """
    Ensure options don't have embedded explanations.
    
    Bad: "A) Aspirin (prevents platelet aggregation)"
    Good: "A) Aspirin"
    
    Args:
        case: Generated case
        
    Returns:
        True if clean, False if needs cleaning
    """
    for option_key, option_text in case["options"].items():
        # Check for parenthetical explanations
        if "(" in option_text and ")" in option_text:
            # Remove explanation
            clean_option = re.sub(r'\s*\([^)]*\)', '', option_text)
            case["options"][option_key] = clean_option
            print(f"Cleaned option {option_key}: {clean_option}")
    
    return True
```

---

### Answer Distribution Balancing

**Cryptographic Shuffling:**
```python
def balance_answer_distribution(
    self,
    questions: List[Dict],
    target_distribution: Dict[str, float] = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
) -> List[Dict]:
    """
    Ensure balanced answer distribution using cryptographic shuffling.
    
    Args:
        questions: List of questions
        target_distribution: Target % for each answer
        
    Returns:
        Questions with balanced answers
        
    Strategy:
        1. For each question, compute hash(question_id)
        2. Use hash to select target answer (A/B/C/D)
        3. Shuffle options to place correct answer at target position
        4. Track overall distribution
        5. If imbalanced, reshuffle recent questions
    """
    import hashlib
    
    for question in questions:
        # Cryptographic hash for reproducibility
        hash_val = int(hashlib.sha256(question["id"].encode()).hexdigest(), 16)
        target_answer = ["A", "B", "C", "D"][hash_val % 4]
        
        # Shuffle options to place correct answer at target
        current_correct = question["correct_answer"]
        if current_correct != target_answer:
            question["options"] = self.shuffle_options(
                question["options"],
                current_correct,
                target_answer
            )
            question["correct_answer"] = target_answer
    
    # Verify distribution
    answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for question in questions:
        answer_counts[question["correct_answer"]] += 1
    
    print(f"Answer Distribution: {answer_counts}")
    
    return questions
```

---

## Complete Pipeline

**Script:** `scripts/generate_from_pdf.py`

**Usage:**
```bash
# Full pipeline (PDF → Guidelines → Questions)
python scripts/generate_from_pdf.py

# Generate only guidelines
python scripts/generate_from_pdf.py --guidelines-only

# Generate only questions (requires existing guidelines)
python scripts/generate_from_pdf.py --questions-only

# Custom PDF
python scripts/generate_from_pdf.py path/to/custom.pdf
```

**Pipeline Code:**
```python
from src.data_creation.pdf_extractor import PDFExtractor
from src.data_creation.guideline_generator import GuidelineGenerator
from src.data_creation.question_generator import QuestionGenerator

# Step 1: Extract PDF
extractor = PDFExtractor("data/standard-treatment-guidelines.pdf")
raw_text = extractor.extract_text()
print(f"Extracted {len(raw_text)} characters")

# Step 2: Generate Guidelines
generator = GuidelineGenerator()
topics = [
    "Acute Coronary Syndrome",
    "Stroke Management",
    "Diabetes Management",
    # ... 17 more topics
]
guidelines = generator.generate_from_text(raw_text, topics)
print(f"Generated {len(guidelines)} guidelines")

# Step 3: Generate Questions
question_gen = QuestionGenerator()
questions = question_gen.generate_questions(num_questions=100, seed=42)
print(f"Generated {len(questions)} questions")

# Step 4: Save
import json
with open("data/processed/questions/questions_1.json", "w") as f:
    json.dump({
        "questions": questions,
        "metadata": {
            "total_questions": len(questions),
            "generation_date": "2025-12-11",
            "generator_version": "v5.0"
        }
    }, f, indent=2)
```

---

## Related Documentation

- [Data Documentation](data_documentation.md)
- [Part 1: Dataset Creation](part_1_dataset_creation.md)

---

**Documentation Author:** Shreya Uprety
