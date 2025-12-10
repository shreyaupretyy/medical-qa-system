"""
Document Processor for Medical Guidelines

This module handles:
1. Loading medical guidelines from JSON
2. Chunking documents into retrievable segments
3. Creating metadata for each chunk
4. Managing document overlap for context preservation

Key Concepts:
--------------
**Chunking Strategy**:
- Fixed-size chunks with overlap
- Default: 500 tokens per chunk, 100 token overlap
- Overlap ensures context isn't lost at chunk boundaries

**Why Chunking?**:
- Long documents (2000+ words) dilute relevance scores
- Smaller chunks = more precise retrieval
- Overlap maintains context across boundaries

Example:
--------
```python
processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
chunks = processor.load_and_chunk_guidelines("data/raw/medical_guidelines.json")
print(f"Created {len(chunks)} searchable chunks from {processor.num_guidelines} guidelines")
```
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class Document:
    """
    A searchable document chunk with metadata.
    
    Attributes:
        content: The text content of the chunk
        metadata: Dictionary containing:
            - guideline_id: Source guideline ID
            - title: Guideline title
            - category: Medical category
            - chunk_index: Position in original document
            - total_chunks: Total chunks from this guideline
            - keywords: Medical keywords
    """
    content: str
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        """String representation showing key info."""
        return (
            f"Document(guideline={self.metadata.get('guideline_id')}, "
            f"chunk={self.metadata.get('chunk_index')}/{self.metadata.get('total_chunks')}, "
            f"length={len(self.content)} chars)"
        )


class DocumentProcessor:
    """
    Processes medical guidelines into searchable chunks.
    
    This class implements a chunking strategy that:
    1. Splits long documents into smaller pieces
    2. Maintains overlap between chunks for context
    3. Preserves metadata from source guidelines
    
    Parameters:
        chunk_size: Target size of each chunk in characters (default: 500)
        chunk_overlap: Number of overlapping characters between chunks (default: 100)
        
    Note:
        chunk_size is approximate - we split at sentence boundaries to avoid
        cutting sentences in half, which would harm comprehension.
    """
    
    def __init__(
        self,
        chunk_size: int = 450,
        chunk_overlap: int = 120
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Approximate characters per chunk (splits at sentence boundaries)
            chunk_overlap: Characters of overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_guidelines = 0
        self.num_chunks = 0
        
    def load_and_chunk_guidelines(self, filepath: str) -> List[Document]:
        """
        Load medical guidelines and split into chunks.
        
        Supports two input formats:
        1) Legacy JSON file (e.g., data/raw/medical_guidelines.json)
        2) Directory of guideline_*.txt files (preserves full text/topics)
        
        Process:
        - Load guidelines (JSON or txt directory)
        - Split content into sentences
        - Group sentences into chunks of ~chunk_size with overlap
        - Create Document objects with metadata
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Guidelines file or directory not found: {path}")
        
        if path.is_dir():
            guidelines = self._load_guidelines_from_dir(path)
        else:
            guidelines = self._load_guidelines_from_json(path)
        
        self.num_guidelines = len(guidelines)
        
        # Process each guideline into chunks
        all_chunks = []
        for guideline in guidelines:
            chunks = self._chunk_guideline(guideline)
            all_chunks.extend(chunks)
        
        self.num_chunks = len(all_chunks)
        self._all_chunks = all_chunks  # Store for statistics calculation
        
        return all_chunks
    
    def _chunk_guideline(self, guideline: Dict[str, Any]) -> List[Document]:
        """
        Split a single guideline into overlapping chunks.
        
        Strategy:
        - Split content into sentences using regex
        - Accumulate sentences until we reach chunk_size
        - Include overlap from previous chunk for context
        - Preserve all metadata in each chunk
        
        Args:
            guideline: Dictionary with guideline data
            
        Returns:
            List of Document objects for this guideline
        """
        # Construct content from available fields if 'content' doesn't exist
        if 'content' in guideline:
            content = guideline['content']
        else:
            # Build content from structured fields
            content_parts = []
            
            # Add title
            if guideline.get('title'):
                content_parts.append(f"Title: {guideline['title']}")
            
            # Add clinical indication
            if guideline.get('clinical_indication'):
                content_parts.append(f"Clinical Indication: {guideline['clinical_indication']}")
            
            # Add diagnostic criteria
            if guideline.get('diagnostic_criteria'):
                if isinstance(guideline['diagnostic_criteria'], str):
                    content_parts.append(f"Diagnostic Criteria: {guideline['diagnostic_criteria']}")
                elif isinstance(guideline['diagnostic_criteria'], list):
                    content_parts.append(f"Diagnostic Criteria: {' '.join(guideline['diagnostic_criteria'])}")
            
            # Add treatment protocol
            if guideline.get('treatment_protocol'):
                if isinstance(guideline['treatment_protocol'], list):
                    content_parts.append(f"Treatment Protocol: {' '.join(guideline['treatment_protocol'])}")
                elif isinstance(guideline['treatment_protocol'], str):
                    content_parts.append(f"Treatment Protocol: {guideline['treatment_protocol']}")
            
            # Add key medications
            if guideline.get('key_medications'):
                if isinstance(guideline['key_medications'], list):
                    content_parts.append(f"Key Medications: {' '.join(guideline['key_medications'])}")
                elif isinstance(guideline['key_medications'], str):
                    content_parts.append(f"Key Medications: {guideline['key_medications']}")
            
            # Add contraindications
            if guideline.get('contraindications'):
                if isinstance(guideline['contraindications'], list):
                    content_parts.append(f"Contraindications: {' '.join(guideline['contraindications'])}")
                elif isinstance(guideline['contraindications'], str):
                    content_parts.append(f"Contraindications: {guideline['contraindications']}")
            
            # Add special considerations
            if guideline.get('special_considerations'):
                content_parts.append(f"Special Considerations: {guideline['special_considerations']}")
            
            # Add original content if available
            if guideline.get('original_content'):
                content_parts.append(f"Original Content: {guideline['original_content']}")
            
            content = ' '.join(content_parts)
        
        # Pre-split on headings/bullets to keep sections tight
        segments = []
        for para in content.split('\n'):
            para = para.strip()
            if not para:
                continue
            # Break bullet-heavy lines into smaller pieces
            if len(para) > 400 and ('* ' in para or '- ' in para):
                parts = re.split(r'(?<=\.)\s+(?=[A-Z])', para)
                segments.extend([p.strip() for p in parts if p.strip()])
            else:
                segments.append(para)
        content_for_split = ' '.join(segments)

        # Split into sentences (handles common abbreviations)
        sentences = self._split_into_sentences(content_for_split)
        
        if not sentences:
            return []
        
        # ACS-specific tighter chunking (150-250 tokens ~ 320 chars)
        eff_chunk_size = self.chunk_size
        eff_overlap = self.chunk_overlap
        title_lower = guideline.get('title', '').lower()
        if 'cardiovascular emergencies' in title_lower or guideline.get('guideline_id') == 'GL_001':
            eff_chunk_size = 320
            eff_overlap = 80

        chunks = []
        current_chunk = []
        current_length = 0
        overlap_sentences = []  # Sentences to include in next chunk for overlap
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, create a chunk
            if current_length + sentence_length > eff_chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap: include last few sentences in next chunk
                overlap_length = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= eff_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences.copy()
                current_length = sum(len(s) for s in current_chunk)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        # Convert to Document objects with ENHANCED metadata (Fix 4)
        documents = []
        for i, chunk_text in enumerate(chunks):
            # FIX 4: Extract rich metadata for better retrieval
            diseases = self._extract_diseases_from_text(chunk_text)
            symptoms = self._extract_symptoms_from_text(chunk_text)
            treatments = self._extract_treatments_from_text(chunk_text)
            
            doc = Document(
                content=chunk_text,
                metadata={
                    'guideline_id': guideline['guideline_id'],
                    'title': guideline['title'],
                    'guideline_title': guideline['title'],  # Alias for compatibility
                    'category': guideline['category'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'keywords': guideline.get('keywords', []),
                    'source': guideline.get('source', 'Generated'),
                    # FIX 4: Rich metadata for retrieval
                    'diseases': diseases,
                    'symptoms': symptoms,
                    'treatments': treatments,
                    'clinical_indication': guideline.get('clinical_indication', ''),
                    # Section tags to bridge queries to guidelines
                    'section_tags': guideline.get('section_tags', []),
                    'symptom_tags': guideline.get('symptom_tags', []),
                    'has_treatment_protocol': 'treatment' in chunk_text.lower() or 'therapy' in chunk_text.lower(),
                    'has_diagnostic_info': 'diagnosis' in chunk_text.lower() or 'criteria' in chunk_text.lower(),
                }
            )
            documents.append(doc)
        
        return documents
    
    def _extract_diseases_from_text(self, text: str) -> List[str]:
        """Extract disease mentions from chunk text for metadata."""
        diseases = []
        disease_patterns = [
            r'\b(pneumonia|sepsis|meningitis|diabetes|hypertension|hypotension|'
            r'myocardial infarction|heart failure|stroke|pulmonary embolism|'
            r'preeclampsia|eclampsia|anemia|infection|shock|arrhythmia|'
            r'tuberculosis|malaria|typhoid|cholera|dengue|hepatitis|'
            r'asthma|copd|bronchitis|gastroenteritis|cellulitis|endocarditis|'
            r'neonatal sepsis|respiratory distress|jaundice|dehydration|'
            r'acute kidney injury|chronic kidney disease|urinary tract infection)\b'
        ]
        text_lower = text.lower()
        for pattern in disease_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            diseases.extend(matches)
        return list(set(diseases))[:10]
    
    def _extract_symptoms_from_text(self, text: str) -> List[str]:
        """Extract symptom mentions from chunk text for metadata."""
        symptoms = []
        symptom_patterns = [
            r'\b(fever|cough|dyspnea|pain|bleeding|vomiting|diarrhea|'
            r'headache|seizure|confusion|weakness|fatigue|rash|swelling|'
            r'tachycardia|bradycardia|hypoxia|cyanosis|jaundice|edema|'
            r'nausea|syncope|dizziness|chest pain|abdominal pain)\b'
        ]
        text_lower = text.lower()
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            symptoms.extend(matches)
        return list(set(symptoms))[:10]
    
    def _extract_treatments_from_text(self, text: str) -> List[str]:
        """Extract treatment mentions from chunk text for metadata."""
        treatments = []
        treatment_patterns = [
            r'\b(antibiotic|antiviral|analgesic|antipyretic|steroid|'
            r'insulin|oxygen|fluids|transfusion|surgery|intubation|'
            r'ceftriaxone|ampicillin|gentamicin|azithromycin|metronidazole|'
            r'paracetamol|ibuprofen|morphine|epinephrine|dopamine|'
            r'furosemide|magnesium sulfate|oxytocin|hydralazine)\b'
        ]
        text_lower = text.lower()
        for pattern in treatment_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            treatments.extend(matches)
        return list(set(treatments))[:10]

    def _load_guidelines_from_json(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load guidelines from a JSON file (legacy format)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_guidelines_from_dir(self, guidelines_dir: Path) -> List[Dict[str, Any]]:
        """
        Load guidelines directly from text files without losing key information or topics.
        
        Expected structure:
            guidelines_dir/
              guideline_01_*.txt   (first line is title, optional Category/Keywords lines)
              guidelines_summary.json (optional metadata for categories)
              glossary_*.txt (optional glossaries)
        """
        summary_path = guidelines_dir / "guidelines_summary.json"
        category_map: Dict[int, str] = {}
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                for g in summary.get('guidelines', []):
                    try:
                        category_map[int(g['id'])] = g.get('category', 'General')
                    except Exception:
                        continue
        
        guidelines: List[Dict[str, Any]] = []
        for path in sorted(guidelines_dir.glob("guideline_*.txt")):
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()
            title = lines[0].lstrip('# ').strip() if lines else path.stem
            
            category = None
            keywords: List[str] = []
            for line in lines[1:6]:
                lower = line.lower()
                if lower.startswith('category:'):
                    category = line.split(':', 1)[1].strip()
                if lower.startswith('keywords:'):
                    keywords = [k.strip() for k in line.split(':', 1)[1].split(',') if k.strip()]
            
            match = re.search(r'guideline_(\d+)_', path.name)
            gid_num = int(match.group(1)) if match else len(guidelines) + 1
            guideline_id = f"GL_{gid_num:03d}"
            
            # Derive section/symptom tags for query-to-guideline bridges
            content_lower = content.lower()
            section_tags = []
            if "red flag" in content_lower or "red flags" in content_lower:
                section_tags.append("red_flags")
            if "diagnostic" in content_lower or "diagnosis" in content_lower:
                section_tags.append("diagnosis")
            if "treatment" in content_lower or "management" in content_lower:
                section_tags.append("management")
            if "investigation" in content_lower or "ecg" in content_lower or "troponin" in content_lower:
                section_tags.append("investigations")
            if "risk stratification" in content_lower or "risk" in content_lower:
                section_tags.append("risk_stratification")

            symptom_tags = []
            for s in ["chest pain", "retrosternal", "dyspnea", "shortness of breath", "weakness", "hemiparesis"]:
                if s in content_lower:
                    symptom_tags.append(s)

            guidelines.append({
                'guideline_id': guideline_id,
                'title': title,
                'content': content,
                'category': category or category_map.get(gid_num, 'General'),
                'keywords': keywords,
                'source': 'GuidelineText',
                'section_tags': section_tags,
                'symptom_tags': symptom_tags
            })
        
        # Optional glossaries (treated as guidelines for retrieval)
        for gpath in sorted(guidelines_dir.glob("glossary_*.txt")):
            gcontent = gpath.read_text(encoding='utf-8')
            gtitle = gpath.stem.replace('_', ' ').title()
            guidelines.append({
                'guideline_id': f"GLOSS_{gpath.stem}",
                'title': gtitle,
                'content': gcontent,
                'category': 'Glossary',
                'keywords': ['glossary', 'definitions'],
                'source': 'Glossary',
                'section_tags': ['glossary'],
                'symptom_tags': []
            })

        return guidelines

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling medical abbreviations.
        
        Medical texts contain abbreviations like "Dr.", "vs.", "e.g." that
        shouldn't be treated as sentence boundaries.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Common medical abbreviations that shouldn't trigger sentence splits
        # We'll use a simple regex that splits on ". " followed by capital letter
        # This isn't perfect but works for most medical text
        
        # Replace common abbreviations temporarily
        text = text.replace('Dr.', 'Dr@')
        text = text.replace('vs.', 'vs@')
        text = text.replace('e.g.', 'eg@')
        text = text.replace('i.e.', 'ie@')
        text = text.replace('etc.', 'etc@')
        
        # Split on sentence boundaries: ". ", "! ", "? " followed by capital or end
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('@', '.') for s in sentences]
        
        # Clean up whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Returns:
            Dictionary with:
            - num_guidelines: Number of source guidelines
            - num_chunks: Total chunks created
            - avg_chunks_per_guideline: Average chunks per guideline
            - chunk_size: Configured chunk size
            - chunk_overlap: Configured overlap
            - avg_chunk_size: Average characters per chunk
            - total_chars: Total characters in all chunks
        """
        # Calculate average chunk size and total chars if we have chunks
        avg_chunk_size = 0
        total_chars = 0
        if hasattr(self, '_all_chunks') and self._all_chunks:
            total_chars = sum(len(chunk.content) for chunk in self._all_chunks)
            avg_chunk_size = total_chars / len(self._all_chunks) if self._all_chunks else 0
        
        return {
            'num_guidelines': self.num_guidelines,
            'num_chunks': self.num_chunks,
            'avg_chunks_per_guideline': self.num_chunks / self.num_guidelines if self.num_guidelines > 0 else 0,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'avg_chunk_size': avg_chunk_size,
            'total_chars': total_chars,
        }


def main():
    """
    Demo: Load and chunk medical guidelines.
    
    This shows how to use DocumentProcessor to prepare guidelines for retrieval.
    """
    print("="*60)
    print("DOCUMENT PROCESSOR DEMO")
    print("="*60)
    
    # Initialize processor with chunking parameters
    processor = DocumentProcessor(
        chunk_size=500,    # ~500 characters per chunk
        chunk_overlap=100  # 100 character overlap between chunks
    )
    
    # Load and chunk guidelines
    guidelines_path = Path(__file__).parent.parent.parent / "data" / "raw" / "medical_guidelines.json"
    
    if not guidelines_path.exists():
        print(f"\n❌ Error: Guidelines not found at {guidelines_path}")
        print("Please run data generation first: python scripts/generate_data.py")
        return
    
    print(f"\n📂 Loading guidelines from: {guidelines_path}")
    chunks = processor.load_and_chunk_guidelines(str(guidelines_path))
    
    # Display statistics
    stats = processor.get_statistics()
    print(f"\n📊 CHUNKING STATISTICS")
    print(f"{'='*60}")
    print(f"Source guidelines:           {stats['num_guidelines']}")
    print(f"Total chunks created:        {stats['num_chunks']}")
    print(f"Avg chunks per guideline:    {stats['avg_chunks_per_guideline']:.1f}")
    print(f"Chunk size:                  {stats['chunk_size']} chars")
    print(f"Chunk overlap:               {stats['chunk_overlap']} chars")
    
    # Show example chunks
    print(f"\n📄 EXAMPLE CHUNKS")
    print(f"{'='*60}")
    
    # Show first chunk from first guideline
    if chunks:
        doc = chunks[0]
        print(f"\nChunk 1 ({doc.metadata['category']} - {doc.metadata['title']})")
        print(f"Guideline ID: {doc.metadata['guideline_id']}")
        print(f"Chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']}")
        print(f"\nContent preview:")
        print(f"{doc.content[:300]}...")
        print(f"\nKeywords: {', '.join(doc.metadata['keywords'][:5])}")
    
    # Show chunk from middle
    if len(chunks) > 10:
        doc = chunks[len(chunks) // 2]
        print(f"\n{'='*60}")
        print(f"\nChunk {len(chunks)//2} ({doc.metadata['category']} - {doc.metadata['title']})")
        print(f"Guideline ID: {doc.metadata['guideline_id']}")
        print(f"Chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata['total_chunks']}")
        print(f"\nContent preview:")
        print(f"{doc.content[:300]}...")
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed {stats['num_guidelines']} guidelines into {stats['num_chunks']} searchable chunks")
    print(f"\nNext step: Create embeddings from these chunks using SentenceTransformers")


if __name__ == "__main__":
    main()
