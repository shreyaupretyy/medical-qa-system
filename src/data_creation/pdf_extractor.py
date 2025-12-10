"""
PDF Extractor Module
=====================

PURPOSE:
    Extract text content from medical PDF documents for processing.
    Handles various PDF formats and structures commonly found in medical guidelines.

TECHNICAL DETAILS:
    - Uses pdfplumber for text extraction (better for tables and structured content)
    - Falls back to pypdf for simpler PDFs
    - Preserves document structure and headings
    - Handles multi-column layouts

USAGE:
    extractor = PDFExtractor()
    content = extractor.extract("path/to/medical.pdf")
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts and structures text content from medical PDF documents.
    
    Attributes:
        extracted_sections: Dict containing extracted content organized by section
        raw_text: Complete raw text from the PDF
    
    Example:
        >>> extractor = PDFExtractor()
        >>> content = extractor.extract("guidelines.pdf")
        >>> print(content['sections'][0]['title'])
    """
    
    def __init__(self):
        self.extracted_sections = {}
        self.raw_text = ""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required libraries are available."""
        try:
            import pdfplumber
            self.use_pdfplumber = True
        except ImportError:
            self.use_pdfplumber = False
            logger.warning("pdfplumber not installed. Falling back to pypdf.")
        
        try:
            from pypdf import PdfReader
            self.use_pypdf = True
        except ImportError:
            self.use_pypdf = False
            logger.warning("pypdf not installed.")
    
    def extract(self, pdf_path: str) -> Dict:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - raw_text: Complete extracted text
                - sections: List of identified sections with titles and content
                - metadata: Document metadata
        
        Example:
            >>> content = extractor.extract("medical_guidelines.pdf")
            >>> print(len(content['sections']))
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting content from: {pdf_path}")
        
        # Extract raw text
        if self.use_pdfplumber:
            raw_text = self._extract_with_pdfplumber(pdf_path)
        elif self.use_pypdf:
            raw_text = self._extract_with_pypdf(pdf_path)
        else:
            raise ImportError("No PDF library available. Install pdfplumber or pypdf.")
        
        self.raw_text = raw_text
        
        # Parse sections
        sections = self._parse_sections(raw_text)
        
        # Extract metadata
        metadata = self._extract_metadata(pdf_path)
        
        result = {
            'raw_text': raw_text,
            'sections': sections,
            'metadata': metadata,
            'source_file': str(pdf_path)
        }
        
        self.extracted_sections = result
        logger.info(f"Extracted {len(sections)} sections from PDF")
        
        return result
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber library."""
        import pdfplumber
        
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"\n--- Page {i+1} ---\n")
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {i+1}: {e}")
        
        return "\n".join(text_parts)
    
    def _extract_with_pypdf(self, pdf_path: Path) -> str:
        """Extract text using pypdf library."""
        from pypdf import PdfReader
        
        text_parts = []
        reader = PdfReader(pdf_path)
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"\n--- Page {i+1} ---\n")
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting page {i+1}: {e}")
        
        return "\n".join(text_parts)
    
    def _parse_sections(self, text: str) -> List[Dict]:
        """
        Parse text into structured sections based on headings.
        
        Identifies section headers based on:
        - Numbered headings (1., 1.1, Chapter 1, etc.)
        - Capitalized headers
        - Common medical document structure patterns
        
        Args:
            text: Raw extracted text
            
        Returns:
            List of section dictionaries with 'title' and 'content' keys
        """
        sections = []
        
        # Patterns for section headers
        header_patterns = [
            r'^(\d+\.?\s+[A-Z][A-Za-z\s]+)',  # "1. Introduction" or "1 Introduction"
            r'^(Chapter\s+\d+[:\s]+.+)',  # "Chapter 1: ..."
            r'^([A-Z][A-Z\s]{5,})\s*$',  # ALL CAPS headers
            r'^(\d+\.\d+\.?\s+[A-Z][A-Za-z\s]+)',  # "1.1 Subsection"
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in header_patterns)
        
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': [], 'level': 0}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a header
            is_header = False
            for pattern in header_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # Save previous section if it has content
                    if current_section['content']:
                        current_section['content'] = '\n'.join(current_section['content'])
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'title': line,
                        'content': [],
                        'level': self._determine_level(line)
                    }
                    is_header = True
                    break
            
            if not is_header:
                current_section['content'].append(line)
        
        # Don't forget the last section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
        
        return sections
    
    def _determine_level(self, header: str) -> int:
        """Determine the hierarchical level of a header."""
        if re.match(r'^\d+\.\d+\.\d+', header):
            return 3
        elif re.match(r'^\d+\.\d+', header):
            return 2
        elif re.match(r'^\d+\.?\s', header):
            return 1
        else:
            return 0
    
    def _extract_metadata(self, pdf_path: Path) -> Dict:
        """Extract PDF metadata."""
        metadata = {
            'filename': pdf_path.name,
            'file_size': pdf_path.stat().st_size,
            'page_count': 0
        }
        
        try:
            if self.use_pypdf:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                metadata['page_count'] = len(reader.pages)
                if reader.metadata:
                    metadata['title'] = reader.metadata.get('/Title', '')
                    metadata['author'] = reader.metadata.get('/Author', '')
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def get_text_chunks(self, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split extracted text into overlapping chunks for embedding.
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with 'text', 'start', 'end' keys
        
        Example:
            >>> chunks = extractor.get_text_chunks(chunk_size=500)
            >>> print(f"Created {len(chunks)} chunks")
        """
        if not self.raw_text:
            raise ValueError("No text extracted. Call extract() first.")
        
        chunks = []
        text = self.raw_text
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in ['. ', '.\n', '? ', '!\n']:
                    last_boundary = text[start:end].rfind(boundary)
                    if last_boundary > chunk_size * 0.5:  # At least half chunk size
                        end = start + last_boundary + len(boundary)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'source': self.extracted_sections.get('source_file', '')
                })
                chunk_id += 1
            
            start = end - overlap if end < len(text) else len(text)
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks


def extract_medical_content(pdf_path: str) -> Dict:
    """
    Convenience function to extract medical content from a PDF.
    
    Args:
        pdf_path: Path to the medical PDF
        
    Returns:
        Dictionary with extracted content
    
    Example:
        >>> content = extract_medical_content("guidelines.pdf")
        >>> print(content['raw_text'][:500])
    """
    extractor = PDFExtractor()
    return extractor.extract(pdf_path)


if __name__ == "__main__":
    # Test the extractor
    import sys
    if len(sys.argv) > 1:
        content = extract_medical_content(sys.argv[1])
        print(f"Extracted {len(content['sections'])} sections")
        print(f"Total text length: {len(content['raw_text'])} characters")

