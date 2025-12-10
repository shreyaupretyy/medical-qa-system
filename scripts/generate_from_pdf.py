"""
Generate Guidelines and Questions from PDF
===========================================

This script uses the data_creation module to:
1. Extract content from standard-treatment-guidelines.pdf
2. Generate structured guideline documents
3. Generate clinical questions from guidelines

Usage:
    python scripts/generate_from_pdf.py [pdf_path] [--guidelines-only] [--questions-only]

Examples:
    # Generate both guidelines and questions from default PDF
    python scripts/generate_from_pdf.py
    
    # Generate from specific PDF
    python scripts/generate_from_pdf.py data/custom-guidelines.pdf
    
    # Generate only guidelines
    python scripts/generate_from_pdf.py --guidelines-only
    
    # Generate only questions
    python scripts/generate_from_pdf.py --questions-only
"""

import sys
import argparse
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from data_creation.pdf_extractor import PDFExtractor
from data_creation.guideline_generator import GuidelineGenerator
from data_creation.question_generator import QuestionGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate guidelines and questions from medical PDF"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=None,
        help="Path to medical guidelines PDF (default: data/standard-treatment-guidelines.pdf)"
    )
    parser.add_argument(
        "--guidelines-only",
        action="store_true",
        help="Only generate guidelines, skip questions"
    )
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help="Only generate questions from existing guidelines"
    )
    parser.add_argument(
        "--guidelines-dir",
        default="data/guidelines",
        help="Directory for guideline output (default: data/guidelines)"
    )
    parser.add_argument(
        "--questions-dir",
        default="data/processed/questions",
        help="Directory for questions output (default: data/processed/questions)"
    )
    
    args = parser.parse_args()
    
    # Determine PDF path
    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
    else:
        pdf_path = REPO_ROOT / "data" / "standard-treatment-guidelines.pdf"
    
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        print("\nUsage: python scripts/generate_from_pdf.py [pdf_path]")
        sys.exit(1)
    
    print("=" * 80)
    print("MEDICAL GUIDELINES AND QUESTIONS GENERATOR")
    print("=" * 80)
    print(f"PDF: {pdf_path}")
    print(f"Guidelines output: {args.guidelines_dir}")
    print(f"Questions output: {args.questions_dir}")
    print("=" * 80)
    
    # Step 1: Generate Guidelines (if not questions-only)
    if not args.questions_only:
        print("\n[1/2] Generating Guidelines from PDF...")
        print("-" * 80)
        
        # Extract PDF content
        print("  Extracting PDF content...")
        extractor = PDFExtractor()
        pdf_content = extractor.extract(str(pdf_path))
        print(f"  ✓ Extracted {len(pdf_content.get('raw_text', ''))} characters")
        
        # Generate guidelines
        print("  Generating structured guidelines...")
        generator = GuidelineGenerator(output_dir=args.guidelines_dir)
        guidelines = generator.generate_from_pdf_content(pdf_content)
        print(f"  ✓ Generated {len(guidelines)} guidelines")
        
        for guideline in guidelines:
            print(f"    - {guideline['name']}")
        
        if args.guidelines_only:
            print("\n" + "=" * 80)
            print("GUIDELINES GENERATION COMPLETE")
            print("=" * 80)
            return
    
    # Step 2: Generate Questions (if not guidelines-only)
    if not args.guidelines_only:
        print("\n[2/2] Generating Clinical Questions...")
        print("-" * 80)
        
        # Load guidelines from directory
        guidelines_path = Path(args.guidelines_dir)
        if not guidelines_path.exists():
            print(f"[ERROR] Guidelines directory not found: {guidelines_path}")
            print("Run with --guidelines-only first to generate guidelines")
            sys.exit(1)
        
        guideline_files = list(guidelines_path.glob("guideline_*.txt"))
        if not guideline_files:
            print(f"[ERROR] No guideline files found in {guidelines_path}")
            print("Run with --guidelines-only first to generate guidelines")
            sys.exit(1)
        
        print(f"  Found {len(guideline_files)} guideline files")
        
        # Generate questions
        print("  Generating clinical questions...")
        question_gen = QuestionGenerator(output_dir=args.questions_dir)
        questions = question_gen.generate_from_guidelines(str(guidelines_path))
        
        print(f"  ✓ Generated {len(questions)} clinical questions")
        print(f"  ✓ Saved to: {args.questions_dir}")
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    
    if not args.questions_only:
        print(f"Guidelines: {args.guidelines_dir} ({len(guidelines)} files)")
    if not args.guidelines_only:
        print(f"Questions: {args.questions_dir} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
