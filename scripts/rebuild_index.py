"""
Rebuild FAISS index with new embedding model.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.embeddings import EmbeddingModel
from retrieval.faiss_store import FAISSVectorStore

def load_guidelines_from_txt_files(guidelines_dir: Path) -> list:
    """Load guidelines from text files."""
    guidelines = []
    
    # Category mapping based on filename
    category_map = {
        '01': 'Cardiovascular',
        '02': 'Neurology', 
        '03': 'Endocrine',
        '04': 'Cardiovascular',
        '05': 'Respiratory',
        '06': 'Respiratory',
        '07': 'Respiratory',
        '08': 'Infectious Disease',
        '09': 'Critical Care',
        '10': 'Gastroenterology',
        '11': 'Nephrology',
        '12': 'Cardiovascular',
        '13': 'Cardiovascular',
        '14': 'Hematology',
        '15': 'Respiratory',
        '16': 'Gastroenterology',
        '17': 'Gastroenterology',
        '18': 'Rheumatology',
        '19': 'Rheumatology',
        '20': 'Psychiatry'
    }
    
    for filepath in sorted(guidelines_dir.glob('guideline_*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        lines = content.split('\n')
        name = lines[0].replace('# ', '') if lines else ''
        
        # Extract ID from filename (e.g., guideline_01_cardiovascular.txt)
        import re
        match = re.search(r'guideline_(\d+)_', filepath.name)
        guideline_num = match.group(1) if match else f"{len(guidelines)+1:02d}"
        # Use 3-digit format to match evaluation expectations: GL_001, GL_002, etc.
        guideline_id = f"GL_{guideline_num.zfill(3)}"
        category = category_map.get(guideline_num, 'General')
        
        guidelines.append({
            'guideline_id': guideline_id,
            'title': name,
            'content': content,
            'category': category,
            'filepath': str(filepath)
        })
    
    return guidelines

def main():
    print("="*60)
    print("REBUILDING FAISS INDEX")
    print("="*60)
    
    # Initialize components
    print("\n[1/3] Initializing embedding model...")
    embedding_model = EmbeddingModel()
    
    print("\n[2/3] Building FAISS index from guidelines...")
    store = FAISSVectorStore(embedding_model)
    
    guidelines_dir = Path(__file__).parent.parent / "data" / "guidelines"
    if not guidelines_dir.exists():
        print(f"\n[ERROR] Guidelines directory not found at {guidelines_dir}")
        return
    
    # Load guidelines from text files
    guidelines = load_guidelines_from_txt_files(guidelines_dir)
    print(f"Loaded {len(guidelines)} guidelines")
    
    # Save to temporary JSON for build_index_from_guidelines
    temp_json = guidelines_dir / "temp_guidelines.json"
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(guidelines, f, indent=2)
    
    store.build_index_from_guidelines(str(temp_json))
    
    # Clean up temp file
    temp_json.unlink()
    
    print("\n[3/3] Saving index to disk...")
    index_dir = Path(__file__).parent.parent / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_index(str(index_dir))
    
    print("\n" + "="*60)
    print("INDEX REBUILD COMPLETE!")
    print("="*60)
    print(f"Total documents: {len(store.documents)}")
    print(f"Embedding dimension: {store.dimension}")
    print(f"Index saved to: {index_dir}")

if __name__ == "__main__":
    main()
