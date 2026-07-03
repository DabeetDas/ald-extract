import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the image to text module
# Use absolute or relative import depending on use context
try:
    from chunker.img_to_txt import describe_image
except ImportError:
    from img_to_txt import describe_image


# ========================================
# 1. Define Data Schemas using Pydantic
# ========================================

class Source(BaseModel):
    type: str # 'text' or 'image'
    path: str
    method: Optional[str] = None # e.g. 'caption'

class ChunkMetadata(BaseModel):
    document_id: str
    chunk_index: int
    source_file: str
    sources: List[Source]
    section: Optional[str] = None

class DocumentChunk(BaseModel):
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata


# ========================================
# 2. Logic to Parse Data and Make Schemas
# ========================================

def process_paper(paper_dir: str, paper_id: str) -> List[DocumentChunk]:
    """
    Processes a single paper directory (which contains content.txt and Images/)
    and outputs a list of built DocumentChunk schemas.
    """
    chunks: List[DocumentChunk] = []
    content_path = os.path.join(paper_dir, "content.txt")
    images_dir = os.path.join(paper_dir, "Images")
    
    # Check if text content exists
    if not os.path.exists(content_path):
        print(f"Warning: {content_path} not found.")
        return chunks
        
    try:
        with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    except Exception as e:
        print(f"Failed to read {content_path}: {e}")
        return chunks
        
    # Standard chunker for text data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    text_chunks = text_splitter.split_text(text_content)
    
    # --- Process Images ---
    # Extract descriptions and generate source metadata 
    image_descriptions = []
    image_sources = []
    
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        for img_file in sorted(os.listdir(images_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, img_file)
                desc = describe_image(img_path)
                
                if desc:
                    # Format exactly as requested in example schema
                    image_descriptions.append(f"[IMAGE DESCRIPTION]: {desc}")
                    
                    image_sources.append(Source(
                        type="image",
                        path=img_path,
                        method="caption"
                    ))

    # --- Construct DocumentChunk Schemas ---
    for i, t_chunk in enumerate(text_chunks):
        # We append image descriptions to chunks to preserve context.
        # Images are aggregated into the text sequence.
        chunk_text = t_chunk
        if image_descriptions:
            chunk_text += "\n\n" + "\n".join(image_descriptions)
            
        sources = [Source(type="text", path=content_path)]
        sources.extend(image_sources)
        
        metadata = ChunkMetadata(
            document_id=paper_id,
            chunk_index=i,
            source_file=content_path,
            sources=sources,
            section=f"chunk_{i}" # Simplified section mapping fallback
        )
        
        doc_chunk = DocumentChunk(
            id=f"{paper_id}_chunk_{i}",
            text=chunk_text,
            metadata=metadata
        )
        
        chunks.append(doc_chunk)
        
    return chunks

def extract_all_data(data_dir: str, output_dir: str, limit: Optional[int] = None) -> int:
    """
    Loops over all paper directories in Data, builds chunk schemas,
    and saves each paper's schemas to a separate JSON file in output_dir.
    Returns the total number of chunks generated across all papers.
    """
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    gibberish_papers = {
        'paper10', 'paper11', 'paper111', 'paper112', 'paper113', 'paper114',
        'paper115', 'paper116', 'paper117', 'paper118', 'paper119', 'paper120',
        'paper121', 'paper122', 'paper123', 'paper124', 'paper125', 'paper126',
        'paper130', 'paper132', 'paper133', 'paper15', 'paper1726', 'paper1729',
        'paper1730', 'paper2411', 'paper25', 'paper26', 'paper27', 'paper28',
        'paper29', 'paper3181', 'paper523', 'paper588', 'paper590', 'paper591',
        'paper644', 'paper645',
    }
        
    papers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Sort them (e.g. paper1, paper2, ... so limit feels deterministic)
    papers = sorted(papers)
    
    total_chunks = 0
    for idx, paper in enumerate(papers):
        if limit and idx >= limit:
            break

        if paper in gibberish_papers:
            print(f"Skipping {paper} (gibberish text)")
            continue
            
        paper_dir = os.path.join(data_dir, paper)
        print(f"Processing directory: {paper}...")
        paper_chunks = process_paper(paper_dir, paper)
        
        if paper_chunks:
            out_path = os.path.join(output_dir, f"{paper}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([json.loads(c.model_dump_json()) for c in paper_chunks], f, indent=2)
            print(f"  -> Saved {len(paper_chunks)} chunks to {out_path}")
            total_chunks += len(paper_chunks)
        
    return total_chunks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Document Chunk JSON Schemas")
    parser.add_argument("--limit", type=int, default=2, help="Limit number of papers to process")
    parser.add_argument("--output-dir", type=str, default="schemas_output", help="Directory to save per-paper JSON schemas")
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    out_dir = os.path.abspath(args.output_dir)
    
    total = extract_all_data(data_path, output_dir=out_dir, limit=args.limit)
    print(f"\nDone! Generated {total} total chunks across papers.")
    print(f"Per-paper JSON files saved in: {out_dir}")

