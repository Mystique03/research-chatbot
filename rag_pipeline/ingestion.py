import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import fitz
#from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from tqdm import tqdm

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 512
CHUCK_OVERLAP = 64
BM25_DIR = Path("bm25_index")
BM25_DIR.mkdir(exist_ok=True)

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder

#pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "research-papers")

def create_index():
    """Create Pinecone index"""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

# Load and process PDF files
def parse_pdf(pdf_path):
    """Parse PDF and return text chunks"""
    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1]/50), b[0]))
        text = " ".join(b[4].strip() for b in blocks_sorted if b[4].strip())
        if text:
            pages.append({"text": text, "page_number": page_number + 1})
    doc.close()
    return pages

def chunk_pages(pages, paper_id):
    """split each page into overlapping chunks. Using metadat for citation"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap= CHUCK_OVERLAP,
        separators=["\n\n", "\n", ". ", "", " "]
    )
    chunks = []
    for page in pages:
        for doc in splitter.create_documents([page["text"]]):
            chunks.append({
                "text": doc.page_content,
                "metadata": {
                    "paper_id": paper_id,
                    "page": page["page_number"],
                    "chunk_index": len(chunks)
                }
            })
    return chunks

def embed_and_store(chunks, paper_id, index):
    """
    Embeds all chunks, upserts into pinecone, and builds BM25 index.
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(chunks)} chunks....")
    vectors = list(get_embedder().embed(texts))


    # Upsert into pinecode
    records = [
        {
            "id": f"{paper_id}_chunk_{i}",
            "values": vec.tolist(),
            "metadata": {**chunk["metadata"], "text": chunk["text"]},
        }
        for i , (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    for i in tqdm(range(0, len(records), 100), desc="Uploading to pincone"):
        index.upsert(vectors=records[i:i+100])

    # Build BM25 index
    bm25_path = BM25_DIR / f"{paper_id}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": BM25Okapi([t.lower().split() for t in texts]),
            "texts": texts,
            "chunks": chunks,
        }, f)

    return bm25_path

# Main ingestion function

def ingest(pdf_path, paper_id):
    """
    Full pipeline: pdf -> text extraction -> chunking -> embedding -> storage
    """

    print(f"Processing {paper_id}...")
    index = create_index()
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages, paper_id)
    bm25_path = embed_and_store(chunks, paper_id, index)

    raw_text = " ".join(page["text"] for page in pages)

    print(f"Ingestion complete for {paper_id}. {len(chunks)} chunks stored.")
    return {
        "paper_id": paper_id,
        "pages": len(pages),
        "chunks": len(chunks),
        "raw_text": raw_text,
        "bm25_path": str(bm25_path)
    }
    


    
    



