import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.agents import orchestrate

BM25_PATH = r"bm25_index\sample_paper_bm25.pkl"

questions = [
    "What is the main contribution of this paper?", 
    "What is LoRA in LLMs?",              
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    result = orchestrate(q, [BM25_PATH])
    print(f"Source : {result['source_type']}")
    print(f"Answer : {result['answer'][:400]}")
    print(f"Sources: {result['sources']}")
    time.sleep(4)