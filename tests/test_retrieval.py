from rag_pipeline.retrieval import retrieve_and_answer
import time

BM25_PATH = r"bm25_index\sample_paper_bm25.pkl"

question = [
    "What is the main contribution of this paper?",
    "What dataset was used for evaluation?",
    "Who are BTS?",
]

for q in question:
    print(f"\nQ: {q}")
    result = retrieve_and_answer(q, [BM25_PATH])
    print(f"Source : {result['source_type']}")
    print(f"Answer : {result['answer'][:300]}")
    print(f"Sources: {result['sources']}")
    time.sleep(4)
