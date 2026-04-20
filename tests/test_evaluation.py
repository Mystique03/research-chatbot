import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.evaluation import run_evaluation, save_results

BM25_PATH = r"bm25_index/sample_paper_bm25.pkl"

qa_pairs = [
    {
        "question":    "What architecture does this paper propose?",
        "ground_truth": "The Transformer, a model based entirely on attention mechanisms.",
    },
    {
        "question":    "What dataset was used for English to German translation?",
        "ground_truth": "WMT 2014 English-German dataset.",
    },
    {
        "question":    "What optimizer was used for training?",
        "ground_truth": "Adam optimizer.",
    },
]

results = run_evaluation(qa_pairs, [BM25_PATH])


print("-----------------EVALUATION RESULTS-----------------")
print(f"\n  RAG Quality")
print(f"  ├─ Faithfulness      : {results['faithfulness']:.3f}")
print(f"  ├─ Answer Relevancy  : {results['answer_relevancy']:.3f}")
print(f"  └─ Context Relevancy : {results['context_entity_recall']:.3f}")
print(f"\n  Latency")
print(f"  ├─ Avg retrieval     : {results['avg_retrieval_latency_sec']}s")
print(f"  ├─ Avg LLM           : {results['avg_llm_latency_sec']}s")
print(f"  └─ Avg total         : {results['avg_total_latency_sec']}s")
print(f"\n  Cost")
print(f"  ├─ Avg per query     : ${results['avg_cost_per_query_usd']}")
print(f"  ├─ Total ({results['total_queries']} queries)  : ${results['total_cost_usd']}")
print(f"  └─ Free tier limit   : $0.00/day (Groq free tier)")

save_results(results, path="data/evaluation_results.json")