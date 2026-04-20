import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import _faithfulness, _answer_relevancy, _context_entity_recall

# Groq only supports n=1; override RAGAS default of n=3
_answer_relevancy.strictness = 1
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings

from rag_pipeline.retrieval import hybrid_search, answer_from_docs

# Groq qwen3-32b pricing (per 1K tokens)
INPUT_COST_PER_1K  = 0.00000029
OUTPUT_COST_PER_1K = 0.00000059
CHARS_PER_TOKEN    = 4

ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    reasoning_effort="none",
))

ragas_embeddings = LangchainEmbeddingsWrapper(
    FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

@dataclass
class QueryMetrics:
    question:          str
    answer:            str
    contexts:          list[str]
    ground_truth:      str
    retrieval_latency: float   # seconds
    llm_latency:       float   # seconds
    total_latency:     float   # seconds
    input_tokens:      int
    output_tokens:     int
    cost_usd:          float

def estimate_token(text):
    return max(1, len(text) // CHARS_PER_TOKEN)

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
    return round(input_cost + output_cost, 8)

def collect_metrics(qa_pairs, bm25_paths):
    """
    Run each question through the pipeline and collect
    latency + cost metrics per query.
    """
    all_metrics = []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        print(f"\nEvaluating query {i+1}/{len(qa_pairs)}: {question}")

        #retrival
        t0 = time.perf_counter()
        result = hybrid_search(question, bm25_paths)
        t1 = time.perf_counter()
        retrieval_latency = round(t1 - t0, 3)

        context_texts = [d["text"] for d in result["docs"]]

        #llm answer
        t2 = time.perf_counter()
        answer = answer_from_docs(question, result["docs"])
        t3 = time.perf_counter()
        llm_latency   = round(t3 - t2, 3)
        total_latency = round(t3 - t0, 3)

        #cost
        input_text    = question + " ".join(context_texts)
        input_tokens  = estimate_token(input_text)
        output_tokens = estimate_token(answer)
        cost          = calculate_cost(input_tokens, output_tokens)

        all_metrics.append(QueryMetrics(
            question=question,
            answer=answer,
            contexts=context_texts,
            ground_truth=pair["ground_truth"],
            retrieval_latency=retrieval_latency,
            llm_latency=llm_latency,
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost
        ))

        time.sleep(8)

    return all_metrics

def run_ragas(metrics):
    dataset = Dataset.from_dict({
        "question":    [m.question     for m in metrics],
        "answer":      [m.answer       for m in metrics],
        "contexts":    [m.contexts     for m in metrics],
        "ground_truth":[m.ground_truth for m in metrics],
    })

    results = evaluate(
        dataset    = dataset,
        metrics    = [_faithfulness, _answer_relevancy, _context_entity_recall],
        llm        = ragas_llm,
        embeddings = ragas_embeddings,
        run_config = RunConfig(max_workers=1, timeout=120),
    )
    return results

def summerise_latency_and_cost(metrics):
    def avg(vals): return round(sum(vals) / len(vals), 3)
    def total(vals): return round(sum(vals), 6)

    return {
        "avg_retrieval_latency_sec": avg([m.retrieval_latency for m in metrics]),
        "avg_llm_latency_sec":       avg([m.llm_latency       for m in metrics]),
        "avg_total_latency_sec":     avg([m.total_latency     for m in metrics]),
        "avg_cost_per_query_usd":    avg([m.cost_usd          for m in metrics]),
        "total_cost_usd":            total([m.cost_usd        for m in metrics]),
        "total_queries":             len(metrics),
    }

# Example usage
def run_evaluation(qa_pairs, bm25_paths):
    print("\n-- Collecting metrics ---------------------------")
    metrics = collect_metrics(qa_pairs, bm25_paths)

    print("\n-- Running RAGAS evaluation ---------------------")
    ragas_scores = run_ragas(metrics)

    print("\n-- Summarising latency + cost -------------------")
    perf_scores = summerise_latency_and_cost(metrics)

    ragas_dict = ragas_scores.to_pandas().mean(numeric_only=True).to_dict()
    return {**ragas_dict, **perf_scores}

def save_results(results: dict, path: str = "evaluation_results.json"):
    """Save evaluation results to a JSON file with a timestamp."""
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results":   results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n  Results saved to {path}")