import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import re
import numpy as np
from fastembed import TextEmbedding
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from tavily import TavilyClient

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Congif
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-papers")
TOP_K = 5
RELEVANCE_THRESHOLD = 0.55

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=3
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def hybrid_search(query, bm25_paths):
    """Perform hybrid search using BM25 and Pinecone vector search"""
    index = pc.Index(INDEX_NAME)

    query_vec = list(get_embedder().embed([query]))[0].tolist()
    dense_matches = index.query(
        vector = query_vec, top_k=TOP_K*2, include_metadata=True
    )["matches"]

    # lookup table: id -> result dict
    id_to_doc = {
        m["id"]: {"id": m["id"], "score": m["score"], "text": m["metadata"]["text"],
                  "metadata": m["metadata"]} 
        for m in dense_matches
    }

    bm25_ranked = []
    for path in bm25_paths:
        with open (path, "rb") as f:
            data = pickle.load(f)

        scores = data["bm25"].get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:TOP_K]

        for idx in top_idx:
            chunk = data["chunks"][idx]
            doc_id = f"{chunk['metadata']['paper_id']}_chunk_{idx}"
            bm25_ranked.append((doc_id, float(scores[idx])))

            if doc_id not in id_to_doc:
                id_to_doc[doc_id] = {
                    "id": doc_id, "score": float(scores[idx]), 
                    "text": data["texts"][idx],
                    "metadata": chunk["metadata"]
                }
    # RRF merge
    rrf_scores = {}
    for rank, match in enumerate(dense_matches):
        rrf_scores[match["id"]] = rrf_scores.get(match["id"], 0) + 1 / (rank + 61)
    for rank, (doc_id, _) in enumerate(bm25_ranked):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 61)

    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_K]
    top_docs = [id_to_doc[i] for i in top_ids if  i in id_to_doc]

    best_score = dense_matches[0]["score"] if dense_matches else 0.0
    return {"docs": top_docs, "best_score": best_score}

# LLM answering

QA_PROMPT = ChatPromptTemplate.from_template("""
You are a research assistant analyzing a paper. Answer the question using the context below.
Cite page numbers inline like (page 3). If the answer is partially in the context, give that partial answer.
                                              
Context:
{context}
                                              
Question: {question}
                                              
Answer:""")

qa_chain = QA_PROMPT | llm | StrOutputParser() | RunnableLambda(_strip_thinking)

def answer_from_docs(question, docs):
    context = "\n\n---\n\n".join(
        f"[Page {doc['metadata'].get('page', '?')}] {doc['text']}" 
        for doc in docs
    )
    return qa_chain.invoke({"context": context, "question": question})

# Web fallback

WEB_PROMPT = ChatPromptTemplate.from_template("""
You are a research assistant. Answer using the web search results below.
Always mention the source URLs.

Search results:
{context}

Question: {question}

Answer:""")

web_chain = WEB_PROMPT | llm | StrOutputParser() | RunnableLambda(_strip_thinking)

def answer_from_web(question):
    results = tavily.search(query=question, max_results=3)
    context = "\n\n---\n\n".join(
        f"[{r['url']}]\n{r['content']}"
        for r in results["results"]
    )
    return {
        "answer":      web_chain.invoke({"context": context, "question": question}),
        "sources":     [r["url"] for r in results["results"]],
        "source_type": "web",
    }

# Main pipeline
def retrieve_and_answer(question, bm25_paths):

    result = hybrid_search(question, bm25_paths)

    # check 1: score too low — signal caller to use researcher agent
    if result["best_score"] < RELEVANCE_THRESHOLD:
        print(f"  Low score ({result['best_score']:.2f}) → delegating to researcher agent")
        return {"source_type": "not_found", "answer": None, "sources": []}

    docs = result["docs"]
    answer = answer_from_docs(question, docs)

    # check 2: LLM couldn't answer from docs — delegate to researcher agent
    not_found_phrases = [
        "i don't know",
        "not in the context",
        "not mentioned",
        "does not contain",
        "no information",
        "cannot find",
        "not provided",
    ]

    if any(phrase in answer.lower() for phrase in not_found_phrases):
        print("LLM could not find answer in docs → delegating to researcher agent")
        return {"source_type": "not_found", "answer": None, "sources": []}
    
    sources = list({
        f"Page {d['metadata'].get('page', '?')} — {d['metadata'].get('paper_id', '')}"
        for d in result["docs"]
    })

    return {
        "answer":      answer,
        "sources":     sources,
        "source_type": "paper",
        "score":       round(result["best_score"], 3)
    }
    