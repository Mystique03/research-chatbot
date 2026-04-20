import os
import re
from dotenv import load_dotenv
load_dotenv()

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from tavily import TavilyClient
import arxiv
from Bio import Entrez

from rag_pipeline.retrieval import retrieve_and_answer, answer_from_docs, answer_from_web

Entrez.email = os.getenv("NCBI_EMAIL", "test@example.com")

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=3
)

# Tools

@tool
def search_arxiv(query: str):
    """Search ArXiv for research papers on ML, AI, and CS topics."""
    try: 
        results = list(
            arxiv.Search(query=query, max_results=3,
                        sort_by=arxiv.SortCriterion.Relevance).results()
        )
        if not results:
            return "No relevant papers found on arXiv."
        return "\n\n".join(
            f"Title: {r.title}\n"
            f"Authors: {', '.join(str(a) for a in r.authors[:3])}\n"
            f"Summary: {r.summary[:300]}...\n"
            f"URL: {r.entry_id}"
            for r in results
        )
    except Exception as e:
        return f"ArXiv search failed: {str(e)}"

@tool
def search_pubmed(query: str):
    """Search PubMed for biomedical and life science research papers."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
        ids    = Entrez.read(handle)["IdList"]
        if not ids:
            return "No results found on PubMed."
        handle  = Entrez.efetch(db="pubmed", id=",".join(ids),
                                rettype="abstract", retmode="text")
        return handle.read()[:1500]
    except Exception as e:
        return f"PubMed search failed: {str(e)}"

@tool
def search_web(query: str):
    """Search the web for general information. Use as last resort."""
    try:
        client  = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query, max_results=3)

        if not results or "results" not in results:
            return "No web results found."
        
        return "\n\n".join(
            f"[{r['url']}]\n{r['content']}" 
            for r in results["results"]
            if r.get("url") and r.get("content")
        )
    except Exception as e:
        return f"Web search failed: {str(e)}"

# Agent 2
researcher_tools = [search_arxiv, search_pubmed, search_web]

researcher = create_react_agent(
    model=llm,
    tools=researcher_tools,
    prompt=(
        "You are a research assistant with access to search tools.\n"
        "For general knowledge questions (people, places, events, pop culture), "
        "use search_web directly.\n"
        "For scientific or academic questions, call search_arxiv first, then "
        "search_pubmed if needed, and search_web as a last resort.\n"
        "Always use exactly one tool call at a time with a plain string query. "
        "Cite sources in your answer."
    )
)

# Agent 1
def orchestrate(question: str, bm25_paths: list[str]) -> dict:
    result = retrieve_and_answer(question, bm25_paths)

    if result["source_type"] == "paper":
        return result

    # "web" or "not_found" — delegate to researcher

    print("\n  Orchestrator → delegating to Researcher agent...")
    try:
        agent_result = researcher.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        answer_text = _strip_thinking(agent_result["messages"][-1].content)
        external_context = [{
            "text":     answer_text,
            "metadata": {"source": "ArXiv / PubMed / Web", "page": "N/A"},
        }]
        answer = answer_from_docs(question, external_context)
        return {
            "answer":      answer,
            "sources":     ["ArXiv / PubMed / Web search"],
            "source_type": "external",
        }
    except Exception as e:
        print(f"  Researcher agent failed ({e}), falling back to Tavily...")
        try:
            return answer_from_web(question)
        except Exception as web_e:
            return {
                "answer":      f"Could not find an answer: {str(web_e)}",
                "sources":     [],
                "source_type": "error",
            }