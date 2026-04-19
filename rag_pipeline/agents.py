import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from tavily import TavilyClient
import arxiv
from Bio import Entrez

from rag_pipeline.retrieval import retrieve_and_answer, answer_from_docs

Entrez.email = os.getenv("NCBI_EMAIL", "test@example.com")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
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
        "You are a research agent. You MUST follow this strict tool order:\n"
        "1. ALWAYS call search_arxiv first for any research or science question.\n"
        "2. If search_arxiv returns no useful results, call search_pubmed next "
        "(especially for biomedical/clinical topics).\n"
        "3. Only call search_web if BOTH search_arxiv and search_pubmed failed "
        "or returned irrelevant results.\n"
        "Never skip directly to search_web. Cite sources in your answer."
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
        answer_text = agent_result["messages"][-1].content
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
        return {
            "answer":      f"Could not find an answer. Error: {str(e)}",
            "sources":     [],
            "source_type": "error",
        }