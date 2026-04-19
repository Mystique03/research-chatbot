import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline.ingestion import ingest
from rag_pipeline.chains import summarize
from rag_pipeline.agents import orchestrate

from dotenv import load_dotenv
load_dotenv()

# App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In memory store for ingested papers 
papers: dict = {}

class QueryRequest(BaseModel):
    question: str
    paper_ids: list[str] = []

class QueryResponse(BaseModel):
    answer: str
    source_type: str
    sources: list[str]

# Endpoints

@app.get("/")
def root():
    return {
        "status": "ok",
        "papers_loaded": list(papers.keys())
    }

@app.post("/ingest")
async def ingest_paper(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}
    
    paper_id = Path(file.filename).stem.replace(" ", "_")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingest(tmp_path, paper_id)
        papers[paper_id] = result
        return {
            "message":  f"'{paper_id}' ingested successfully.",
            "pages":    result["pages"],
            "chunks":   result["chunks"],
            "paper_id": paper_id,
        }
    finally:
        os.unlink(tmp_path)

@app.post("/summarize/{paper_id}")
def summarizr_paper(paper_id):
    if paper_id not in papers:
        raise HTTPException(
            status_code=404,
            detail=f"'{paper_id}' not found. Ingest it first."
        )
    return {"summary": summarize(papers[paper_id]["raw_text"])}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not papers:
        raise HTTPException(
            status_code=400,
            detail="No papers ingested yet. Upload a PDF first."
        )
    
    ids = req.paper_ids or list(papers.keys())
    bm25_paths = [
        papers[pid]["bm25_path"]
        for pid in ids
        if pid in papers
    ]

    result = orchestrate(req.question, bm25_paths)
    return QueryResponse(**result)

@app.get("/papers")
def list_papers():
    """List all ingested papers with their metadata."""
    return {
        pid: {"pages": info["pages"], "chunks": info["chunks"]}
        for pid, info in papers.items()
    }

