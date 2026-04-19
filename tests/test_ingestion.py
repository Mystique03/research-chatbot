from rag_pipeline.ingestion import ingest

result = ingest(
    pdf_path=r"data\attention_is_all_you_need.pdf",
    paper_id="sample_paper"
)

print("Pages   :", result["pages"])
print("Chunks  :", result["chunks"])
print("BM25 at :", result["bm25_path"])
print("Text preview:", result["raw_text"][:200])