# tests/test_chains.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.ingestion import ingest
from rag_pipeline.chains import summarize

# reuse the paper you already ingested
result  = ingest(r"data\attention_is_all_you_need.pdf", "sample_paper")
summary = summarize(result["raw_text"])

print(summary)