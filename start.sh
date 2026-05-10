#!/bin/bash
set -e

uvicorn api.main:app --host 0.0.0.0 --port 8000 &

streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false
