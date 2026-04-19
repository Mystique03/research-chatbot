import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(
    page_title="Reasearch Assistant",
    page_icon=":books:",
    layout="wide"
)
st.title("Research Assistant :books:")
st.caption("Upload research papers and ask questions about them!")

# session state
if "papers" not in st.session_state:
    st.session_state.papers = {}
if "chat" not in st.session_state:
    st.session_state.chat = []

# Sidebar
with st.sidebar:
    st.header("Upload Papers")
    uploaded_files = st.file_uploader(
        "Choose a PDF", type="pdf", accept_multiple_files=True
    )

    if st.button("Process", type="primary") and uploaded_files:
        for f in uploaded_files:
            # Ingest paper
            with st.spinner(f"Ingesting {f.name}..."):
                resp =requests.post(
                    f"{API}/ingest",
                    files = {"file": (f.name, f.getvalue(), "application/pdf")}
                )
            if resp.status_code != 200:
                st.error(f"Failed: {resp.text}")
                continue

            paper_id = resp.json()["paper_id"]
            st.success(f"Ingested '{paper_id}' successfully!")

            # summerize
            with st.spinner(f"Generating summary..."):
                time.sleep(2)
                s = requests.post(f"{API}/summarize/{paper_id}")

            summary = s.json().get("summary", "") if s.status_code == 200 else ""
            st.session_state.papers[paper_id] = summary

    if st.session_state.papers:
        st.divider()
        st.subheader("Loaded")
        for pid in st.session_state.papers:
            st.write(f"- {pid}")

# Tabs
tab_summary, tab_chat = st.tabs(["Summary", "Chat"])

# Summary tab
with tab_summary:
    if not st.session_state.papers:
        st.info("Upload a paper to see its summary.")
    else:
        for paper_id, summary in st.session_state.papers.items():
            st.subheader(paper_id)
            st.markdown(summary or "Summary not available.")
            st.divider()

# Chat tab
with tab_chat:
    # replay history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")
            if msg.get("source_type") == "external":
                st.caption("Answer from ArXiv / PubMed / Web")

    # input 
    if prompt := st.chat_input("Ask about the paper..."):
        if not st.session_state.papers:
            st.warning("Please upload a paper first.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = requests.post(
                        f"{API}/query",
                        json={"question": prompt},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.write(f"- {s}")
                    if data["source_type"] == "external":
                        st.caption("Answer from ArXiv / PubMed / Web")

                    st.session_state.chat.append({
                        "role":        "assistant",
                        "content":     data["answer"],
                        "sources":     data["sources"],
                        "source_type": data["source_type"],
                    })
                else:
                    st.error(f"API error: {resp.text}")
