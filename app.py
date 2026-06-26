# app.py
import streamlit as st
import pandas as pd
from chunking import load_and_chunk
from vectorstore import build_vectorstore, load_vectorstore
from rag import build_rag_chain, ask
from evaluate import evaluate
from cleaner import clean_csv

st.set_page_config(page_title="RAG Builder", layout="wide")
st.title("RAG Builder")
st.caption("Upload any CSV, select text columns, and ask questions.")

with st.sidebar:
    st.header("Setup")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")

        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        st.subheader("Select Text Columns to Embed")
        selected_text_cols = st.multiselect(
            "Text columns",
            text_cols,
            default=text_cols[:3] if len(text_cols) >= 3 else text_cols
        )

        st.subheader("Select Metadata Columns (optional)")
        selected_meta_cols = st.multiselect(
            "Metadata columns",
            df.columns.tolist()
        )

        if st.button("Build Vector Store", type="primary"):
            if not selected_text_cols:
                st.error("Please select at least one text column.")
            else:
                with st.spinner("Cleaning CSV..."):
                    df_clean, clean_report = clean_csv(df, selected_text_cols)
                    st.info(
                        f"Cleaning done — "
                        f"{clean_report['rows_dropped_null']} null rows dropped, "
                        f"{clean_report['rows_dropped_duplicates']} duplicates removed. "
                        f"{clean_report['clean_rows']} rows remaining."
                    )

                with st.spinner("Chunking and embedding... this may take a minute."):
                    chunks = load_and_chunk(
                        df_clean,
                        text_columns=selected_text_cols,
                        metadata_columns=selected_meta_cols
                    )
                    build_vectorstore(chunks)
                    st.session_state["vectorstore_ready"] = True
                    st.success(f"Vector store built from {len(chunks)} chunks!")

if st.session_state.get("vectorstore_ready"):
    st.header("Ask a Question")

    vectorstore = load_vectorstore()
    rag_chain, retriever = build_rag_chain(vectorstore)

    query = st.text_input("Enter your question:")

    if st.button("Ask", type="primary") and query:
        with st.spinner("Thinking..."):
            answer, sources = ask(rag_chain, retriever, query)
            eval_results = evaluate(query, answer)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources Retrieved")
        for i, doc in enumerate(sources):
            with st.expander(f"Source {i+1} — {doc.metadata.get('source', 'unknown')}"):
                st.write(doc.page_content)
                st.json(doc.metadata)

        st.subheader("Evaluation")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Answer Relevance", eval_results['verdict'])
            st.metric("Relevance Score", eval_results['relevance_score'])

        with col2:
            st.write("Keywords checked:", eval_results['keywords_checked'])
            st.write(f"Matched: {eval_results['keywords_matched']} / {len(eval_results['keywords_checked'])}")

else:
    st.info("Upload a CSV and build the vector store from the sidebar to get started.")