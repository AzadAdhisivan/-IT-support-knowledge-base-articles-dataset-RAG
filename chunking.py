import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(df, text_columns, metadata_columns=None, chunk_size=800, chunk_overlap=100):
    docs = []

    for col in text_columns:
        if col not in df.columns:
            continue

        col_df = df[df[col].notna()][([col] + (metadata_columns or []))]

        for _, row in col_df.iterrows():
            metadata = {"source": col}
            if metadata_columns:
                metadata.update({m: row[m] for m in metadata_columns if m in df.columns})

            docs.append(Document(
                page_content=str(row[col]),
                metadata=metadata
            ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)