import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

df = pd.read_csv("synthetic_knowledge_items.csv")

docs = []

for _, row in df.iterrows():

    docs.append(
        Document(
            page_content=row["ki_text"],
            metadata={
                "topic": row["ki_topic"],
                "source": "ki_text"
            }
        )
    )

    docs.append(
        Document(
            page_content=row["alt_ki_text"],
            metadata={
                "topic": row["ki_topic"],
                "source": "alt_ki_text"
            }
        )
    )

    docs.append(
        Document(
            page_content=row["bad_ki_text"],
            metadata={
                "topic": row["ki_topic"],
                "source": "bad_ki_text"
            }
        )
    )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:5]):
    print(f"\n----- Chunk {i+1} -----")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)