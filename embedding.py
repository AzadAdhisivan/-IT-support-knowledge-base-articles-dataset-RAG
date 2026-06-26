from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunking import chunks

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"Embedding {len(chunks)} chunks into FAISS... this may take a moment.")

# Embed chunks and store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk so you don't re-embed every time
vectorstore.save_local("faiss_index")

print("Vector store saved to ./faiss_index/")

# Quick test — similarity search
query = "How do I reset my password?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 3 results for: '{query}'")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print("Topic:", doc.metadata["topic"])
    print("Source:", doc.metadata["source"])
    print(doc.page_content[:200])