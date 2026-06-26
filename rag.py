# rag.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load embedding model (for query embedding only)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load FAISS index from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever — fetches top 3 relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Local LLM via Ollama
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.3
)

# Prompt
prompt = PromptTemplate.from_template("""
You are a helpful IT support assistant.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

# Format retrieved chunks into plain text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question
query = "How to set up an email app?"

print("\n--- Answer ---")
answer = rag_chain.invoke(query)
print(answer)

print("\n--- Sources Used ---")
docs = retriever.invoke(query)
for doc in docs:
    print(f"Topic: {doc.metadata['topic']} | Source: {doc.metadata['source']}")