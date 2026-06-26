# vectorstore.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_vectorstore(chunks, save_path="faiss_index"):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

def load_vectorstore(save_path="faiss_index"):
    embeddings = get_embeddings()
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )