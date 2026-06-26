# rag.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
You are a helpful assistant.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def ask(rag_chain, retriever, query):
    answer = rag_chain.invoke(query)
    sources = retriever.invoke(query)
    return answer, sources