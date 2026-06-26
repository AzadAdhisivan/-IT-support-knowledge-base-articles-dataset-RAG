# IT Support Knowledge Base RAG System

A Retrieval-Augmented Generation (RAG) pipeline that answers IT support questions by retrieving relevant knowledge articles and generating accurate answers using a local LLM.

## Project Overview

This project builds an end-to-end RAG system on a synthetic IT support knowledge base dataset. Each topic in the dataset contains three versions of an article — a good article, an alternative good article, and an intentionally bad article — making it suitable for RAG evaluation.

## Pipeline
CSV Dataset

↓

chunking.py       → splits articles into overlapping chunks

↓

vectorstore.py    → embeds chunks and stores in FAISS

↓

rag.py            → retrieves top 3 relevant chunks → Llama 3.2 generates answer


## Tech Stack

| Component | Tool |
|---|---|
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS |
| LLM | Llama 3.2 1B via Ollama (local) |
| Framework | LangChain |

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/AzadAdhisivan/-IT-support-knowledge-base-articles-dataset-RAG.git
cd -IT-support-knowledge-base-articles-dataset-RAG
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Pull the local LLM via Ollama**

```bash
ollama pull llama3.2:1b
```

## Usage

**Step 1 — Chunk the dataset**

```bash
python chunking.py
```

**Step 2 — Build the FAISS vector store**

```bash
python vectorstore.py
```

This embeds all chunks and saves the FAISS index to `faiss_index/` locally. Only needs to be run once.

**Step 3 — Run the RAG pipeline**

```bash
python rag.py
```

Edit the `query` variable in `rag.py` to ask any IT support question.

## Project Structure
RAG BUILDER/

│

├── synthetic_knowledge_items.csv  # IT support knowledge base dataset

├── chunking.py                    # Loads CSV and splits into chunks

├── vectorstore.py                 # Embeds chunks and builds FAISS index

├── rag.py                         # Retrieval + LLM answer generation

├── requirements.txt               # Python dependencies

└── .gitignore

## Example Output
Query: "How do I reset my password?"
--- Answer ---

To reset your forgotten computer password, go to the Password Reset Tool

website and click on Forgot Password. Enter your username or email address

and follow the instructions sent to your registered email.
--- Sources Used ---

Topic: Resetting a Forgotten Computer Password | Source: ki_text

Topic: Resetting a Forgotten Computer Password | Source: alt_ki_text

Topic: Resetting a Forgotten Computer Password | Source: ki_text

## Dataset Structure

The CSV contains IT support articles with the following columns:

| Column | Description |
|---|---|
| `ki_topic` | Topic of the article |
| `ki_text` | High quality knowledge article |
| `alt_ki_text` | Alternative well-written version |
| `bad_ki_text` | Intentionally poorly written version |

The three versions per topic make this dataset suitable for evaluating RAG retrieval quality — specifically whether the system retrieves high quality articles over poorly written ones.

## How It Works

**1. Chunking**
The CSV is loaded into LangChain Document objects with metadata (topic and source). Each document is split into chunks of 800 characters with 100 character overlap using RecursiveCharacterTextSplitter.

**2. Embedding and Vector Store**
Each chunk is converted into a 384-dimensional vector using the BAAI/bge-small-en-v1.5 embedding model. These vectors are stored in a FAISS index saved to disk.

**3. Retrieval**
When a query is received, it is embedded using the same model. FAISS performs cosine similarity search and returns the top 3 most semantically similar chunks.

**4. Generation**
The retrieved chunks are passed as context to Llama 3.2 1B running locally via Ollama. The LLM generates a grounded answer based only on the retrieved context.

## Why Local LLM

Using Ollama with Llama 3.2 1B means:
- No API keys required
- No token limits or costs
- Data stays on your machine
- Works fully offline after setup