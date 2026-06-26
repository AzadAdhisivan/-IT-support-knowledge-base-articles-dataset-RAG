# RAG Builder

A flexible, end-to-end Retrieval-Augmented Generation (RAG) system that works with **any CSV file**. Upload a CSV, select your text columns, and ask questions — powered by a fully local LLM with no API keys required.

## Project Overview

This project builds a complete RAG pipeline with a Streamlit UI. It supports any CSV dataset, automatically detects text columns, cleans the data, embeds it into a FAISS vector store, and uses a local LLM via Ollama to generate grounded answers with evaluation scoring.

## Pipeline
CSV Upload (Streamlit UI)

↓

cleaner.py        → drops null rows and duplicates

↓

chunking.py       → splits text into overlapping chunks

↓

vectorstore.py    → embeds chunks using bge-small-en-v1.5 → stores in FAISS

↓

rag.py            → retrieves top 3 relevant chunks → Llama 3.2 generates answer

↓

evaluate.py       → scores answer relevance via keyword matching

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS |
| LLM | Llama 3.2 1B via Ollama (local) |
| Framework | LangChain |

## Features

- Upload any CSV file through the UI
- Auto-detects text and numeric columns
- Select multiple text columns to embed
- Select metadata columns for source attribution
- CSV cleaning before chunking (null rows, duplicates)
- GPU or CPU selection for embedding
- Answer relevance evaluation with keyword scoring
- Source document viewer per answer

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed
- Nvidia GPU recommended for large CSVs (optional)

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

**Run the Streamlit app**

```bash
streamlit run app.py
```

Then in the UI:
1. Upload your CSV file
2. Select which text columns to embed
3. Optionally select metadata columns
4. Choose CPU or GPU for embedding
5. Click **Build Vector Store**
6. Type your question and click **Ask**

## Project Structure
RAG BUILDER/

│

├── app.py               # Streamlit UI — main entry point

├── cleaner.py           # CSV cleaning (nulls, duplicates)

├── chunking.py          # Splits documents into chunks

├── vectorstore.py       # Embeds chunks and builds FAISS index

├── rag.py               # Retrieval + LLM answer generation

├── evaluate.py          # Answer relevance evaluation

├── requirements.txt     # Python dependencies

└── .gitignore

## Example Output
Query: "How do I reset my password?"
--- Answer ---

To reset your forgotten computer password, go to the Password Reset Tool

website and click on Forgot Password. Enter your username or email address

and follow the instructions sent to your registered email.
--- Sources Retrieved ---

Source 1 — ki_text

Source 2 — alt_ki_text

Source 3 — ki_text
--- Evaluation ---

Answer Relevance : PASS

Relevance Score  : 1.0

Keywords Checked : ['reset', 'password']

Matched          : 2 / 2

## How It Works

**1. Cleaning**
The uploaded CSV is cleaned before processing — rows with null values across all selected text columns are dropped, and duplicate rows are removed.

**2. Chunking**
Text columns are loaded into LangChain Document objects with metadata. Each document is split into chunks of 800 characters with 100 character overlap using RecursiveCharacterTextSplitter.

**3. Embedding and Vector Store**
Each chunk is converted into a 384-dimensional vector using the BAAI/bge-small-en-v1.5 embedding model. These vectors are stored in a FAISS index saved to disk. GPU acceleration is supported for faster embedding on large CSVs.

**4. Retrieval**
When a query is received, it is embedded using the same model. FAISS performs cosine similarity search and returns the top 3 most semantically similar chunks.

**5. Generation**
The retrieved chunks are passed as context to Llama 3.2 1B running locally via Ollama. The LLM generates a grounded answer based only on the retrieved context. If the answer is not in the context, it returns "I don't know."

**6. Evaluation**
Answer relevance is scored by extracting meaningful keywords from the query (excluding stop words and punctuation) and checking how many appear in the generated answer. A score of 0.5 or above is a PASS.

## Why Local LLM

Using Ollama with Llama 3.2 1B means:
- No API keys required
- No token limits or costs
- Data stays on your machine
- Works fully offline after setup