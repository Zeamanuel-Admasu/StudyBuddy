# StudyBuddy — Local RAG with LangChain + FAISS + Ollama

StudyBuddy is a small learning project that answers questions using **only your local documents** (RAG: Retrieval-Augmented Generation).

It:
- Builds a **FAISS** vector index from files in `data/`
- Retrieves relevant chunks for each question
- Generates answers using a **local Ollama model**
- Shows **sources** with tags like `[S1]`, `[S2]` so you can see where answers come from
- Asks a **clarifying question** when retrieval seems weak (so it does not guess)

---

## Project Structure

```
studdybuddy/
  data/
    notes.txt
  ingest.py
  chat_rag.py
  requirements.txt
  .env.example
  .gitignore
  README.md
```

---

## Requirements

- Python 3.10+ recommended
- Ollama installed on Windows
- An Ollama model downloaded (example: `llama3.1:8b`)

---

## Setup (Windows PowerShell)

### 1) Create and activate a virtual environment

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
pip install -U langchain-huggingface
```

---

## Install & Configure Ollama

### 1) Verify Ollama is installed

```powershell
ollama --version
```

If you see a version like `ollama version is ...`, you are good.

### 2) Download a model (pull)

```powershell
ollama pull llama3.1:8b
```

### 3) Confirm it is downloaded

```powershell
ollama list
```

You should see something like:

- `llama3.1:8b`

### 4) Quick test (optional)

```powershell
ollama run llama3.1:8b
```

Exit the chat with:

```
/bye
```

---

## Add Your Documents

Put your files inside the `data/` folder.

Supported:
- `.txt`
- `.pdf` (text-based PDFs)

Example:

```
data/
  notes.txt
  lecture.pdf
```

> If your PDF is scanned images (no selectable text), `PyPDFLoader` will not extract text. You would need OCR.

---

## Build the Vector Index (Ingest)

Run:

```powershell
python ingest.py
```

This creates:

```
faiss_index/
```

---

## Run the Chatbot

Run:

```powershell
python chat_rag.py
```

Exit by typing:

```
exit
```

---

## How RAG Works (Simple Explanation)

1) **Ingest (one-time)**
   - Load documents from `data/`
   - Split into chunks
   - Create embeddings (vectors) for each chunk
   - Store vectors in FAISS

2) **Chat (every question)**
   - Embed your question
   - Retrieve the most relevant chunks from FAISS
   - Send: question + retrieved chunks to the model
   - The model answers using only the retrieved context

---

## Sources: `[S1]`, `[S2]`, ...

When you ask a question, StudyBuddy retrieves multiple chunks and labels them:

- `[S1]` = most relevant chunk
- `[S2]` = second most relevant chunk
- etc.

The prompt includes an **Available sources** list to stop the model from inventing citation tags.

---

## Clarification Loop (Weak Retrieval)

If retrieval seems weak (example: you type `hi` or ask something unrelated), StudyBuddy asks you to clarify:

- “Can you rephrase or give a keyword/name from the document?”

Then it re-runs retrieval using your clarification before answering.

---

## Common Issues

### 1) Ollama model not found (404)

Run:

```powershell
ollama list
```

Make sure the model name in `chat_rag.py` matches exactly what you see (for example `llama3.1:8b`).

### 2) Ingest creates only 1 chunk

If ingest prints:

```
Saved FAISS index with 1 chunks.
```

Your document is very small or chunk size is large. This is fine for small documents.

### 3) PDF has no text

If your PDF is scanned images, `PyPDFLoader` will not extract text. You would need OCR.

---

## GitHub Push (Quick Steps)

```powershell
git init
git add .
git commit -m "Initial StudyBuddy RAG project"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

---

## Next Steps (Planned)

- Add an agent version (`chat_agent.py`) that can decide when to search notes and when to do math
- Upgrade to LangGraph for cleaner branching/loops and better state handling
