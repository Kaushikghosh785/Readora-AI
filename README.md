# Readora AI — Talk to Your Documents 📄✨

**Readora AI** is a full-stack, Retrieval-Augmented Generation (RAG) platform that allows users to interact conversationally with complex PDF documents. Built with a **FastAPI** backend, **LangChain**, **ChromaDB**, and **Google Gemini**, Readora delivers precise, grounded answers accompanied by page-level citations.

---

## 🚀 Key Features

- **Document Grounding (RAG)**: Answers questions using *only* the context extracted from uploaded documents, preventing hallucinations.
- **Page-Level Source Citations**: Automatically tracks and renders page citations for every response.
- **Dynamic Vector Search**: Uses **Maximal Marginal Relevance (MMR)** over ChromaDB to balance context relevance and diversity.
- **Transient Error Resilience**: Implements custom exponential backoff retry mechanisms to handle API rate limits and model overloads gracefully.
- **Modern Glassmorphic UI**: Sleek, responsive web interface built with HTML5, Tailwind CSS, Lucide icons, and Marked.js.

---

## 🛠️ Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM Engine**: Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai`
- **Embedding Model**: Google Embeddings (`text-embedding-004`)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Frontend**: HTML, Tailwind CSS, Marked.js, Lucide Icons

---

## 📁 Repository Structure

```text
my_project/
├── app.py              # FastAPI server & RAG pipeline
├── index.html          # Web frontend interface
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Excluded files list
└── README.md           # Project documentation
