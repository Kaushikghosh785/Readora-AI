import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

print("1. Loading environment variables...")
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing! Set it in your .env file locally, "
        "or in Render's dashboard under Environment > Environment Variables."
    )

print("2. Initializing FastAPI...")
app = FastAPI(title="Readora Backend API")

# NOTE: allow_credentials must stay False while allow_origins is "*".
# The frontend calls the API with a relative path (same-origin), so
# CORS barely matters for the deployed app -- this wildcard just keeps
# things working if you ever call the API from a different origin
# (e.g. testing from another host) without cookies/auth headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("3. Connecting models...")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    output_dimensionality=768,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    max_retries=5,
)

# Resolve paths relative to this file, not the process's current working
# directory -- Render (and some process managers) may launch the app from
# a different cwd than you expect, which causes "file not found" errors
# even though the file is right next to app.py in your repo.
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
PDF_PATH = BASE_DIR / "GRU.pdf"
INDEX_HTML_PATH = BASE_DIR / "index.html"

print("4. Checking Vector DB...")
if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
    print("   -> Loading existing Chroma DB...")
    vector_db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model
    )
else:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Missing '{PDF_PATH.name}'. Make sure it is committed to your "
            f"repo in the same folder as app.py (expected at: {PDF_PATH})."
        )
    print("   -> First-time setup: Indexing GRU.pdf into Chroma DB...")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)


def invoke_with_retry(prompt, max_attempts=5, base_delay=2):
    """
    Calls the LLM, retrying with exponential backoff if the model is
    temporarily overloaded (503 UNAVAILABLE). This is a transient,
    server-side condition on Google's end -- not a bad model name --
    so retrying after a short wait usually succeeds.

    This function is blocking (time.sleep + a synchronous SDK call), so
    it must always be run off the event loop -- see run_in_threadpool
    below in chat_endpoint. Do not call it directly inside an `async def`.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s, 16s, 32s
                print(f"   -> Model overloaded (attempt {attempt}/{max_attempts}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise  # non-503 errors fail immediately, no point retrying
    raise last_error


class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def serve_frontend():
    return FileResponse(str(INDEX_HTML_PATH))


@app.get("/api/health")
async def health_check():
    # Handy for confirming the deployed backend is actually reachable,
    # separately from the RAG pipeline (e.g. curl your-app.onrender.com/api/health)
    return {"status": "ok"}


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters).")

    try:
        # retriever.invoke and invoke_with_retry are both blocking (sync SDK
        # calls, plus time.sleep on retries). Running them directly inside
        # this async endpoint would block the whole server for every other
        # concurrent request. run_in_threadpool offloads them to a worker
        # thread so the event loop stays free.
        retrieved_docs = await run_in_threadpool(retriever.invoke, question)
        if not retrieved_docs:
            return {"answer": "I could not find the answer in the PDF.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        sources = [doc.metadata.get("page", 0) + 1 for doc in retrieved_docs if "page" in doc.metadata]

        prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I could not find the answer in the PDF."

Context:
{context}

Question:
{question}"""

        response = await run_in_threadpool(invoke_with_retry, prompt)

        raw_content = response.content if hasattr(response, 'content') else response
        if isinstance(raw_content, list):
            answer_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            answer_text = str(raw_content)

        return {
            "answer": answer_text,
            "sources": sorted(set(sources))
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Render assigns a random port at runtime via the PORT env var and
    # routes external traffic to it -- binding to a hardcoded 127.0.0.1:8000
    # only accepts connections from inside the container itself, so Render's
    # proxy can never reach it. Binding 0.0.0.0 + $PORT fixes that.
    # reload=True is a dev-only feature (auto-restarts on file changes) and
    # should never run in production.
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Ready! Listening on 0.0.0.0:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)