import os
import time
import uuid
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
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

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "index.html"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_ROOT = BASE_DIR / "chroma_sessions"
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_ROOT.mkdir(exist_ok=True)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB per upload
SESSION_TTL_SECONDS = 3 * 60 * 60  # sessions expire after 3 hours

SESSIONS = {}


def _cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s["created"] > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSIONS.pop(sid, None)
        shutil.rmtree(CHROMA_ROOT / sid, ignore_errors=True)
        (UPLOAD_DIR / f"{sid}.pdf").unlink(missing_ok=True)


class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
async def serve_frontend():
    return FileResponse(str(INDEX_HTML_PATH))


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


def _add_batch_with_retry(vector_db, batch, max_retries=5, base_delay=3):
    """Adds a batch of document chunks to ChromaDB with backoff retry to prevent 429 quota errors."""
    for attempt in range(1, max_retries + 1):
        try:
            if vector_db is None:
                return Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_model,
                )
            else:
                vector_db.add_documents(documents=batch)
                return vector_db
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = base_delay * (2 ** (attempt - 1))
                print(f"   -> Embedding quota reached (attempt {attempt}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Failed to generate embeddings after max retries due to Gemini API rate limits.")


def _process_pdf(file_path: Path, session_id: str):
    """Blocking work: parse the PDF, split it, and embed it into Chroma in small batches."""
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    if not documents:
        raise ValueError(
            "Couldn't extract any text from this PDF. It may be a scanned "
            "image with no selectable text (OCR isn't supported yet)."
        )

    # Slightly larger chunks = fewer embedding requests
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    persist_dir = CHROMA_ROOT / session_id
    batch_size = 10  # Batch 10 chunks at a time to stay under free tier rate limits
    vector_db = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Indexing batch {i // batch_size + 1} of {(len(chunks) + batch_size - 1) // batch_size}...")
        
        if vector_db is None:
            vector_db = Chroma(
                collection_name=session_id,
                embedding_function=embedding_model,
                persist_directory=str(persist_dir)
            )
        
        vector_db = _add_batch_with_retry(vector_db, batch)
        time.sleep(1.5)  # Pause between batches to avoid overloading API rate limit

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    return retriever, len(documents)


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    _cleanup_expired_sessions()

    session_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{session_id}.pdf"
    saved_path.write_bytes(contents)

    try:
        retriever, num_pages = await run_in_threadpool(_process_pdf, saved_path, session_id)
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        shutil.rmtree(CHROMA_ROOT / session_id, ignore_errors=True)
        print(f"Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    SESSIONS[session_id] = {
        "retriever": retriever,
        "filename": file.filename,
        "pages": num_pages,
        "size_bytes": len(contents),
        "created": time.time(),
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": num_pages,
        "size_bytes": len(contents),
    }


@app.get("/api/document/{session_id}")
async def get_document(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Document not found or session expired.")
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document file missing on server.")
    return FileResponse(str(file_path), media_type="application/pdf", filename=session["filename"])


def invoke_with_retry(prompt, max_attempts=5, base_delay=2):
    """Calls the LLM, retrying with exponential backoff if overloaded."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                wait = base_delay * (2 ** (attempt - 1))
                print(f"   -> Model busy/overloaded (attempt {attempt}/{max_attempts}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise last_error


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters).")

    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="No document found for this session. Please upload a PDF first."
        )

    retriever = session["retriever"]

    try:
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
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Ready! Listening on 0.0.0.0:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)