import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
    raise ValueError("GOOGLE_API_KEY is missing from your .env file!")

print("2. Initializing FastAPI...")
app = FastAPI(title="Readora Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("3. Connecting models...")

# Fixed: "models/embedding-001" is deprecated and returns a 404 on v1beta.
# Replaced with "models/gemini-embedding-001", the current supported embedding model.
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    output_dimensionality=768,  # keeps vector size compact and consistent
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    max_retries=5,
)

CHROMA_DIR = "./chroma_db"
PDF_PATH = "GRU.pdf"

print("4. Checking Vector DB...")
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("   -> Loading existing Chroma DB...")
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )
else:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing '{PDF_PATH}'. Ensure it is saved in {os.getcwd()}")
    print("   -> First-time setup: Indexing GRU.pdf into Chroma DB...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
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
    return FileResponse("index.html")


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        retrieved_docs = retriever.invoke(request.question)
        if not retrieved_docs:
            return {"answer": "I could not find the answer in the PDF.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        sources = [doc.metadata.get("page", 0) + 1 for doc in retrieved_docs if "page" in doc.metadata]

        prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I could not find the answer in the PDF."

Context:
{context}

Question:
{request.question}"""

        response = invoke_with_retry(prompt)

        # Some model responses return content as a list of parts instead of
        # a plain string. Normalize it so the frontend always gets clean text.
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
            "sources": list(set(sources))
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Ready! Access app at: http://127.0.0.1:8000\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)