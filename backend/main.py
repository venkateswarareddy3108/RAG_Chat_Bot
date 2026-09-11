"""
Corpus API — production FastAPI backend for document-scoped RAG.

Features:
  - SSE ingestion + chat streams
  - Per-document retrieval scoping
  - Client or server Gemini API key
  - PDF file serving for in-app preview
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import settings
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("corpus.api")

rag_engine: RAGEngine | None = None


def _safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\- ()\[\]]+", "_", base).strip("._")
    return cleaned or "document.pdf"


def _unique_destination(filename: str) -> Path:
    """Avoid silently overwriting uploads with the same name."""
    candidate = settings.upload_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for i in range(1, 10_000):
        alt = settings.upload_dir / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
    raise HTTPException(status_code=500, detail="Could not allocate upload filename")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global rag_engine
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Corpus API")
    rag_engine = RAGEngine()
    logger.info("RAG engine ready")
    yield
    logger.info("Shutting down Corpus API")


app = FastAPI(
    title="Corpus API",
    description="Document-scoped retrieval-augmented Q&A",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_engine() -> RAGEngine:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return rag_engine


def _resolve_api_key(
    header_key: str | None = None,
    body_key: str | None = None,
) -> str:
    """Prefer frontend-provided key; fall back to server .env."""
    for candidate in (header_key, body_key, settings.gemini_api_key):
        if candidate and candidate.strip():
            return candidate.strip()
    raise HTTPException(
        status_code=400,
        detail="Gemini API key required. Enter it in Settings or set GEMINI_API_KEY in backend/.env",
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "corpus-api",
        "version": "2.1.0",
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }


@app.get("/api/key-status")
async def key_status():
    """Reports only whether a server-side fallback key exists (never the client key)."""
    key = settings.gemini_api_key
    return {
        "configured": bool(key),
        "preview": f"{key[:8]}…" if key else "",
        "source": "server" if key else "none",
    }


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    x_gemini_api_key: str | None = Header(default=None, alias="X-Gemini-Api-Key"),
):
    engine = _require_engine()
    api_key = _resolve_api_key(header_key=x_gemini_api_key)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    filename = _safe_filename(file.filename)
    destination = _unique_destination(filename)
    stored_name = destination.name

    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    logger.info("Upload received: %s → %s", filename, stored_name)

    async def event_stream():
        async for update in engine.process_pdf(
            str(destination),
            stored_name,
            api_key,
            original_name=file.filename,
        ):
            yield {"event": "pipeline", "data": json.dumps(update)}

    return EventSourceResponse(event_stream())


@app.post("/api/chat")
async def chat(
    request: Request,
    x_gemini_api_key: str | None = Header(default=None, alias="X-Gemini-Api-Key"),
):
    engine = _require_engine()
    body = await request.json()
    question = (body.get("question") or "").strip()
    doc_id = (body.get("doc_id") or "").strip() or None
    body_key = (body.get("api_key") or "").strip() or None

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    if not doc_id:
        raise HTTPException(
            status_code=400,
            detail="Select a document first. Answers are scoped to the selected PDF only.",
        )
    if not engine.document_exists(doc_id):
        raise HTTPException(status_code=404, detail="Selected document was not found")

    api_key = _resolve_api_key(header_key=x_gemini_api_key, body_key=body_key)

    async def event_stream():
        async for update in engine.query(question, api_key, doc_id=doc_id):
            yield {"event": "pipeline", "data": json.dumps(update)}

    return EventSourceResponse(event_stream())


@app.get("/api/documents")
async def list_documents():
    return {"documents": _require_engine().get_documents()}


@app.get("/api/documents/{doc_id}/file")
async def get_document_file(doc_id: str):
    """Stream the original PDF for in-app preview."""
    engine = _require_engine()
    meta = engine.get_document(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    path = engine.get_pdf_path(doc_id)
    if path is None or not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"PDF file for '{meta.get('name', doc_id)}' was not found in uploads. "
                "Re-upload the document to enable preview."
            ),
        )

    logger.info("Serving PDF %s (%s)", doc_id, path.name)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        headers={
            "Content-Disposition": f'inline; filename="{path.name}"',
            "Cache-Control": "private, max-age=60",
        },
    )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    engine = _require_engine()
    if not engine.document_exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    engine.delete_document(doc_id)
    logger.info("Deleted document %s", doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
