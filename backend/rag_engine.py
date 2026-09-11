"""
RAG engine: PDF → extract → clean → chunk → batch → BGE-M3 → ChromaDB → Gemini.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import chromadb
import pymupdf
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger("corpus.rag")


class RAGEngine:
    """Retrieval-augmented generation pipeline with streaming status events."""

    COLLECTION_NAME = "rag_bge_m3"

    def __init__(self) -> None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ChromaDB at %s", settings.chroma_dir)
        self.chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.embedding_model,
            },
        )

        logger.info("Loading embedding model %s", settings.embedding_model)
        self.embedder = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model ready")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.documents_meta = self._load_docs_meta()
        if self.collection.count() == 0 and self.documents_meta:
            self.documents_meta = {}
            self._save_docs_meta()

    # ── Metadata ──────────────────────────────────

    def _load_docs_meta(self) -> dict:
        path = settings.docs_meta_file
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_docs_meta(self) -> None:
        with settings.docs_meta_file.open("w", encoding="utf-8") as f:
            json.dump(self.documents_meta, f, indent=2)

    def _refresh_docs_meta(self) -> None:
        """Re-read docs_meta.json so file preview works after restarts/uploads."""
        self.documents_meta = self._load_docs_meta()

    # ── Text pipeline ─────────────────────────────

    def _extract_text(self, pdf_path: str) -> tuple[str, list[str], int]:
        doc = pymupdf.open(pdf_path)
        pages = [page.get_text() for page in doc]
        total = len(doc)
        doc.close()
        return "\n".join(pages), pages, total

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if re.fullmatch(r"[\d\-–—·•.|]+", stripped):
                continue
            lines.append(stripped)
        return "\n".join(lines).strip()

    def _chunk_text(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)

    def _create_batches(self, chunks: list[str]) -> list[list[str]]:
        size = settings.embedding_batch_size
        return [chunks[i : i + size] for i in range(0, len(chunks), size)]

    # ── Embeddings ────────────────────────────────

    def _embed_batch_sync(self, batch: list[str]) -> list[list[float]]:
        vectors = self.embedder.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = [v.tolist() for v in vectors]
        if len(embeddings) != len(batch):
            raise ValueError(
                f"Embedding mismatch: sent {len(batch)}, got {len(embeddings)}"
            )
        return embeddings

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_batch_sync, batch)

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for batch in self._create_batches(texts):
            all_embeddings.extend(await self._embed_batch(batch))
        return all_embeddings

    # ── Vector store ──────────────────────────────

    def _store_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        doc_id: str,
        doc_name: str,
    ) -> None:
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "doc_name": doc_name, "chunk_index": i}
            for i in range(len(chunks))
        ]
        batch_size = 166
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                documents=chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

    def _similarity_search(
        self,
        query_embedding: list[float],
        k: int | None = None,
        doc_id: str | None = None,
    ):
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k or settings.retrieval_top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        return self.collection.query(**kwargs)

    def get_documents(self) -> list[dict]:
        self._refresh_docs_meta()
        docs = list(self.documents_meta.values())
        docs.sort(key=lambda d: d.get("processed_at", 0), reverse=True)
        return docs

    def document_exists(self, doc_id: str) -> bool:
        self._refresh_docs_meta()
        return doc_id in self.documents_meta

    def get_document(self, doc_id: str) -> dict | None:
        self._refresh_docs_meta()
        return self.documents_meta.get(doc_id)

    def get_pdf_path(self, doc_id: str) -> Path | None:
        self._refresh_docs_meta()
        meta = self.documents_meta.get(doc_id)
        if not meta:
            return None

        candidates: list[str] = []
        for key in ("stored_name", "name"):
            value = meta.get(key)
            if value:
                candidates.append(Path(str(value)).name)

        # Also try common sanitized variants
        for name in list(candidates):
            candidates.append(name.replace(" ", "_"))

        seen: set[str] = set()
        for name in candidates:
            if not name or name in seen:
                continue
            seen.add(name)
            path = settings.upload_dir / name
            if path.exists() and path.is_file():
                return path

        # Last resort: case-insensitive match in uploads/
        if settings.upload_dir.exists():
            wanted = {n.lower() for n in seen}
            for path in settings.upload_dir.iterdir():
                if path.is_file() and path.name.lower() in wanted:
                    return path
        return None

    def delete_document(self, doc_id: str) -> None:
        results = self.collection.get(where={"doc_id": doc_id}, include=[])
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        meta = self.documents_meta.pop(doc_id, None)
        self._save_docs_meta()
        if meta:
            stored = meta.get("stored_name") or meta.get("name")
            if stored:
                path = settings.upload_dir / Path(stored).name
                try:
                    if path.exists():
                        path.unlink()
                except OSError as exc:
                    logger.warning("Could not delete PDF file %s: %s", path, exc)

    # ── Ingestion ─────────────────────────────────

    async def process_pdf(
        self,
        pdf_path: str,
        filename: str,
        api_key: str,
        original_name: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del api_key  # embeddings are local (BGE-M3); key reserved for future use
        doc_id = str(uuid.uuid4())[:8]
        display_name = original_name or filename

        yield _status("upload", "completed", f"Validated “{display_name}”")

        yield _status("extraction", "processing", "Extracting text from PDF…")
        try:
            full_text, _pages, total_pages = await asyncio.to_thread(
                self._extract_text, pdf_path
            )
            if not full_text.strip():
                yield _status("extraction", "error", "PDF has no extractable text")
                return
            yield _status(
                "extraction",
                "completed",
                f"{total_pages} pages · {len(full_text):,} characters",
            )
        except Exception as exc:
            yield _status("extraction", "error", str(exc))
            return

        yield _status("cleaning", "processing", "Normalizing text…")
        try:
            cleaned = await asyncio.to_thread(self._clean_text, full_text)
            char_count = len(cleaned)
            if not char_count:
                yield _status("cleaning", "error", "No text left after cleaning")
                return
            yield _status("cleaning", "completed", f"{char_count:,} characters retained")
        except Exception as exc:
            yield _status("cleaning", "error", str(exc))
            return

        yield _status("chunking", "processing", "Creating semantic chunks…")
        try:
            chunks = await asyncio.to_thread(self._chunk_text, cleaned)
            yield _status(
                "chunking",
                "completed",
                f"{len(chunks)} chunks · size {settings.chunk_size} · overlap {settings.chunk_overlap}",
            )
        except Exception as exc:
            yield _status("chunking", "error", str(exc))
            return

        yield _status("batching", "processing", "Building embedding batches…")
        try:
            batches = self._create_batches(chunks)
            yield _status(
                "batching",
                "completed",
                f"{len(batches)} batch(es) · up to {settings.embedding_batch_size} each",
            )
        except Exception as exc:
            yield _status("batching", "error", str(exc))
            return

        yield _status(
            "embedding",
            "processing",
            f"Encoding {len(chunks)} chunks with BGE-M3…",
        )
        try:
            embeddings: list[list[float]] = []
            for batch in batches:
                embeddings.extend(await self._embed_batch(batch))
            dim = len(embeddings[0]) if embeddings else 0
            yield _status(
                "embedding",
                "completed",
                f"{len(embeddings)} vectors · {dim} dimensions",
            )
        except Exception as exc:
            yield _status("embedding", "error", str(exc))
            return

        yield _status("storage", "processing", "Persisting vectors to ChromaDB…")
        try:
            await asyncio.to_thread(
                self._store_chunks, chunks, embeddings, doc_id, display_name
            )
            self.documents_meta[doc_id] = {
                "id": doc_id,
                "name": display_name,
                "stored_name": filename,
                "pages": total_pages,
                "chunks": len(chunks),
                "characters": char_count,
                "processed_at": time.time(),
            }
            self._save_docs_meta()
            yield _status(
                "storage",
                "completed",
                f"Stored {len(chunks)} vectors in “{self.COLLECTION_NAME}”",
            )
        except Exception as exc:
            yield _status("storage", "error", str(exc))
            return

        yield {
            "step": "complete",
            "status": "completed",
            "details": f"“{display_name}” is ready for questions",
            "doc_id": doc_id,
            "timestamp": time.time(),
        }

    # ── Query ─────────────────────────────────────

    async def query(
        self,
        question: str,
        api_key: str,
        doc_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        scope_name = "all documents"
        if doc_id:
            meta = self.get_document(doc_id) or {}
            scope_name = meta.get("name") or doc_id

        yield _status("query_processing", "processing", "Parsing question…")
        display = f"“{question[:80]}…”" if len(question) > 80 else f"“{question}”"
        yield _status(
            "query_processing",
            "completed",
            f"Query: {display} · scoped to “{scope_name}”",
        )

        yield _status("query_embedding", "processing", "Encoding query with BGE-M3…")
        try:
            query_embedding = await self._generate_embeddings([question])
            dim = len(query_embedding[0]) if query_embedding else 0
            yield _status(
                "query_embedding",
                "completed",
                f"Query vector · {dim} dimensions",
            )
        except Exception as exc:
            yield _status("query_embedding", "error", str(exc))
            return

        yield _status(
            "similarity_search",
            "processing",
            f"Searching within “{scope_name}”…",
        )
        try:
            results = await asyncio.to_thread(
                self._similarity_search, query_embedding[0], None, doc_id
            )
            docs = results["documents"][0] if results["documents"] else []
            if not docs:
                yield _status(
                    "similarity_search",
                    "error",
                    f"No relevant passages found in “{scope_name}”.",
                )
                return
            distances = results["distances"][0] if results["distances"] else []
            scores = [f"{1 - d:.3f}" for d in distances[:3]]
            yield _status(
                "similarity_search",
                "completed",
                f"{len(docs)} passages · scores {', '.join(scores)}",
            )
        except Exception as exc:
            yield _status("similarity_search", "error", str(exc))
            return

        yield _status("context_retrieval", "processing", "Assembling context…")
        context = "\n\n---\n\n".join(docs)
        sources: list[str] = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                src = meta.get("doc_name", "Unknown")
                if src not in sources:
                    sources.append(src)
        yield _status(
            "context_retrieval",
            "completed",
            f"{len(docs)} passages · {len(context):,} chars · {', '.join(sources)}",
        )

        yield _status("llm_response", "processing", "Generating answer…")
        prompt = (
            "You are a precise research assistant. Answer only from the context "
            "taken from the selected document. "
            "If the context is insufficient, say so clearly. "
            "Do not use knowledge outside the provided context.\n\n"
            f"Document: {scope_name}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            client = genai.Client(api_key=api_key)
            full_response = ""
            stream = await client.aio.models.generate_content_stream(
                model=settings.llm_model,
                contents=prompt,
            )
            async for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield {
                        "step": "llm_response",
                        "status": "streaming",
                        "token": chunk.text,
                        "timestamp": time.time(),
                    }

            yield _status(
                "llm_response",
                "completed",
                f"Response ready · {len(full_response):,} characters",
            )
            yield {
                "step": "complete",
                "status": "completed",
                "details": "Query complete",
                "sources": sources,
                "timestamp": time.time(),
            }
        except Exception as exc:
            yield _status("llm_response", "error", str(exc))
            return


def _status(step: str, status: str, details: str) -> dict[str, Any]:
    logger.info("[%s] %s — %s", step.upper(), status.upper(), details)
    return {
        "step": step,
        "status": status,
        "details": details,
        "timestamp": time.time(),
    }
