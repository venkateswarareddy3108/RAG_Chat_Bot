# Corpus — Document Intelligence

Professional RAG workspace:

```
PDF → Extract → Clean → Chunk → Batch → BGE-M3 → ChromaDB → Gemini
```

```
RAG/
├── backend/     # FastAPI + BGE-M3 + ChromaDB
└── frontend/    # Express UI + /api proxy
```

| Service  | URL                   |
|----------|-----------------------|
| UI       | http://localhost:3000 |
| API      | http://127.0.0.1:8000 |

## Backend

```powershell
cd backend
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Configure `backend/.env`:

```
GEMINI_API_KEY=your_key
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=BAAI/bge-m3
```

## Frontend

```powershell
cd frontend
npm install
npm start
```

Open **http://localhost:3000**.
