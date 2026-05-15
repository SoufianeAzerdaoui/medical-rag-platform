#!/usr/bin/env python3
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.retrieval.pipeline import MedicalRagPipeline

app = FastAPI(title="Medical RAG API")

# Enable CORS for the Web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve PDF documents
PDF_DIR = project_root / "docs"
if PDF_DIR.exists():
    app.mount("/pdf", StaticFiles(directory=str(PDF_DIR)), name="pdf")

# Database setup for history
HISTORY_DB = project_root / "data" / "conversation_history.sqlite"
HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  conv_id TEXT, 
                  role TEXT, 
                  content TEXT, 
                  metadata TEXT,
                  created_at TIMESTAMP,
                  FOREIGN KEY(conv_id) REFERENCES conversations(id))''')
    conn.commit()
    conn.close()

init_db()

# Pipeline instance
pipeline = MedicalRagPipeline()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Medical RAG API is running"}

class ChatRequest(BaseModel):
    query: str
    conversation_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    normalized_query: str

# --- OpenAI Compatibility Models ---
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAICompletionRequest(BaseModel):
    model: Optional[str] = "medical-rag-model"
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str

class OpenAICompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Fetch history from DB
        conn = sqlite3.connect(HISTORY_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT role, content FROM messages WHERE conv_id = ? ORDER BY created_at ASC LIMIT 10", (request.conversation_id,))
        history_rows = c.fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in history_rows]
        
        # 2. Run pipeline with history
        result = pipeline.run(request.query, history=history)
        
        # 3. Save to history
        # Ensure conversation exists
        c.execute("INSERT OR IGNORE INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                  (request.conversation_id, request.query[:50] + "...", datetime.now()))
        
        # Save User Message
        c.execute("INSERT INTO messages (conv_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (request.conversation_id, "user", request.query, datetime.now()))
        
        # Save Assistant Message
        c.execute("INSERT INTO messages (conv_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                  (request.conversation_id, "assistant", result["answer"], json.dumps(result["sources"]), datetime.now()))
        
        conn.commit()
        conn.close()
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            normalized_query=result["normalized_query"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenAI Compatible Endpoint for AnythingLLM ---
@app.post("/v1/chat/completions", response_model=OpenAICompletionResponse)
async def openai_chat(request: OpenAICompletionRequest):
    try:
        # 1. Extract last user message
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break
        
        if not user_query:
            raise HTTPException(status_code=400, detail="No user message found")

        # 2. Run our specialized pipeline
        result = pipeline.run(user_query)
        
        # 3. Format as OpenAI Response
        return OpenAICompletionResponse(
            id=f"chatcmpl-{datetime.now().timestamp()}",
            created=int(datetime.now().timestamp()),
            model=request.model or "medical-rag-pipeline",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=result["answer"]),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        # Use the specialized medical_rag.sqlite for metadata aggregation
        db_path = project_root / "data" / "indexes" / "medical_rag.sqlite"
        if not db_path.exists():
            return {"error": "Metadata database not found"}
            
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # 1. Overview
        c.execute("SELECT COUNT(DISTINCT doc_id) as total_docs, COUNT(*) as total_results FROM metadata_chunks")
        overview = dict(c.fetchone())
        
        # 2. Interpretation status (Abnormal vs Normal)
        c.execute("SELECT interpretation_status, COUNT(*) as count FROM metadata_chunks WHERE interpretation_status IS NOT NULL GROUP BY interpretation_status")
        status_dist = [dict(r) for r in c.fetchall()]
        
        # 3. Top Analytes
        c.execute("SELECT analyte_norm, COUNT(*) as count FROM metadata_chunks WHERE analyte_norm IS NOT NULL GROUP BY analyte_norm ORDER BY count DESC LIMIT 10")
        top_analytes = [dict(r) for r in c.fetchall()]
        
        # 4. Department Distribution
        c.execute("SELECT section_norm, COUNT(*) as count FROM metadata_chunks WHERE section_norm IS NOT NULL GROUP BY section_norm ORDER BY count DESC LIMIT 5")
        top_sections = [dict(r) for r in c.fetchall()]
        
        conn.close()
        
        return {
            "overview": overview,
            "status_distribution": status_dist,
            "top_markers": top_analytes,
            "top_sections": top_sections
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/conversations")
async def get_conversations():
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/conversations/{conv_id}")
async def get_messages(conv_id: str):
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM messages WHERE conv_id = ? ORDER BY created_at ASC", (conv_id,))
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for r in rows:
        msg = dict(r)
        if msg["metadata"]:
            msg["metadata"] = json.loads(msg["metadata"])
        messages.append(msg)
    return messages

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
