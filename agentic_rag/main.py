from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag_pipeline.schemas import ChatRequest, ChatResponse
from agentic_rag_pipeline.service import AgenticRAGService, ConfigurationError
from agentic_rag_pipeline.settings import load_settings


settings = load_settings()
service = AgenticRAGService(settings)

app = FastAPI(
    title="ALD-LLaMat Agentic RAG API",
    version="2.0.0",
    description="Agentic ALD backend with Strategic Agent planning, ReWOO-style tool execution, synthesis, and validation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "ALD-LLaMat agentic retrieval service is running.",
        "health_endpoint": "/health",
        "chat_endpoint": "/api/chat",
    }


@app.get("/health")
def health() -> dict:
    return service.health()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return service.chat(request)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agentic pipeline failed: {exc}") from exc
