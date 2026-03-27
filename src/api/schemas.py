"""Pydantic request/response models for the RAG API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Query ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the RAG system")


class SourceItem(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: List[SourceItem] = []


# ── Ingest ───────────────────────────────────────────────────
class IngestFileResponse(BaseModel):
    filename: str
    chunks_created: int
    strategy: str
    split_method: str


class IngestDirectoryRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the directory")
    recursive: bool = True
    strategy: str = "fixed"
    split_method: str = "fixed_size"


class FileDetail(BaseModel):
    file: str
    chunks: int


class IngestDirectoryResponse(BaseModel):
    files_processed: int
    total_chunks: int
    strategy: str
    split_method: str
    details: List[FileDetail] = []


# ── Health ───────────────────────────────────────────────────
class ServiceStatus(BaseModel):
    status: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    neo4j: ServiceStatus
    ollama: ServiceStatus


# ── Errors ───────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str
