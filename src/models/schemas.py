"""Data models and schemas for RAG system."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ContentTypeEnum(Enum):
    """Content type enumeration for stored chunks."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


@dataclass
class SearchResult:
    """A search result containing chunk text and similarity score."""
    text: str
    score: float
    content_type: str = "text"
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate fields after initialization."""
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if not isinstance(self.content_type, str):
            raise TypeError("content_type must be a string")


@dataclass
class StoredChunk:
    """A chunk stored in the vector database with multi-modal support."""
    id: int
    text: str
    embedding: List[float]
    content_type: str = "text"
    source_file: str = ""
    mime_type: str = "text/plain"
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate fields after initialization."""
        if not isinstance(self.id, int):
            raise TypeError("id must be an integer")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.embedding, list):
            raise TypeError("embedding must be a list")
        if not isinstance(self.content_type, str):
            raise TypeError("content_type must be a string")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "content_type": self.content_type,
            "source_file": self.source_file,
            "mime_type": self.mime_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class Question:
    """A validated user question."""
    text: str
    min_length: int = 3
    max_length: int = 1000
    
    def __post_init__(self):
        """Validate question after initialization."""
        if not isinstance(self.text, str):
            raise TypeError("Question text must be a string")
        
        text_stripped = self.text.strip()
        
        if not text_stripped:
            raise ValueError("Question cannot be empty")
        
        if len(text_stripped) < self.min_length:
            raise ValueError(f"Question must be at least {self.min_length} characters long")
        
        if len(text_stripped) > self.max_length:
            raise ValueError(f"Question must be less than {self.max_length} characters long")
        
        if not all(c.isalnum() or c.isspace() or c in "?!.,;:'-()\"" for c in self.text):
            raise ValueError("Question contains invalid characters")
    
    @property
    def value(self) -> str:
        """Get the validated question text."""
        return self.text.strip()


@dataclass
class Answer:
    """A RAG system answer with context."""
    text: str
    sources: List[SearchResult]
    question: str
    
    def __post_init__(self):
        """Validate answer after initialization."""
        if not isinstance(self.text, str):
            raise TypeError("Answer text must be a string")
        if not isinstance(self.sources, list):
            raise TypeError("Sources must be a list")
        if not isinstance(self.question, str):
            raise TypeError("Question must be a string")
    
    def to_dict(self) -> dict:
        """Convert answer to dictionary representation."""
        return {
            "text": self.text,
            "question": self.question,
            "sources": [
                {"text": s.text, "score": s.score}
                for s in self.sources
            ]
        }


@dataclass
class EmbeddingResult:
    """Result of text embedding operation."""
    embedding: List[float]
    text: str
    model: str
    
    def __post_init__(self):
        """Validate embedding after initialization."""
        if not isinstance(self.embedding, list):
            raise TypeError("Embedding must be a list of floats")
        if not all(isinstance(x, (int, float)) for x in self.embedding):
            raise TypeError("Embedding values must be numeric")
        if not isinstance(self.text, str):
            raise TypeError("Text must be a string")
        if not isinstance(self.model, str):
            raise TypeError("Model must be a string")


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    uri: str
    user: str
    password: str
    vector_index_name: str
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("Database URI must be a non-empty string")
        if not isinstance(self.user, str):
            raise ValueError("Database user must be a string")
        if not isinstance(self.password, str):
            raise ValueError("Database password must be a string")
        if not isinstance(self.vector_index_name, str) or not self.vector_index_name:
            raise ValueError("Vector index name must be a non-empty string")
