"""Custom exceptions for RAG system."""


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class EmbeddingError(RAGException):
    """Exception raised during embedding generation."""
    pass


class LLMError(RAGException):
    """Exception raised during LLM interaction."""
    pass


class DatabaseError(RAGException):
    """Exception raised during database operations."""
    pass


class ValidationError(RAGException):
    """Exception raised during input validation."""
    pass


class RetrievalError(RAGException):
    """Exception raised during document retrieval."""
    pass


class ProcessingError(RAGException):
    """Exception raised during content processing."""
    pass


class VisionError(RAGException):
    """Exception raised during vision model operations."""
    pass
