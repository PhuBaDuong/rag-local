"""Content processing module for multi-modal ingestion."""

from src.processing.base import ProcessorBase, ContentType, ProcessedChunk, ChunkStrategy, ParentSplitMethod
from src.processing.router import ContentRouter
from src.processing.text import TextProcessor
from src.processing.image import ImageProcessor
from src.processing.pdf import PDFProcessor

__all__ = [
    "ProcessorBase",
    "ContentType",
    "ProcessedChunk",
    "ChunkStrategy",
    "ParentSplitMethod",
    "ContentRouter",
    "TextProcessor",
    "ImageProcessor",
    "PDFProcessor",
]


def get_default_router() -> ContentRouter:
    """Get a ContentRouter with all default processors registered."""
    router = ContentRouter()
    router.register_processor(TextProcessor())
    router.register_processor(ImageProcessor())
    router.register_processor(PDFProcessor())
    return router

