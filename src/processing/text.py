"""Text processor for plain text and markdown files."""

from pathlib import Path
from typing import List

from src.processing.base import ProcessorBase, ContentType, ProcessedChunk
from src.utils.exceptions import ProcessingError
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.logger_config import get_logger

logger = get_logger("text_processor")


class TextProcessor(ProcessorBase):
    """Processor for plain text and markdown files.
    
    Splits text content into overlapping chunks for embedding.
    """
    
    SUPPORTED_MIME_TYPES = [
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "text/html",
        "text/xml",
        "application/json",
        "application/xml",
    ]
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """Initialize the text processor.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @property
    def supported_mime_types(self) -> List[str]:
        return self.SUPPORTED_MIME_TYPES
    
    @property
    def content_type(self) -> ContentType:
        return ContentType.TEXT
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size  # Fallback if overlap >= chunk_size
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def process(self, file_path: Path) -> List[ProcessedChunk]:
        """Process a text file and return chunks.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of ProcessedChunk objects
        """
        if not file_path.exists():
            raise ProcessingError(f"Text file not found: {file_path}")
        
        logger.info(f"Processing text file: {file_path}")
        
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Text file is empty: {file_path}")
                return []
            
            # Determine MIME type from extension
            suffix = file_path.suffix.lower()
            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".markdown": "text/markdown",
                ".csv": "text/csv",
                ".html": "text/html",
                ".htm": "text/html",
                ".xml": "text/xml",
                ".json": "application/json",
            }
            mime_type = mime_map.get(suffix, "text/plain")
            
            # Chunk the text
            text_chunks = self._chunk_text(content)
            
            # Create ProcessedChunk objects
            chunks = []
            for i, text in enumerate(text_chunks):
                chunks.append(ProcessedChunk(
                    text=text,
                    content_type=ContentType.TEXT,
                    source_file=str(file_path),
                    mime_type=mime_type,
                    chunk_index=i,
                    metadata={
                        "filename": file_path.name,
                        "total_chunks": len(text_chunks),
                        "original_size": len(content),
                    },
                ))
            
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks
            
        except UnicodeDecodeError as e:
            raise ProcessingError(f"Cannot decode text file {file_path}: {e}")
        except Exception as e:
            raise ProcessingError(f"Failed to process text file {file_path}: {e}")
    
    def process_content(self, content: bytes, source_name: str, mime_type: str) -> List[ProcessedChunk]:
        """Process raw text bytes.
        
        Args:
            content: Raw text bytes
            source_name: Name to use as source identifier
            mime_type: MIME type of the content
            
        Returns:
            List of ProcessedChunk objects
        """
        logger.info(f"Processing text content: {source_name}")
        
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")  # Fallback encoding
        
        if not text.strip():
            return []
        
        text_chunks = self._chunk_text(text)
        
        return [
            ProcessedChunk(
                text=chunk,
                content_type=ContentType.TEXT,
                source_file=source_name,
                mime_type=mime_type,
                chunk_index=i,
                metadata={"total_chunks": len(text_chunks)},
            )
            for i, chunk in enumerate(text_chunks)
        ]

