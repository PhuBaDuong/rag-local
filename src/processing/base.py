"""Base classes for content processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path


class ContentType(Enum):
    """Supported content types for ingestion."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


class ChunkStrategy(Enum):
    """Chunking strategy used to create chunks."""
    FIXED = "fixed"
    PARENT_CHILD = "parent_child"


class ParentSplitMethod(Enum):
    """Method used to split text into parent chunks."""
    FIXED_SIZE = "fixed_size"
    TITLE = "title"
    TAG = "tag"


@dataclass
class ProcessedChunk:
    """A processed chunk ready for embedding and storage.
    
    Attributes:
        text: The text content or description (for images)
        content_type: Type of the original content
        source_file: Original source file path
        mime_type: MIME type of the source file
        chunk_index: Index of this chunk within the source
        metadata: Additional metadata (page number, image dimensions, etc.)
        chunk_strategy: Strategy used to create this chunk
        parent_text: Full text of the parent chunk (only set on child chunks)
        parent_index: Index identifying which parent this child belongs to
        parent_split_method: Method used to split into parent chunks
    """
    text: str
    content_type: ContentType
    source_file: str
    mime_type: str
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED
    parent_text: Optional[str] = None
    parent_index: Optional[int] = None
    parent_split_method: Optional[ParentSplitMethod] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.content_type, ContentType):
            raise TypeError("content_type must be a ContentType enum")
        if not isinstance(self.source_file, str):
            raise TypeError("source_file must be a string")
        if not isinstance(self.mime_type, str):
            raise TypeError("mime_type must be a string")
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.chunk_strategy, ChunkStrategy):
            raise TypeError("chunk_strategy must be a ChunkStrategy enum")
        if self.parent_text is not None and not isinstance(self.parent_text, str):
            raise TypeError("parent_text must be a string or None")
        if self.parent_index is not None and (not isinstance(self.parent_index, int) or self.parent_index < 0):
            raise ValueError("parent_index must be a non-negative integer or None")
        if self.parent_split_method is not None and not isinstance(self.parent_split_method, ParentSplitMethod):
            raise TypeError("parent_split_method must be a ParentSplitMethod enum or None")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "text": self.text,
            "content_type": self.content_type.value,
            "source_file": self.source_file,
            "mime_type": self.mime_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "chunk_strategy": self.chunk_strategy.value,
        }
        if self.parent_text is not None:
            result["parent_text"] = self.parent_text
        if self.parent_index is not None:
            result["parent_index"] = self.parent_index
        if self.parent_split_method is not None:
            result["parent_split_method"] = self.parent_split_method.value
        return result


class ProcessorBase(ABC):
    """Abstract base class for content processors.
    
    Each processor handles a specific content type (text, image, PDF, etc.)
    and converts it into ProcessedChunk objects ready for embedding.
    """
    
    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """Return list of MIME types this processor can handle."""
        pass
    
    @property
    @abstractmethod
    def content_type(self) -> ContentType:
        """Return the content type this processor handles."""
        pass
    
    @abstractmethod
    def process(self, file_path: Path, strategy: str = "fixed", split_method: str = "fixed_size") -> List[ProcessedChunk]:
        """Process a file and return chunks ready for embedding.
        
        Args:
            file_path: Path to the file to process
            strategy: "fixed" for flat chunking, "parent_child" for hierarchical
            split_method: Parent split method — "fixed_size", "title", or "tag"
            
        Returns:
            List of ProcessedChunk objects
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def process_content(self, content: bytes, source_name: str, mime_type: str) -> List[ProcessedChunk]:
        """Process raw content bytes.
        
        Args:
            content: Raw bytes of the content
            source_name: Name to use as source identifier
            mime_type: MIME type of the content
            
        Returns:
            List of ProcessedChunk objects
        """
        pass
    
    def can_process(self, mime_type: str) -> bool:
        """Check if this processor can handle the given MIME type."""
        return mime_type.lower() in [m.lower() for m in self.supported_mime_types]
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about this processor."""
        return {
            "content_type": self.content_type.value,
            "supported_mime_types": self.supported_mime_types,
        }

