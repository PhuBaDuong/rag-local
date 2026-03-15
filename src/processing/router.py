"""Content router for detecting file types and routing to appropriate processors."""

import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Type

from src.processing.base import ProcessorBase, ContentType, ProcessedChunk
from src.utils.exceptions import ProcessingError
from src.logger_config import get_logger

logger = get_logger("router")

# Initialize mimetypes database
mimetypes.init()

# Additional MIME type mappings not in standard library
EXTRA_MIME_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".webp": "image/webp",
}


class ContentRouter:
    """Routes files to appropriate processors based on content type.
    
    The router detects file types using MIME detection and routes
    to registered processors that can handle each type.
    """
    
    def __init__(self):
        """Initialize the content router."""
        self._processors: Dict[str, ProcessorBase] = {}
        self._type_to_processor: Dict[ContentType, ProcessorBase] = {}
    
    def register_processor(self, processor: ProcessorBase) -> None:
        """Register a processor for its supported MIME types.
        
        Args:
            processor: Processor instance to register
        """
        for mime_type in processor.supported_mime_types:
            self._processors[mime_type.lower()] = processor
            logger.debug(f"Registered processor for {mime_type}")
        
        self._type_to_processor[processor.content_type] = processor
        logger.info(f"Registered {processor.content_type.value} processor")
    
    def detect_mime_type(self, file_path: Path) -> str:
        """Detect the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string (e.g., 'image/png', 'text/plain')
        """
        # Check extension-based detection first
        suffix = file_path.suffix.lower()
        if suffix in EXTRA_MIME_TYPES:
            return EXTRA_MIME_TYPES[suffix]
        
        # Use mimetypes library
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if mime_type:
            return mime_type
        
        # Try to detect from file content using filetype library if available
        try:
            import filetype
            kind = filetype.guess(str(file_path))
            if kind:
                return kind.mime
        except ImportError:
            logger.debug("filetype library not available, using extension-based detection only")
        
        # Default to octet-stream for unknown types
        logger.warning(f"Could not detect MIME type for {file_path}, defaulting to application/octet-stream")
        return "application/octet-stream"
    
    def get_content_type(self, mime_type: str) -> ContentType:
        """Map MIME type to ContentType enum.
        
        Args:
            mime_type: MIME type string
            
        Returns:
            ContentType enum value
        """
        mime_lower = mime_type.lower()
        
        if mime_lower.startswith("text/"):
            return ContentType.TEXT
        elif mime_lower.startswith("image/"):
            return ContentType.IMAGE
        elif mime_lower == "application/pdf":
            return ContentType.PDF
        else:
            return ContentType.UNKNOWN
    
    def get_processor(self, mime_type: str) -> Optional[ProcessorBase]:
        """Get the processor for a MIME type.
        
        Args:
            mime_type: MIME type string
            
        Returns:
            Processor instance or None if no processor registered
        """
        return self._processors.get(mime_type.lower())
    
    def can_process(self, file_path: Path) -> bool:
        """Check if a file can be processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if a processor is available for this file type
        """
        mime_type = self.detect_mime_type(file_path)
        return mime_type.lower() in self._processors
    
    def process_file(
        self,
        file_path: Path,
        strategy: str = "fixed",
        split_method: str = "fixed_size",
    ) -> List[ProcessedChunk]:
        """Process a file using the appropriate processor.
        
        Args:
            file_path: Path to the file to process
            strategy: "fixed" for flat chunking, "parent_child" for hierarchical
            split_method: Parent split method — "fixed_size", "title", or "tag"
            
        Returns:
            List of ProcessedChunk objects
            
        Raises:
            ProcessingError: If no processor is available or processing fails
        """
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")
        
        mime_type = self.detect_mime_type(file_path)
        processor = self.get_processor(mime_type)
        
        if processor is None:
            raise ProcessingError(f"No processor registered for MIME type: {mime_type}")
        
        logger.info(f"Processing {file_path} with {processor.content_type.value} processor")
        return processor.process(file_path, strategy=strategy, split_method=split_method)
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        extensions = set()
        for mime_type in self._processors.keys():
            ext = mimetypes.guess_extension(mime_type)
            if ext:
                extensions.add(ext)
        return sorted(extensions)
    
    def get_registered_processors(self) -> Dict[str, str]:
        """Get dictionary of registered MIME types to processor types."""
        return {
            mime: proc.content_type.value 
            for mime, proc in self._processors.items()
        }

