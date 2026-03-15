"""Image processor for converting images to searchable text descriptions."""

import mimetypes
from pathlib import Path
from typing import List, Dict, Any

from src.processing.base import ProcessorBase, ContentType, ProcessedChunk
from src.models.vision.ollama import get_vision_model
from src.utils.exceptions import ProcessingError, VisionError
from src.logger_config import get_logger

logger = get_logger("image_processor")


class ImageProcessor(ProcessorBase):
    """Processor for image files.
    
    Uses a vision model to generate text descriptions of images,
    which are then embedded for semantic search.
    """
    
    SUPPORTED_MIME_TYPES = [
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/tiff",
    ]
    
    def __init__(self, vision_model=None):
        """Initialize the image processor.
        
        Args:
            vision_model: Optional vision model instance (uses default if not provided)
        """
        self._vision_model = vision_model
    
    @property
    def vision_model(self):
        """Get the vision model (lazy initialization)."""
        if self._vision_model is None:
            self._vision_model = get_vision_model()
        return self._vision_model
    
    @property
    def supported_mime_types(self) -> List[str]:
        """Return list of MIME types this processor can handle."""
        return self.SUPPORTED_MIME_TYPES
    
    @property
    def content_type(self) -> ContentType:
        """Return the content type this processor handles."""
        return ContentType.IMAGE
    
    def _get_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from an image file."""
        metadata = {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
        }
        
        # Try to get image dimensions using PIL if available
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["format"] = img.format
                metadata["mode"] = img.mode
        except ImportError:
            logger.debug("PIL not available, skipping image dimension extraction")
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
        
        return metadata
    
    def process(self, file_path: Path, strategy: str = "fixed", split_method: str = "fixed_size") -> List[ProcessedChunk]:
        """Process an image file and return chunks.
        
        Args:
            file_path: Path to the image file
            strategy: Chunking strategy (ignored for images — always single chunk)
            split_method: Parent split method (ignored for images)
            
        Returns:
            List containing a single ProcessedChunk with the image description
        """
        if not file_path.exists():
            raise ProcessingError(f"Image file not found: {file_path}")
        
        logger.info(f"Processing image: {file_path}")
        
        try:
            # Get image description from vision model
            description = self.vision_model.describe(file_path)
            
            # Get metadata
            metadata = self._get_image_metadata(file_path)
            metadata["image_path"] = str(file_path.absolute())
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/unknown"
            
            # Create a single chunk for the image
            chunk = ProcessedChunk(
                text=description,
                content_type=ContentType.IMAGE,
                source_file=str(file_path),
                mime_type=mime_type,
                chunk_index=0,
                metadata=metadata,
            )
            
            logger.info(f"Generated description for {file_path.name} ({len(description)} chars)")
            return [chunk]
            
        except VisionError as e:
            raise ProcessingError(f"Failed to process image {file_path}: {e}")
        except Exception as e:
            raise ProcessingError(f"Unexpected error processing image {file_path}: {e}")
    
    def process_content(self, content: bytes, source_name: str, mime_type: str) -> List[ProcessedChunk]:
        """Process raw image bytes.
        
        Args:
            content: Raw image bytes
            source_name: Name to use as source identifier
            mime_type: MIME type of the content
            
        Returns:
            List containing a single ProcessedChunk with the image description
        """
        logger.info(f"Processing image from bytes: {source_name}")
        
        try:
            description = self.vision_model.describe_bytes(content)
            
            chunk = ProcessedChunk(
                text=description,
                content_type=ContentType.IMAGE,
                source_file=source_name,
                mime_type=mime_type,
                chunk_index=0,
                metadata={"file_size": len(content)},
            )
            
            return [chunk]
            
        except VisionError as e:
            raise ProcessingError(f"Failed to process image bytes: {e}")

