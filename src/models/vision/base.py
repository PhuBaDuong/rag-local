"""Abstract base class for vision model providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class VisionModelBase(ABC):
    """Base class for vision model providers.
    
    Vision models are used to generate text descriptions of images,
    enabling the RAG system to understand and search visual content.
    """
    
    @abstractmethod
    def describe(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Generate a text description of an image.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt for the description
            
        Returns:
            Text description of the image
            
        Raises:
            VisionError: If description generation fails
        """
        pass
    
    @abstractmethod
    def describe_bytes(self, image_bytes: bytes, prompt: Optional[str] = None) -> str:
        """Generate a text description from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            prompt: Optional custom prompt for the description
            
        Returns:
            Text description of the image
            
        Raises:
            VisionError: If description generation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the vision model."""
        pass

