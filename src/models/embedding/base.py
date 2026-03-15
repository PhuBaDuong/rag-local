"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class EmbedderBase(ABC):
    """Base class for text embedding providers."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        pass
