"""Ollama embedding provider implementation."""

import requests
import time
from typing import List
from src.models.embedding.base import EmbedderBase
from src.utils.exceptions import EmbeddingError
from src.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_TIMEOUT, RETRY_DELAY
from src.logger_config import get_logger

logger = get_logger("embedder")

MAX_RETRIES = 3


class OllamaEmbedder(EmbedderBase):
    """Ollama embedding provider."""
    
    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        """
        Initialize Ollama embedder.
        
        Args:
            base_url: Ollama base URL (uses config default if not provided)
            model: Model name (uses config default if not provided)
            timeout: Request timeout in seconds (uses config default if not provided)
        """
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = model or OLLAMA_EMBEDDING_MODEL
        self.timeout = timeout or EMBEDDING_TIMEOUT
        self.url = f"{self.base_url}/api/embeddings"
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for text with error handling and retry logic."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Embedding text (attempt {attempt + 1}/{MAX_RETRIES})")
                response = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                logger.debug(f"Successfully embedded text (dimension: {len(embedding)})")
                return embedding
                
            except requests.exceptions.Timeout:
                logger.warning(f"Embedding request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    error_msg = "Embedding failed: Max retries exceeded due to timeout"
                    logger.error(error_msg)
                    raise EmbeddingError(error_msg)
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error to Ollama (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    error_msg = f"Embedding failed: Cannot connect to Ollama at {self.base_url}"
                    logger.error(error_msg)
                    raise EmbeddingError(error_msg)
                    
            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg)
                
            except KeyError:
                error_msg = "Invalid response from Ollama - missing 'embedding' key"
                logger.error(error_msg)
                raise EmbeddingError(error_msg)
                
            except Exception as e:
                error_msg = f"Unexpected error during embedding: {str(e)}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg)
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout
        }


# Global singleton instance
_embedder = None


def get_embedder() -> OllamaEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder()
    return _embedder


def embed_text(text: str) -> List[float]:
    """Convenience function for embedding text."""
    return get_embedder().embed(text)
