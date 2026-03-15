"""Ollama vision model provider implementation using LLaVA."""

import base64
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.models.vision.base import VisionModelBase
from src.utils.exceptions import VisionError
from src.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL, VISION_TIMEOUT, RETRY_DELAY
from src.logger_config import get_logger

logger = get_logger("vision")

MAX_RETRIES = 3
MAX_IMAGE_SIZE_MB = 20

# Default prompt for image description
DEFAULT_PROMPT = """Describe this image in detail. Include:
1. What objects, people, or elements are visible
2. Any text visible in the image
3. The overall context or purpose of the image
4. If it's a diagram, chart, or technical image, explain its structure and meaning

Be thorough but concise. Focus on information that would be useful for search and retrieval."""


class OllamaVisionModel(VisionModelBase):
    """Ollama vision model provider using LLaVA or similar models."""
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: int = None
    ):
        """Initialize Ollama vision model.

        Args:
            base_url: Ollama base URL (uses config default if not provided)
            model: Vision model name (uses OLLAMA_VISION_MODEL from config)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = model or OLLAMA_VISION_MODEL
        self.timeout = timeout or VISION_TIMEOUT
        self.url = f"{self.base_url}/api/generate"
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        file_size = image_path.stat().st_size
        max_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            raise VisionError(
                f"Image too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Max allowed: {MAX_IMAGE_SIZE_MB}MB"
            )
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _encode_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        max_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise VisionError(
                f"Image too large ({len(image_bytes) / 1024 / 1024:.1f}MB). "
                f"Max allowed: {MAX_IMAGE_SIZE_MB}MB"
            )
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def _generate(self, image_base64: str, prompt: str) -> str:
        """Generate description using the vision model."""
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Vision model request (attempt {attempt + 1}/{MAX_RETRIES})")
                response = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [image_base64],
                        "stream": False,
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                description = result.get("response", "").strip()
                
                if not description:
                    raise VisionError("Empty response from vision model")
                
                logger.debug(f"Generated description ({len(description)} chars)")
                return description
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Vision request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise VisionError("Vision model failed: Max retries exceeded due to timeout") from e
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error to Ollama (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise VisionError(f"Cannot connect to Ollama at {self.base_url}") from e
                    
            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP error from Ollama: {e.response.status_code}"
                logger.error(error_msg)
                raise VisionError(error_msg) from e
                
            except VisionError:
                raise
                
            except Exception as e:
                error_msg = f"Unexpected error during vision processing: {str(e)}"
                logger.error(error_msg)
                raise VisionError(error_msg) from e
    
    def describe(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Generate a text description of an image."""
        if not image_path.exists():
            raise VisionError(f"Image file not found: {image_path}")
        
        logger.info(f"Describing image: {image_path}")
        image_base64 = self._encode_image(image_path)
        return self._generate(image_base64, prompt or DEFAULT_PROMPT)
    
    def describe_bytes(self, image_bytes: bytes, prompt: Optional[str] = None) -> str:
        """Generate a text description from image bytes."""
        if not image_bytes:
            raise VisionError("Empty image bytes provided")
        
        logger.debug("Describing image from bytes")
        image_base64 = self._encode_bytes(image_bytes)
        return self._generate(image_base64, prompt or DEFAULT_PROMPT)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the vision model."""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }


# Global singleton instance
_vision_model: Optional[OllamaVisionModel] = None


def get_vision_model() -> OllamaVisionModel:
    """Get or create the global vision model instance."""
    global _vision_model
    if _vision_model is None:
        _vision_model = OllamaVisionModel()
    return _vision_model


def describe_image(image_path: Path, prompt: Optional[str] = None) -> str:
    """Convenience function for describing an image."""
    return get_vision_model().describe(image_path, prompt)

