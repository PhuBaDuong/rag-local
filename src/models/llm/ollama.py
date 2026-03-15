"""Ollama LLM provider implementation."""

import requests
import time
from src.models.llm.base import LLMBase
from src.utils.exceptions import LLMError
from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TEMPERATURE, LLM_TIMEOUT, RETRY_DELAY
from src.logger_config import get_logger

logger = get_logger("llm")

MAX_RETRIES = 2


class OllamaLLM(LLMBase):
    """Ollama LLM provider."""
    
    def __init__(self, base_url: str = None, model: str = None, temperature: float = None, timeout: int = None):
        """
        Initialize Ollama LLM.
        
        Args:
            base_url: Ollama base URL (uses config default if not provided)
            model: Model name (uses config default if not provided)
            temperature: Temperature parameter (uses config default if not provided)
            timeout: Request timeout in seconds (uses config default if not provided)
        """
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = model or OLLAMA_LLM_MODEL
        self.temperature = temperature if temperature is not None else OLLAMA_TEMPERATURE
        self.timeout = timeout or LLM_TIMEOUT
        self.url = f"{self.base_url}/api/generate"
    
    def generate(self, prompt_text: str) -> str:
        """Send a prompt to the LLM and return the response text."""
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"LLM prompt call (attempt {attempt + 1}/{MAX_RETRIES})")
                response = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "prompt": prompt_text,
                        "stream": False,
                        "temperature": self.temperature,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["response"]

            except requests.exceptions.Timeout:
                logger.warning(f"LLM prompt timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise LLMError("LLM request timed out") from None

            except requests.exceptions.RequestException as e:
                raise LLMError(f"LLM request failed: {e}") from e

            except KeyError:
                raise LLMError("Invalid LLM response - missing 'response' key") from None
    
    def get_model_info(self) -> dict:
        """Get information about the LLM."""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout": self.timeout
        }


# Global singleton instance
_llm = None


def get_llm() -> OllamaLLM:
    """Get or create the global LLM instance."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm
