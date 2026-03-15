"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Base class for language model providers."""
    
    @abstractmethod
    def generate(self, prompt_text: str) -> str:
        """
        Send a prompt to the LLM and return the response text.
        
        Args:
            prompt_text: The prompt to send to the LLM
            
        Returns:
            Generated response string
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the LLM."""
        pass
