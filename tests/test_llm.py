"""Unit tests for LLM functionality."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.llm.ollama import OllamaLLM
from src.core.pipeline import generate_answer
from src.utils.exceptions import LLMError


class TestLLM(unittest.TestCase):
    """Test suite for LLM functions."""
    
    @patch('src.models.llm.ollama.requests.post')
    def test_generate_success(self, mock_post):
        """Test successful prompt generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "This is the answer."}
        mock_post.return_value = mock_response
        
        llm = OllamaLLM()
        result = llm.generate("Some prompt text")
        
        self.assertEqual(result, "This is the answer.")
        mock_post.assert_called_once()

    @patch('src.core.pipeline.get_llm')
    def test_generate_answer_success(self, mock_get_llm):
        """Test successful answer generation via convenience function."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Napoleon invaded in 1812."
        mock_get_llm.return_value = mock_llm

        result = generate_answer("Context about Napoleon", "When did Napoleon invade Russia?")
        self.assertEqual(result, "Napoleon invaded in 1812.")
    
    def test_generate_answer_empty_context(self):
        """Test answer generation with empty context."""
        result = generate_answer("", "Question?")
        self.assertIn("No relevant", result)
    
    def test_generate_answer_empty_question(self):
        """Test answer generation with empty question."""
        result = generate_answer("Context", "")
        self.assertIn("valid", result.lower())
    
    def test_get_model_info(self):
        """Test getting model info."""
        llm = OllamaLLM()
        info = llm.get_model_info()
        
        self.assertIn("provider", info)
        self.assertEqual(info["provider"], "ollama")
        self.assertIn("model", info)


if __name__ == "__main__":
    unittest.main()
