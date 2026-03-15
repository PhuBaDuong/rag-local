"""Unit tests for embedding functionality."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.embedding.ollama import OllamaEmbedder
from src.utils.exceptions import EmbeddingError


class TestEmbedding(unittest.TestCase):
    """Test suite for embedding functions."""
    
    @patch('src.models.embedding.ollama.requests.post')
    def test_embed_text_success(self, mock_post):
        """Test successful embedding."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder()
        result = embedder.embed("test text")
        
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once()
    
    def test_embed_empty_text(self):
        """Test embedding with empty text."""
        embedder = OllamaEmbedder()
        result = embedder.embed("")
        self.assertEqual(result, [])
    
    def test_get_model_info(self):
        """Test getting model info."""
        embedder = OllamaEmbedder()
        info = embedder.get_model_info()
        
        self.assertIn("provider", info)
        self.assertEqual(info["provider"], "ollama")
        self.assertIn("model", info)


if __name__ == "__main__":
    unittest.main()
