"""Pytest configuration and fixtures for RAG system tests."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_neo4j_driver():
    """Provide a mock Neo4j driver for testing."""
    with patch('src.db.database.GraphDatabase') as mock_db:
        driver = MagicMock()
        mock_db.driver.return_value = driver
        yield driver


@pytest.fixture
def mock_ollama_api():
    """Provide a mock Ollama API for testing."""
    with patch('src.models.embedding.ollama.requests.post') as mock_post:
        yield mock_post


@pytest.fixture
def mock_config():
    """Provide a mock configuration setup."""
    with patch('src.config.OLLAMA_BASE_URL', 'http://localhost:11434'):
        with patch('src.config.NEO4J_URI', 'bolt://localhost:7687'):
            yield {
                'ollama_url': 'http://localhost:11434',
                'neo4j_uri': 'bolt://localhost:7687'
            }
