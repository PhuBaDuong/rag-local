"""unit tests for data models and schemas."""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.schemas.schemas import SearchResult, Question, Answer


class TestModels(unittest.TestCase):
    """Test suite for data models."""
    
    def test_search_result_valid(self):
        """Test creating valid SearchResult."""
        result = SearchResult(text="Sample text", score=0.85)
        self.assertEqual(result.text, "Sample text")
        self.assertEqual(result.score, 0.85)
    
    def test_search_result_invalid_score_range(self):
        """Test SearchResult with invalid score."""
        with self.assertRaises(ValueError):
            SearchResult(text="Text", score=1.5)
    
    def test_question_valid(self):
        """Test creating valid Question."""
        q = Question(text="What is this?")
        self.assertEqual(q.value, "What is this?")
    
    def test_question_too_short(self):
        """Test Question that's too short."""
        with self.assertRaises(ValueError):
            Question(text="ab")
    
    def test_answer_valid(self):
        """Test creating valid Answer."""
        sources = [SearchResult(text="Source 1", score=0.9)]
        answer = Answer(text="The answer", sources=sources, question="Question?")
        
        self.assertEqual(answer.text, "The answer")
        self.assertEqual(len(answer.sources), 1)


if __name__ == "__main__":
    unittest.main()
