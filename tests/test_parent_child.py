"""Unit tests for parent-child chunking."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.processing.text import TextProcessor
from src.processing.base import (
    ProcessedChunk, ContentType, ChunkStrategy, ParentSplitMethod,
)


class TestFixedSizeParentSplitting(unittest.TestCase):
    """Test _split_parents_fixed."""

    def setUp(self):
        self.proc = TextProcessor(
            chunk_size=50, chunk_overlap=10,
            parent_chunk_size=200, parent_chunk_overlap=50,
        )

    def test_empty_text(self):
        self.assertEqual(self.proc._split_parents_fixed(""), [])
        self.assertEqual(self.proc._split_parents_fixed("   "), [])

    def test_small_text_single_parent(self):
        text = "Hello world"
        parents = self.proc._split_parents_fixed(text)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], text)

    def test_large_text_multiple_parents(self):
        text = "x" * 500
        parents = self.proc._split_parents_fixed(text)
        self.assertGreater(len(parents), 1)
        # Each parent <= parent_chunk_size
        for p in parents:
            self.assertLessEqual(len(p), 200)

    def test_overlap_between_parents(self):
        text = "abcdefghij" * 50  # 500 chars
        parents = self.proc._split_parents_fixed(text)
        # With overlap=50 and size=200, step=150
        # Verify overlapping content between consecutive parents
        if len(parents) >= 2:
            end_of_first = parents[0][-50:]
            start_of_second = parents[1][:50]
            self.assertEqual(end_of_first, start_of_second)


class TestTitleSplitting(unittest.TestCase):
    """Test _split_parents_by_title (Markdown headings)."""

    def setUp(self):
        self.proc = TextProcessor(
            chunk_size=50, chunk_overlap=10,
            parent_chunk_size=200, parent_chunk_overlap=50,
        )

    def test_markdown_headings(self):
        text = (
            "# Introduction\n"
            "This is the intro section with some content.\n\n"
            "## Chapter 1\n"
            "Chapter one has important details.\n\n"
            "## Chapter 2\n"
            "Chapter two continues the story.\n"
        )
        parents = self.proc._split_parents_by_title(text)
        self.assertEqual(len(parents), 3)
        self.assertIn("Introduction", parents[0])
        self.assertIn("Chapter 1", parents[1])
        self.assertIn("Chapter 2", parents[2])

    def test_no_headings_falls_back_to_fixed(self):
        text = "Just a plain paragraph " * 20
        parents = self.proc._split_parents_by_title(text)
        # Should fall back to fixed-size splitting
        self.assertGreater(len(parents), 0)

    def test_oversized_section_is_subsplit(self):
        text = (
            "# Small\nshort\n\n"
            "# Big\n" + "x" * 600 + "\n"
        )
        proc = TextProcessor(
            chunk_size=50, chunk_overlap=10,
            parent_chunk_size=100, parent_chunk_overlap=20,
        )
        parents = proc._split_parents_by_title(text)
        # The big section (600 chars) should be sub-split since > 2×100=200
        self.assertGreater(len(parents), 2)

    def test_single_heading_falls_back(self):
        text = "# Only Heading\nSome text here."
        parents = self.proc._split_parents_by_title(text)
        # Only one section → falls back to fixed
        self.assertGreater(len(parents), 0)


class TestTagSplitting(unittest.TestCase):
    """Test _split_parents_by_tag (HTML semantic tags)."""

    def setUp(self):
        self.proc = TextProcessor(
            chunk_size=50, chunk_overlap=10,
            parent_chunk_size=200, parent_chunk_overlap=50,
        )

    def test_html_sections(self):
        html = (
            "<section>First section content here</section>"
            "<section>Second section content here</section>"
            "<section>Third section content here</section>"
        )
        parents = self.proc._split_parents_by_tag(html)
        self.assertEqual(len(parents), 3)
        self.assertIn("First", parents[0])
        self.assertIn("Second", parents[1])
        self.assertIn("Third", parents[2])

    def test_html_headings_as_splits(self):
        html = (
            "<h1>Title</h1><p>Intro paragraph.</p>"
            "<h2>Section A</h2><p>Content A.</p>"
            "<h2>Section B</h2><p>Content B.</p>"
        )
        parents = self.proc._split_parents_by_tag(html)
        self.assertGreaterEqual(len(parents), 3)

    def test_no_semantic_tags_falls_back(self):
        html = "<p>Just a paragraph with no semantic tags</p>" * 10
        parents = self.proc._split_parents_by_tag(html)
        # Single section → falls back to fixed
        self.assertGreater(len(parents), 0)

    def test_div_splits(self):
        html = "<div>Block one</div><div>Block two</div>"
        parents = self.proc._split_parents_by_tag(html)
        self.assertEqual(len(parents), 2)


class TestParentChildChunking(unittest.TestCase):
    """Test the full parent-child chunking pipeline."""

    def setUp(self):
        self.proc = TextProcessor(
            chunk_size=50, chunk_overlap=10,
            parent_chunk_size=200, parent_chunk_overlap=50,
        )

    def test_parent_child_fixed_size(self):
        text = "word " * 100  # 500 chars
        results = self.proc._chunk_text_parent_child(text, ParentSplitMethod.FIXED_SIZE)
        self.assertGreater(len(results), 0)
        # Each result is (child_text, parent_text, parent_index)
        for child_text, parent_text, parent_idx in results:
            self.assertIsInstance(child_text, str)
            self.assertIsInstance(parent_text, str)
            self.assertIsInstance(parent_idx, int)
            self.assertLessEqual(len(child_text), 50)
            self.assertLessEqual(len(parent_text), 200)

    def test_parent_child_title(self):
        text = "# Intro\nSome intro text.\n\n## Body\nBody content here.\n"
        results = self.proc._chunk_text_parent_child(text, ParentSplitMethod.TITLE)
        self.assertGreater(len(results), 0)
        parent_indices = set(r[2] for r in results)
        self.assertGreaterEqual(len(parent_indices), 2)

    def test_parent_child_tag(self):
        html = "<section>Alpha content</section><section>Beta content</section>"
        results = self.proc._chunk_text_parent_child(html, ParentSplitMethod.TAG)
        self.assertGreater(len(results), 0)
        parent_indices = set(r[2] for r in results)
        self.assertGreaterEqual(len(parent_indices), 2)


class TestProcessedChunkParentChild(unittest.TestCase):
    """Test ProcessedChunk with parent-child fields."""

    def test_create_child_chunk(self):
        chunk = ProcessedChunk(
            text="Child text",
            content_type=ContentType.TEXT,
            source_file="test.txt",
            mime_type="text/plain",
            chunk_index=0,
            chunk_strategy=ChunkStrategy.PARENT_CHILD,
            parent_text="Parent text content",
            parent_index=0,
            parent_split_method=ParentSplitMethod.FIXED_SIZE,
        )
        self.assertEqual(chunk.chunk_strategy, ChunkStrategy.PARENT_CHILD)
        self.assertEqual(chunk.parent_text, "Parent text content")
        self.assertEqual(chunk.parent_index, 0)
        self.assertEqual(chunk.parent_split_method, ParentSplitMethod.FIXED_SIZE)

    def test_to_dict_includes_parent_fields(self):
        chunk = ProcessedChunk(
            text="Child",
            content_type=ContentType.TEXT,
            source_file="test.txt",
            mime_type="text/plain",
            chunk_strategy=ChunkStrategy.PARENT_CHILD,
            parent_text="Parent",
            parent_index=1,
            parent_split_method=ParentSplitMethod.TITLE,
        )
        d = chunk.to_dict()
        self.assertEqual(d["chunk_strategy"], "parent_child")
        self.assertEqual(d["parent_text"], "Parent")
        self.assertEqual(d["parent_index"], 1)
        self.assertEqual(d["parent_split_method"], "title")

    def test_to_dict_excludes_none_parent_fields(self):
        chunk = ProcessedChunk(
            text="Fixed chunk",
            content_type=ContentType.TEXT,
            source_file="test.txt",
            mime_type="text/plain",
        )
        d = chunk.to_dict()
        self.assertEqual(d["chunk_strategy"], "fixed")
        self.assertNotIn("parent_text", d)
        self.assertNotIn("parent_index", d)
        self.assertNotIn("parent_split_method", d)

    def test_invalid_parent_index_raises(self):
        with self.assertRaises(ValueError):
            ProcessedChunk(
                text="Bad",
                content_type=ContentType.TEXT,
                source_file="t.txt",
                mime_type="text/plain",
                parent_index=-1,
            )

    def test_backward_compat_default_fixed(self):
        """Existing code creating ProcessedChunk without new fields still works."""
        chunk = ProcessedChunk(
            text="Legacy chunk",
            content_type=ContentType.TEXT,
            source_file="old.txt",
            mime_type="text/plain",
            chunk_index=5,
            metadata={"key": "value"},
        )
        self.assertEqual(chunk.chunk_strategy, ChunkStrategy.FIXED)
        self.assertIsNone(chunk.parent_text)
        self.assertIsNone(chunk.parent_index)
        self.assertIsNone(chunk.parent_split_method)


class TestTextProcessorParentChildProcess(unittest.TestCase):
    """Test TextProcessor.process() with strategy='parent_child'."""

    def setUp(self):
        self.proc = TextProcessor(
            chunk_size=50, chunk_overlap=10,
            parent_chunk_size=200, parent_chunk_overlap=50,
        )

    def test_process_parent_child_fixed_size(self):
        import tempfile
        from pathlib import Path

        text = "Hello world. " * 40  # ~520 chars
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            chunks = self.proc.process(tmp, strategy="parent_child", split_method="fixed_size")
            self.assertGreater(len(chunks), 0)
            for c in chunks:
                self.assertEqual(c.chunk_strategy, ChunkStrategy.PARENT_CHILD)
                self.assertIsNotNone(c.parent_text)
                self.assertIsNotNone(c.parent_index)
                self.assertEqual(c.parent_split_method, ParentSplitMethod.FIXED_SIZE)
        finally:
            tmp.unlink()

    def test_process_parent_child_title(self):
        import tempfile
        from pathlib import Path

        text = "# Title A\nContent A here.\n\n## Title B\nContent B here.\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            chunks = self.proc.process(tmp, strategy="parent_child", split_method="title")
            self.assertGreater(len(chunks), 0)
            self.assertTrue(all(c.chunk_strategy == ChunkStrategy.PARENT_CHILD for c in chunks))
        finally:
            tmp.unlink()

    def test_process_fixed_strategy_unchanged(self):
        import tempfile
        from pathlib import Path

        text = "Some simple text content here."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            chunks = self.proc.process(tmp, strategy="fixed")
            self.assertGreater(len(chunks), 0)
            for c in chunks:
                self.assertEqual(c.chunk_strategy, ChunkStrategy.FIXED)
                self.assertIsNone(c.parent_text)
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()
