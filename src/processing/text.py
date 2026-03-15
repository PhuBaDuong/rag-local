"""Text processor for plain text and markdown files."""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

from src.processing.base import (
    ProcessorBase, ContentType, ProcessedChunk,
    ChunkStrategy, ParentSplitMethod,
)
from src.utils.exceptions import ProcessingError
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP
from src.logger_config import get_logger

logger = get_logger("text_processor")


class _HTMLSectionParser(HTMLParser):
    """Splits HTML into sections at semantic tag boundaries."""

    SPLIT_TAGS = {"section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "div"}

    def __init__(self):
        super().__init__()
        self.sections: List[str] = []
        self._current: List[str] = []

    def _flush(self):
        text = "".join(self._current).strip()
        if text:
            self.sections.append(text)
        self._current = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SPLIT_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        self._current.append(data)

    def close(self):
        super().close()
        self._flush()


class TextProcessor(ProcessorBase):
    """Processor for plain text and markdown files.
    
    Splits text content into overlapping chunks for embedding.
    """
    
    SUPPORTED_MIME_TYPES = [
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "text/html",
        "text/xml",
        "application/json",
        "application/xml",
    ]
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        parent_chunk_size: int = PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP,
    ):
        """Initialize the text processor.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            parent_chunk_size: Size of parent chunks in characters
            parent_chunk_overlap: Overlap between parent chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
    
    @property
    def supported_mime_types(self) -> List[str]:
        return self.SUPPORTED_MIME_TYPES
    
    @property
    def content_type(self) -> ContentType:
        return ContentType.TEXT
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size  # Fallback if overlap >= chunk_size
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    # ------------------------------------------------------------------
    # Parent splitting methods
    # ------------------------------------------------------------------

    def _split_parents_fixed(self, text: str) -> List[str]:
        """Split text into parent chunks by fixed character size."""
        if not text or not text.strip():
            return []
        parents = []
        step = self.parent_chunk_size - self.parent_chunk_overlap
        if step <= 0:
            step = self.parent_chunk_size
        for i in range(0, len(text), step):
            chunk = text[i:i + self.parent_chunk_size]
            if chunk.strip():
                parents.append(chunk)
        return parents

    def _split_parents_by_title(self, text: str) -> List[str]:
        """Split text at Markdown heading boundaries (# H1, ## H2, etc.).

        If a section exceeds 2× parent_chunk_size it is further split by
        fixed-size to keep parent chunks manageable.  Falls back to
        fixed-size splitting when no headings are found.
        """
        # Split at lines starting with one or more '#'
        sections: List[str] = []
        current: List[str] = []

        for line in text.splitlines(keepends=True):
            if re.match(r"^#{1,6}\s", line):
                # flush previous section
                section_text = "".join(current).strip()
                if section_text:
                    sections.append(section_text)
                current = [line]
            else:
                current.append(line)
        # flush last section
        section_text = "".join(current).strip()
        if section_text:
            sections.append(section_text)

        if len(sections) <= 1:
            # No headings found — fall back to fixed-size
            return self._split_parents_fixed(text)

        # Sub-split oversized sections
        max_size = self.parent_chunk_size * 2
        result: List[str] = []
        for section in sections:
            if len(section) > max_size:
                result.extend(self._split_parents_fixed(section))
            else:
                result.append(section)
        return result

    def _split_parents_by_tag(self, html_text: str) -> List[str]:
        """Split HTML text at semantic tag boundaries.

        Uses stdlib html.parser to split at <section>, <article>,
        <h1>–<h6>, <div>.  Falls back to fixed-size when no semantic
        tags produce multiple sections.
        """
        parser = _HTMLSectionParser()
        parser.feed(html_text)
        parser.close()

        sections = parser.sections
        if len(sections) <= 1:
            # No meaningful tag boundaries — fall back to fixed-size
            return self._split_parents_fixed(html_text if not sections else sections[0])

        # Sub-split oversized sections
        max_size = self.parent_chunk_size * 2
        result: List[str] = []
        for section in sections:
            if len(section) > max_size:
                result.extend(self._split_parents_fixed(section))
            else:
                result.append(section)
        return result

    # ------------------------------------------------------------------
    # Parent-child orchestrator
    # ------------------------------------------------------------------

    def _chunk_text_parent_child(
        self,
        text: str,
        split_method: ParentSplitMethod = ParentSplitMethod.FIXED_SIZE,
    ) -> List[tuple]:
        """Split text into parent chunks then child chunks.

        Returns a list of (child_text, parent_text, parent_index) tuples.
        """
        if split_method == ParentSplitMethod.TITLE:
            parents = self._split_parents_by_title(text)
        elif split_method == ParentSplitMethod.TAG:
            parents = self._split_parents_by_tag(text)
        else:
            parents = self._split_parents_fixed(text)

        results: List[tuple] = []
        for parent_idx, parent_text in enumerate(parents):
            children = self._chunk_text(parent_text)
            for child_text in children:
                results.append((child_text, parent_text, parent_idx))
        return results
    
    def process(
        self,
        file_path: Path,
        strategy: str = "fixed",
        split_method: str = "fixed_size",
    ) -> List[ProcessedChunk]:
        """Process a text file and return chunks.
        
        Args:
            file_path: Path to the text file
            strategy: "fixed" for flat chunking, "parent_child" for hierarchical
            split_method: Parent split method — "fixed_size", "title", or "tag"
            
        Returns:
            List of ProcessedChunk objects
        """
        if not file_path.exists():
            raise ProcessingError(f"Text file not found: {file_path}")
        
        logger.info(f"Processing text file: {file_path}")
        
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Text file is empty: {file_path}")
                return []
            
            # Determine MIME type from extension
            suffix = file_path.suffix.lower()
            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".markdown": "text/markdown",
                ".csv": "text/csv",
                ".html": "text/html",
                ".htm": "text/html",
                ".xml": "text/xml",
                ".json": "application/json",
            }
            mime_type = mime_map.get(suffix, "text/plain")
            
            if strategy == "parent_child":
                return self._process_parent_child(
                    content, str(file_path), mime_type, file_path.name, len(content), split_method
                )
            
            # Default: fixed-size chunking
            text_chunks = self._chunk_text(content)
            
            # Create ProcessedChunk objects
            chunks = []
            for i, text in enumerate(text_chunks):
                chunks.append(ProcessedChunk(
                    text=text,
                    content_type=ContentType.TEXT,
                    source_file=str(file_path),
                    mime_type=mime_type,
                    chunk_index=i,
                    metadata={
                        "filename": file_path.name,
                        "total_chunks": len(text_chunks),
                        "original_size": len(content),
                    },
                    chunk_strategy=ChunkStrategy.FIXED,
                ))
            
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks
            
        except UnicodeDecodeError as e:
            raise ProcessingError(f"Cannot decode text file {file_path}: {e}")
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to process text file {file_path}: {e}")
    
    def process_content(self, content: bytes, source_name: str, mime_type: str) -> List[ProcessedChunk]:
        """Process raw text bytes.
        
        Args:
            content: Raw text bytes
            source_name: Name to use as source identifier
            mime_type: MIME type of the content
            
        Returns:
            List of ProcessedChunk objects
        """
        logger.info(f"Processing text content: {source_name}")
        
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")  # Fallback encoding
        
        if not text.strip():
            return []
        
        text_chunks = self._chunk_text(text)
        
        return [
            ProcessedChunk(
                text=chunk,
                content_type=ContentType.TEXT,
                source_file=source_name,
                mime_type=mime_type,
                chunk_index=i,
                metadata={"total_chunks": len(text_chunks)},
                chunk_strategy=ChunkStrategy.FIXED,
            )
            for i, chunk in enumerate(text_chunks)
        ]

    # ------------------------------------------------------------------
    # Parent-child processing helper
    # ------------------------------------------------------------------

    def _process_parent_child(
        self,
        content: str,
        source_file: str,
        mime_type: str,
        filename: str,
        original_size: int,
        split_method: str,
    ) -> List[ProcessedChunk]:
        """Create parent-child ProcessedChunk objects from text content."""
        method_enum = ParentSplitMethod(split_method)
        pc_tuples = self._chunk_text_parent_child(content, method_enum)

        if not pc_tuples:
            return []

        chunks = []
        for i, (child_text, parent_text, parent_idx) in enumerate(pc_tuples):
            chunks.append(ProcessedChunk(
                text=child_text,
                content_type=ContentType.TEXT,
                source_file=source_file,
                mime_type=mime_type,
                chunk_index=i,
                metadata={
                    "filename": filename,
                    "original_size": original_size,
                },
                chunk_strategy=ChunkStrategy.PARENT_CHILD,
                parent_text=parent_text,
                parent_index=parent_idx,
                parent_split_method=method_enum,
            ))

        logger.info(
            f"Created {len(chunks)} child chunks from "
            f"{len(set(c.parent_index for c in chunks))} parents ({split_method})"
        )
        return chunks

