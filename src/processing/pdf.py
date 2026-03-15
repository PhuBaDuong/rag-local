"""PDF processor for extracting text and images from PDF files."""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.processing.base import (
    ProcessorBase, ContentType, ProcessedChunk,
    ChunkStrategy, ParentSplitMethod,
)
from src.vision.ollama import get_vision_model
from src.utils.exceptions import ProcessingError, VisionError
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP
from src.logger_config import get_logger

logger = get_logger("pdf_processor")


class PDFProcessor(ProcessorBase):
    """Processor for PDF files.
    
    Extracts text content and embedded images from PDFs.
    Images are processed with a vision model to generate descriptions.
    """
    
    SUPPORTED_MIME_TYPES = [
        "application/pdf",
    ]
    
    def __init__(
        self, 
        vision_model=None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        parent_chunk_size: int = PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP,
        extract_images: bool = True,
    ):
        """Initialize the PDF processor.
        
        Args:
            vision_model: Optional vision model for image processing
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            parent_chunk_size: Size of parent chunks
            parent_chunk_overlap: Overlap between parent chunks
            extract_images: Whether to extract and process embedded images
        """
        self._vision_model = vision_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.extract_images = extract_images
    
    @property
    def vision_model(self):
        """Get the vision model (lazy initialization)."""
        if self._vision_model is None:
            self._vision_model = get_vision_model()
        return self._vision_model
    
    @property
    def supported_mime_types(self) -> List[str]:
        return self.SUPPORTED_MIME_TYPES
    
    @property
    def content_type(self) -> ContentType:
        return ContentType.PDF
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks
    
    def _extract_with_pymupdf(self, file_path: Path) -> tuple[str, List[bytes]]:
        """Extract text and images using PyMuPDF."""
        import fitz  # PyMuPDF
        
        text_content = []
        images = []
        
        doc = fitz.open(str(file_path))
        for page_num, page in enumerate(doc):
            # Extract text
            page_text = page.get_text()
            if page_text.strip():
                text_content.append(f"[Page {page_num + 1}]\n{page_text}")
            
            # Extract images if enabled
            if self.extract_images:
                for img_index, img in enumerate(page.get_images()):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha > 3:  # CMYK
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        images.append(pix.tobytes("png"))
                    except Exception as e:
                        logger.warning(f"Could not extract image {img_index} from page {page_num}: {e}")
        
        doc.close()
        return "\n\n".join(text_content), images
    
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

    def _split_parents_by_title_pdf(self, text: str) -> List[str]:
        """Split PDF text at [Page N] markers and heading-like lines.

        Heading-like lines: ALL CAPS lines, short lines (<80 chars) that
        don't end with a period, or lines starting with a number followed
        by a period (e.g. '1. Introduction').
        Falls back to fixed-size if no split points are detected.
        """
        # Pattern matches [Page N] markers and heading-like lines
        heading_re = re.compile(
            r"^(?:"
            r"\[Page\s+\d+\]"          # [Page N] markers
            r"|[A-Z][A-Z\s]{4,}$"       # ALL CAPS lines (5+ chars)
            r"|\d+\.\s+\S"              # Numbered headings like '1. Intro'
            r")",
            re.MULTILINE,
        )

        positions = [m.start() for m in heading_re.finditer(text)]

        if len(positions) <= 1:
            return self._split_parents_fixed(text)

        # Ensure we start from 0
        if positions[0] != 0:
            positions.insert(0, 0)

        sections: List[str] = []
        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        # Sub-split oversized sections
        max_size = self.parent_chunk_size * 2
        result: List[str] = []
        for section in sections:
            if len(section) > max_size:
                result.extend(self._split_parents_fixed(section))
            else:
                result.append(section)
        return result

    def process(
        self,
        file_path: Path,
        strategy: str = "fixed",
        split_method: str = "fixed_size",
    ) -> List[ProcessedChunk]:
        """Process a PDF file and return chunks."""
        if not file_path.exists():
            raise ProcessingError(f"PDF file not found: {file_path}")
        
        logger.info(f"Processing PDF: {file_path}")
        chunks: List[ProcessedChunk] = []
        chunk_index = 0
        
        try:
            # Extract text and images
            full_text, images = self._extract_with_pymupdf(file_path)
            
            # Process text chunks
            if full_text.strip():
                if strategy == "parent_child":
                    text_chunks = self._build_parent_child_chunks(
                        full_text, str(file_path), file_path.name, split_method,
                    )
                    chunks.extend(text_chunks)
                    chunk_index = len(text_chunks)
                else:
                    raw_chunks = self._chunk_text(full_text)
                    for text in raw_chunks:
                        chunks.append(ProcessedChunk(
                            text=text,
                            content_type=ContentType.PDF,
                            source_file=str(file_path),
                            mime_type="application/pdf",
                            chunk_index=chunk_index,
                            metadata={"type": "text", "filename": file_path.name},
                            chunk_strategy=ChunkStrategy.FIXED,
                        ))
                        chunk_index += 1
                    logger.info(f"Extracted {len(raw_chunks)} text chunks from PDF")
            
            # Process images
            if images and self.extract_images:
                logger.info(f"Processing {len(images)} images from PDF")
                for i, img_bytes in enumerate(images):
                    try:
                        description = self.vision_model.describe_bytes(img_bytes)
                        chunks.append(ProcessedChunk(
                            text=f"[PDF Image {i + 1}] {description}",
                            content_type=ContentType.PDF,
                            source_file=str(file_path),
                            mime_type="application/pdf",
                            chunk_index=chunk_index,
                            metadata={"type": "image", "image_index": i, "filename": file_path.name},
                        ))
                        chunk_index += 1
                    except VisionError as e:
                        logger.warning(f"Could not process PDF image {i}: {e}")
            
            logger.info(f"Processed PDF: {len(chunks)} total chunks")
            return chunks
            
        except ImportError:
            raise ProcessingError("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to process PDF {file_path}: {e}")

    def _build_parent_child_chunks(
        self,
        full_text: str,
        source_file: str,
        filename: str,
        split_method: str,
    ) -> List[ProcessedChunk]:
        """Build parent-child ProcessedChunks from PDF text."""
        if split_method == "title":
            parents = self._split_parents_by_title_pdf(full_text)
        elif split_method == "tag":
            # HTML tag splitting doesn't apply to PDF, fall back to fixed
            parents = self._split_parents_fixed(full_text)
        else:
            parents = self._split_parents_fixed(full_text)

        chunks: List[ProcessedChunk] = []
        for parent_idx, parent_text in enumerate(parents):
            children = self._chunk_text(parent_text)
            for child_text in children:
                chunks.append(ProcessedChunk(
                    text=child_text,
                    content_type=ContentType.PDF,
                    source_file=source_file,
                    mime_type="application/pdf",
                    chunk_index=len(chunks),
                    metadata={"type": "text", "filename": filename},
                    chunk_strategy=ChunkStrategy.PARENT_CHILD,
                    parent_text=parent_text,
                    parent_index=parent_idx,
                    parent_split_method=ParentSplitMethod(split_method) if split_method != "tag" else ParentSplitMethod.FIXED_SIZE,
                ))

        logger.info(
            f"Created {len(chunks)} child chunks from "
            f"{len(parents)} parents ({split_method})"
        )
        return chunks
    
    def process_content(self, content: bytes, source_name: str, mime_type: str) -> List[ProcessedChunk]:
        """Process raw PDF bytes."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            return self.process(tmp_path)
        finally:
            tmp_path.unlink()

