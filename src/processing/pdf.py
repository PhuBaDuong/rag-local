"""PDF processor for extracting text and images from PDF files."""

from pathlib import Path
from typing import List, Dict, Any, Optional

from src.processing.base import ProcessorBase, ContentType, ProcessedChunk
from src.vision.ollama import get_vision_model
from src.utils.exceptions import ProcessingError, VisionError
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
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
        extract_images: bool = True,
    ):
        """Initialize the PDF processor.
        
        Args:
            vision_model: Optional vision model for image processing
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract and process embedded images
        """
        self._vision_model = vision_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
    
    def process(self, file_path: Path) -> List[ProcessedChunk]:
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
                text_chunks = self._chunk_text(full_text)
                for text in text_chunks:
                    chunks.append(ProcessedChunk(
                        text=text,
                        content_type=ContentType.PDF,
                        source_file=str(file_path),
                        mime_type="application/pdf",
                        chunk_index=chunk_index,
                        metadata={"type": "text", "filename": file_path.name},
                    ))
                    chunk_index += 1
                logger.info(f"Extracted {len(text_chunks)} text chunks from PDF")
            
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
        except Exception as e:
            raise ProcessingError(f"Failed to process PDF {file_path}: {e}")
    
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

