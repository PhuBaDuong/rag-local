"""Document ingestion into vector database."""

from src.ingestion.ingestion import (
    ingest_file,
    ingest_directory,
    store_processed_chunk,
    store_parent_child_chunks,
)

__all__ = [
    "ingest_file",
    "ingest_directory",
    "store_processed_chunk",
    "store_parent_child_chunks",
]
