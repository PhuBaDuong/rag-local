"""Document ingestion into vector database."""

import json
from pathlib import Path
from neo4j.exceptions import Neo4jError
from typing import List, Any, Optional, Dict

from src.embedding.ollama import embed_text
from src.config import NEO4J_URI, VECTOR_INDEX_NAME
from src.logger_config import get_logger
from src.retrieval.database import get_driver, create_vector_index
from src.processing.base import ProcessedChunk

logger = get_logger("ingestion")


def store_processed_chunk(
    tx: Any,
    chunk_id: int,
    chunk: ProcessedChunk,
    embedding: List[float]
) -> None:
    """Store a ProcessedChunk with full metadata in Neo4j."""
    tx.run("""
        CREATE (c:Chunk {
            id: $id,
            text: $text,
            embedding: $embedding,
            content_type: $content_type,
            source_file: $source_file,
            mime_type: $mime_type,
            chunk_index: $chunk_index,
            metadata: $metadata
        })
    """,
        id=chunk_id,
        text=chunk.text,
        embedding=embedding,
        content_type=chunk.content_type.value,
        source_file=chunk.source_file,
        mime_type=chunk.mime_type,
        chunk_index=chunk.chunk_index,
        metadata=json.dumps(chunk.metadata),
    )

def ingest_file(file_path: Path, start_id: int = 0) -> int:
    """Ingest a single file using the appropriate processor.

    Args:
        file_path: Path to the file to ingest
        start_id: Starting chunk ID (for batch ingestion)

    Returns:
        Number of chunks ingested
    """
    from src.processing import get_default_router

    router = get_default_router()
    driver = get_driver()

    # Verify connection
    try:
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("✅ Database connection verified")
    except Neo4jError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise Exception(f"Cannot connect to Neo4j: {str(e)}")

    # Ensure vector index exists
    logger.info(f"Ensuring vector index exists: {VECTOR_INDEX_NAME}")
    if not create_vector_index(VECTOR_INDEX_NAME):
        raise Exception(f"Cannot create vector index: {VECTOR_INDEX_NAME}")

    # Process the file
    logger.info(f"Processing file: {file_path}")
    chunks = router.process_file(file_path)

    if not chunks:
        logger.warning(f"No chunks created from file: {file_path}")
        return 0

    # Ingest chunks
    logger.info(f"Ingesting {len(chunks)} chunks from {file_path.name}...")
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = start_id + i
                logger.debug(f"Processing chunk {i + 1}/{len(chunks)}")
                embedding = embed_text(chunk.text)
                session.execute_write(store_processed_chunk, chunk_id, chunk, embedding)

                if (i + 1) % 10 == 0:
                    logger.info(f"Ingested {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to ingest chunk {i}: {str(e)}")
                raise

    logger.info(f"✅ Ingested {len(chunks)} chunks from {file_path.name}")
    return len(chunks)


def ingest_directory(
    directory_path: Path,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Ingest all supported files from a directory.

    Args:
        directory_path: Path to the directory
        recursive: Whether to search subdirectories
        extensions: List of file extensions to include (None = all supported)

    Returns:
        Dictionary mapping file paths to number of chunks ingested
    """
    from src.processing import get_default_router

    router = get_default_router()
    driver = get_driver()

    # Verify connection
    try:
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("✅ Database connection verified")
    except Neo4jError as e:
        raise Exception(f"Cannot connect to Neo4j: {str(e)}")

    # Ensure vector index exists
    if not create_vector_index(VECTOR_INDEX_NAME):
        raise Exception(f"Cannot create vector index: {VECTOR_INDEX_NAME}")

    # Find all files
    pattern = "**/*" if recursive else "*"
    all_files = list(directory_path.glob(pattern))

    # Filter by extension if specified
    if extensions:
        ext_set = {e.lower().lstrip('.') for e in extensions}
        all_files = [f for f in all_files if f.suffix.lower().lstrip('.') in ext_set]

    # Filter to only processable files
    processable_files = [f for f in all_files if f.is_file() and router.can_process(f)]

    logger.info(f"Found {len(processable_files)} processable files in {directory_path}")

    results: Dict[str, int] = {}
    total_chunks = 0

    for file_path in processable_files:
        try:
            chunks = router.process_file(file_path)
            if not chunks:
                continue

            with driver.session() as session:
                for i, chunk in enumerate(chunks):
                    chunk_id = total_chunks + i
                    embedding = embed_text(chunk.text)
                    session.execute_write(store_processed_chunk, chunk_id, chunk, embedding)

            results[str(file_path)] = len(chunks)
            total_chunks += len(chunks)
            logger.info(f"Ingested {len(chunks)} chunks from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {str(e)}")
            results[str(file_path)] = 0

    logger.info(f"✅ Directory ingestion complete: {total_chunks} total chunks from {len(results)} files")
    return results
