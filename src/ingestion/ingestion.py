"""Document ingestion into vector database."""

import json
from collections import defaultdict
from pathlib import Path
from neo4j.exceptions import Neo4jError
from typing import List, Any, Optional, Dict

from src.models.embedding.ollama import embed_text
from src.config import NEO4J_URI, VECTOR_INDEX_NAME, CHUNKING_STRATEGY, PARENT_SPLIT_METHOD
from src.logger_config import get_logger
from src.processing.base import ProcessedChunk, ChunkStrategy
from src.db.database import get_driver, create_vector_index

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
            metadata: $metadata,
            chunk_strategy: $chunk_strategy
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
        chunk_strategy=chunk.chunk_strategy.value,
    )


def store_parent_child_chunks(
    tx: Any,
    parent_id: int,
    parent_text: str,
    children: List[ProcessedChunk],
    embeddings: List[List[float]],
    split_method: str,
    source_file: str,
    content_type: str,
) -> None:
    """Store a ParentChunk and its child Chunks with HAS_CHILD relationships.

    Creates one ParentChunk node (no embedding) and multiple child Chunk nodes
    (with embeddings), connected by HAS_CHILD relationships.
    """
    # Create parent node (no embedding — not vector-indexed)
    tx.run("""
        CREATE (p:ParentChunk {
            id: $id,
            text: $text,
            source_file: $source_file,
            content_type: $content_type,
            split_method: $split_method
        })
    """,
        id=parent_id,
        text=parent_text,
        source_file=source_file,
        content_type=content_type,
        split_method=split_method,
    )

    # Create child nodes and relationships
    for child, embedding in zip(children, embeddings):
        tx.run("""
            MATCH (p:ParentChunk {id: $parent_id})
            CREATE (c:Chunk {
                id: $child_id,
                text: $text,
                embedding: $embedding,
                content_type: $content_type,
                source_file: $source_file,
                mime_type: $mime_type,
                chunk_index: $chunk_index,
                metadata: $metadata,
                chunk_strategy: 'child'
            })
            CREATE (p)-[:HAS_CHILD]->(c)
        """,
            parent_id=parent_id,
            child_id=child.chunk_index,
            text=child.text,
            embedding=embedding,
            content_type=child.content_type.value,
            source_file=child.source_file,
            mime_type=child.mime_type,
            chunk_index=child.chunk_index,
            metadata=json.dumps(child.metadata),
        )

def ingest_file(
    file_path: Path,
    start_id: int = 0,
    strategy: str = CHUNKING_STRATEGY,
    split_method: str = PARENT_SPLIT_METHOD,
) -> int:
    """Ingest a single file using the appropriate processor.

    Args:
        file_path: Path to the file to ingest
        start_id: Starting chunk ID (for batch ingestion)
        strategy: "fixed" or "parent_child"
        split_method: "fixed_size", "title", or "tag" (used with parent_child)

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
    logger.info(f"Processing file: {file_path} (strategy={strategy}, split_method={split_method})")
    chunks = router.process_file(file_path, strategy=strategy, split_method=split_method)

    if not chunks:
        logger.warning(f"No chunks created from file: {file_path}")
        return 0

    # Route to appropriate storage method
    if strategy == "parent_child" and any(c.chunk_strategy == ChunkStrategy.PARENT_CHILD for c in chunks):
        return _ingest_parent_child_chunks(chunks, start_id, driver)
    else:
        return _ingest_fixed_chunks(chunks, start_id, driver, file_path)


def _ingest_fixed_chunks(chunks, start_id, driver, file_path):
    """Store fixed-size chunks (original behavior)."""
    logger.info(f"Ingesting {len(chunks)} fixed chunks from {file_path.name}...")
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

    logger.info(f"✅ Ingested {len(chunks)} fixed chunks from {file_path.name}")
    return len(chunks)


def _ingest_parent_child_chunks(chunks, start_id, driver):
    """Store parent-child chunks in Neo4j with ParentChunk nodes and HAS_CHILD relationships."""
    # Group children by parent_index
    parent_groups: Dict[int, List[ProcessedChunk]] = defaultdict(list)
    for chunk in chunks:
        parent_idx = chunk.parent_index if chunk.parent_index is not None else 0
        parent_groups[parent_idx].append(chunk)

    total_stored = 0
    with driver.session() as session:
        for parent_idx in sorted(parent_groups.keys()):
            children = parent_groups[parent_idx]
            parent_text = children[0].parent_text or ""
            split_method = children[0].parent_split_method.value if children[0].parent_split_method else "fixed_size"
            source_file = children[0].source_file
            content_type = children[0].content_type.value

            # Generate embeddings for all children
            embeddings = []
            for child in children:
                logger.debug(f"Embedding child chunk {child.chunk_index}")
                embeddings.append(embed_text(child.text))

            parent_id = start_id + parent_idx
            session.execute_write(
                store_parent_child_chunks,
                parent_id, parent_text, children, embeddings,
                split_method, source_file, content_type,
            )
            total_stored += len(children)
            logger.info(f"Stored parent {parent_idx} with {len(children)} children")

    logger.info(f"✅ Ingested {total_stored} child chunks across {len(parent_groups)} parents")
    return total_stored


def ingest_directory(
    directory_path: Path,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    strategy: str = CHUNKING_STRATEGY,
    split_method: str = PARENT_SPLIT_METHOD,
) -> Dict[str, int]:
    """Ingest all supported files from a directory.

    Args:
        directory_path: Path to the directory
        recursive: Whether to search subdirectories
        extensions: List of file extensions to include (None = all supported)
        strategy: "fixed" or "parent_child"
        split_method: "fixed_size", "title", or "tag" (used with parent_child)

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
            chunks = router.process_file(file_path, strategy=strategy, split_method=split_method)
            if not chunks:
                continue

            if strategy == "parent_child" and any(c.chunk_strategy == ChunkStrategy.PARENT_CHILD for c in chunks):
                num = _ingest_parent_child_chunks(chunks, total_chunks, driver)
            else:
                with driver.session() as session:
                    for i, chunk in enumerate(chunks):
                        chunk_id = total_chunks + i
                        embedding = embed_text(chunk.text)
                        session.execute_write(store_processed_chunk, chunk_id, chunk, embedding)
                num = len(chunks)

            results[str(file_path)] = num
            total_chunks += num
            logger.info(f"Ingested {num} chunks from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {str(e)}")
            results[str(file_path)] = 0

    logger.info(f"✅ Directory ingestion complete: {total_chunks} total chunks from {len(results)} files")
    return results
