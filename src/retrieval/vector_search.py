"""Vector similarity search for document retrieval."""

from neo4j.exceptions import Neo4jError
from typing import List, Tuple
from src.models.embedding.ollama import embed_text
from src.config import VECTOR_INDEX_NAME, SEARCH_TOP_K
from src.logger_config import get_logger
from src.db.database import get_driver

logger = get_logger("vector_search")

def retrieve(question: str, top_k: int = SEARCH_TOP_K) -> List[Tuple[str, float]]:
    """Retrieve relevant chunks using vector search with parent expansion.

    When a matched chunk is a child of a ParentChunk, the parent text and
    all sibling children are returned as a single expanded context entry.
    Fixed-size chunks (no parent) are returned as-is.
    """
    driver = get_driver()
    
    if not question or not question.strip():
        logger.warning("Empty question provided for retrieval")
        return []
    
    try:
        logger.debug(f"Embedding question: {question[:50]}...")
        question_embedding = embed_text(question)
        
        if not question_embedding:
            logger.error("Failed to generate embedding for question")
            return []
        
        logger.debug(f"Searching for top {top_k} similar chunks")
        
        # Step 1: Vector search to find matching Chunk nodes
        vector_query = f"""
        CALL db.index.vector.queryNodes(
            '{VECTOR_INDEX_NAME}',
            $topK,
            $embedding
        )
        YIELD node, score
        RETURN node.text AS text, score, node.chunk_strategy AS chunk_strategy, elementId(node) AS node_id
        """

        with driver.session() as session:
            result = session.run(
                vector_query,
                topK=top_k,
                embedding=question_embedding
            )
            
            matches = []
            child_node_ids = []
            for record in result:
                matches.append({
                    "text": record["text"],
                    "score": record["score"],
                    "chunk_strategy": record["chunk_strategy"],
                    "node_id": record["node_id"],
                })
                if record["chunk_strategy"] == "child":
                    child_node_ids.append(record["node_id"])

            # Step 2: Expand child chunks → parent + siblings
            expanded = {}
            if child_node_ids:
                expand_query = """
                MATCH (parent:ParentChunk)-[:HAS_CHILD]->(child:Chunk)
                WHERE elementId(child) IN $child_ids
                OPTIONAL MATCH (parent)-[:HAS_CHILD]->(sibling:Chunk)
                RETURN parent.text AS parent_text,
                       collect(DISTINCT sibling.text) AS sibling_texts,
                       elementId(child) AS matched_child_id
                """
                expand_result = session.run(expand_query, child_ids=child_node_ids)
                for rec in expand_result:
                    matched_id = rec["matched_child_id"]
                    parent_text = rec["parent_text"] or ""
                    sibling_texts = rec["sibling_texts"] or []
                    # Build expanded context: parent text + all unique sibling texts
                    all_texts = [parent_text] + [s for s in sibling_texts if s != parent_text]
                    expanded[matched_id] = "\n\n".join(t for t in all_texts if t)

            # Step 3: Build final results
            records = []
            seen_expanded_parents = set()
            for m in matches:
                if m["chunk_strategy"] == "child" and m["node_id"] in expanded:
                    expanded_text = expanded[m["node_id"]]
                    # Deduplicate: only include each parent context once
                    parent_key = expanded_text[:200]
                    if parent_key in seen_expanded_parents:
                        continue
                    seen_expanded_parents.add(parent_key)
                    records.append((expanded_text, m["score"]))
                else:
                    records.append((m["text"], m["score"]))

            logger.info(f"Retrieved {len(records)} chunks (top_k={top_k}, expanded={len(seen_expanded_parents)} parents)")
            return records
            
    except Neo4jError as e:
        logger.error(f"Database query failed: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        raise
    