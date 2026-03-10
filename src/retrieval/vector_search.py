"""Vector similarity search for document retrieval."""

from neo4j.exceptions import Neo4jError
from typing import List, Tuple
from src.embedding.ollama import embed_text
from src.config import VECTOR_INDEX_NAME, SEARCH_TOP_K
from src.logger_config import get_logger
from src.retrieval.database import get_driver

logger = get_logger("vector_search")

def retrieve(question: str, top_k: int = SEARCH_TOP_K) -> List[Tuple[str, float]]:
    """Retrieve relevant chunks using vector search with error handling"""
    driver = get_driver()  # Get shared driver instance
    
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
        
        query = f"""
        CALL db.index.vector.queryNodes(
            '{VECTOR_INDEX_NAME}',
            $topK,
            $embedding
        )
        YIELD node, score
        RETURN node.text AS text, score
        """

        with driver.session() as session:
            result = session.run(
                query,
                topK=top_k,
                embedding=question_embedding
            )
            
            records = [(record["text"], record["score"]) for record in result]
            logger.info(f"Retrieved {len(records)} chunks (top_k={top_k})")
            return records
            
    except Neo4jError as e:
        logger.error(f"Database query failed: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        raise
    