"""
Database connection management for RAG system
Handles driver lifecycle and cleanup
"""

import atexit
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, VECTOR_INDEX_NAME
from src.logger_config import get_logger

logger = get_logger("database")

# Global driver instance
_driver = None

def get_driver():
    """Get or create the global Neo4j driver instance"""
    global _driver
    
    if _driver is None:
        logger.info(f"Creating Neo4j driver connection to {NEO4J_URI}")
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.debug("Driver created successfully")
    
    return _driver

def close_driver() -> None:
    """Close the global driver instance"""
    global _driver
    
    if _driver is not None:
        try:
            logger.debug("Closing Neo4j driver...")
            _driver.close()
            _driver = None
            logger.debug("Driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing driver: {str(e)}")

def vector_index_exists(index_name: str = VECTOR_INDEX_NAME) -> bool:
    """
    Check if a vector index already exists
    
    Args:
        index_name: Name of the vector index to check
        
    Returns:
        bool: True if index exists, False otherwise
    """
    driver = get_driver()
    
    try:
        logger.debug(f"Checking if vector index exists: {index_name}")
        with driver.session() as session:
            result = session.run(
                "SHOW INDEXES WHERE name = $name",
                name=index_name
            )
            indexes = list(result)
            exists = len(indexes) > 0
            
            if exists:
                logger.info(f"✅ Vector index already exists: {index_name}")
            else:
                logger.debug(f"Vector index not found: {index_name}")
            
            return exists
            
    except Neo4jError as e:
        logger.warning(f"Could not check index: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking index: {str(e)}")
        return False

def create_vector_index(index_name: str = VECTOR_INDEX_NAME) -> bool:
    """
    Create a vector index for embeddings
    
    Args:
        index_name: Name of the vector index to create
        
    Returns:
        bool: True if created successfully, False otherwise
    """
    driver = get_driver()
    
    try:
        # Check if index already exists
        if vector_index_exists(index_name):
            logger.info(f"Vector index already exists, skipping creation")
            return True
        
        logger.info(f"Creating vector index: {index_name}")
        
        with driver.session() as session:
            # Create vector index for chunk embeddings
            # Using cosine similarity for better semantic search
            query = f"""
            CREATE VECTOR INDEX {index_name} 
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            
            session.run(query)
            logger.info(f"✅ Vector index created successfully: {index_name}")
            
            # Give Neo4j a moment to initialize the index
            import time
            time.sleep(1)
            
            return True
            
    except Neo4jError as e:
        if "already exists" in str(e).lower():
            logger.info(f"Vector index already exists: {index_name}")
            return True
        else:
            logger.error(f"Failed to create vector index: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error creating vector index: {str(e)}")
        return False

# Register driver cleanup on program exit
atexit.register(close_driver)
