"""
Configuration management for RAG system
Loads settings from environment variables with sensible defaults
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# ============================================
# NEO4J DATABASE CONFIGURATION
# ============================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ============================================
# OLLAMA (LLM & EMBEDDINGS) CONFIGURATION
# ============================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# ============================================
# VISION MODEL CONFIGURATION
# ============================================
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "120"))  # Vision model timeout (seconds)

# ============================================
# DOCUMENT INGESTION CONFIGURATION
# ============================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "resources/")

# ============================================
# PARENT-CHILD CHUNKING CONFIGURATION
# ============================================
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "fixed")          # "fixed" or "parent_child"
PARENT_SPLIT_METHOD = os.getenv("PARENT_SPLIT_METHOD", "fixed_size")  # "fixed_size", "title", or "tag"

# ============================================
# VECTOR SEARCH CONFIGURATION
# ============================================
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk_embedding_index")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "3"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.8"))
MAX_ANSWER_CRITIQUES = int(os.getenv("MAX_ANSWER_CRITIQUES", "3"))

# ============================================
# LOGGING CONFIGURATION
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/rag_system.log")

# ============================================
# APPLICATION CONFIGURATION
# ============================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ============================================
# NETWORK REQUEST CONFIGURATION
# ============================================
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))      # Embedding API timeout (seconds)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))                 # LLM API timeout (seconds)
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))                   # Delay between retries (seconds)

# ============================================
# INPUT VALIDATION CONFIGURATION
# ============================================
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))  # Max characters
MIN_QUESTION_LENGTH = int(os.getenv("MIN_QUESTION_LENGTH", "3"))     # Min characters
