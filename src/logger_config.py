"""
Logging configuration for RAG system
Sets up structured logging with file and console output
"""

import logging
import os
from src.config import LOG_LEVEL, LOG_FILE, DEBUG_MODE

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# Create logger
logger = logging.getLogger("rag_system")
logger.setLevel(LOG_LEVEL)

# Console handler (always show errors and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO if not DEBUG_MODE else logging.DEBUG)
console_formatter = logging.Formatter(log_format, datefmt=date_format)
console_handler.setFormatter(console_formatter)

# File handler (log everything)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(LOG_LEVEL)
file_formatter = logging.Formatter(log_format, datefmt=date_format)
file_handler.setFormatter(file_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(f"rag_system.{name}")
