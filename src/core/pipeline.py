"""Core RAG pipeline logic."""

from typing import Tuple
from src.retrieval.vector_search import retrieve
from src.llm.ollama import generate_answer
from src.config import MAX_QUESTION_LENGTH, MIN_QUESTION_LENGTH
from src.logger_config import get_logger

logger = get_logger("pipeline")


def validate_question(question: str) -> Tuple[bool, str]:
    """
    Validate user input question.
    
    Args:
        question: User's question text
        
    Returns:
        Tuple: (is_valid, error_message)
    """
    # Check if question is a string
    if not isinstance(question, str):
        return False, "Question must be text"
    
    # Check if question is empty or just whitespace
    if not question or not question.strip():
        return False, "Question cannot be empty. Please ask a valid question."
    
    # Check minimum length
    question_len = len(question.strip())
    if question_len < MIN_QUESTION_LENGTH:
        return False, f"Question must be at least {MIN_QUESTION_LENGTH} characters long."
    
    # Check maximum length
    if question_len > MAX_QUESTION_LENGTH:
        return False, f"Question must be less than {MAX_QUESTION_LENGTH} characters long (you provided {question_len})."
    
    # Check if question contains only valid characters (basic check)
    # Allow alphanumeric, spaces, and common punctuation
    if not all(c.isalnum() or c.isspace() or c in "?!.,;:'-()\"" for c in question):
        return False, "Question contains invalid characters. Please use standard text and punctuation."
    
    return True, ""


def rag_pipeline(question: str) -> str:
    """
    Run the complete RAG pipeline: validate → retrieve → generate.
    
    Args:
        question: User's question
        
    Returns:
        Generated answer string
    """
    try:
        # 1. Validate input
        is_valid, error_msg = validate_question(question)
        if not is_valid:
            logger.warning(f"Invalid question: {error_msg}")
            return error_msg
        
        logger.debug(f"Question validated: {question[:50]}...")
        
        # 2. Retrieve relevant chunks
        logger.debug(f"Retrieving chunks for question: {question}")
        results = retrieve(question)
        
        if not results:
            logger.warning(f"No results found for question: {question}")
            return "No relevant information found in the knowledge base."

        # 3. Combine context
        context = "\n\n".join(
            [f"[Source {i+1} | score={score:.3f}]\n{text}"
             for i, (text, score) in enumerate(results)]
        )

        logger.info(f"Retrieved {len(results)} chunks, generating answer...")

        # 4. Generate answer
        answer = generate_answer(context, question)
        logger.info(f"Answer generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return f"Error processing question: {str(e)}"
