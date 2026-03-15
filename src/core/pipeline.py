"""Core RAG pipeline logic."""

from typing import Tuple, List
from src.retrieval.vector_search import retrieve
from src.models.llm.ollama import get_llm
from src.utils.exceptions import LLMError
from src.core.stepback import generate_stepback
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
        
        # 2. Generate step-back question for broader retrieval
        print("Generating step-back question for improved retrieval...")
        stepback_question = generate_stepback(question)
        logger.debug(f"Step-back question: {stepback_question[:50]}...")
        
        # 3. Retrieve chunks for both original and step-back questions
        original_results = retrieve(question)
        stepback_results = retrieve(stepback_question)
        
        # Merge and deduplicate results, keeping best scores
        seen_texts = {}
        for text, score in original_results + stepback_results:
            if text not in seen_texts or score > seen_texts[text]:
                seen_texts[text] = score
        
        results = sorted(seen_texts.items(), key=lambda x: x[1], reverse=True)
        
        if not results:
            logger.warning(f"No results found for question: {question}")
            return "No relevant information found in the knowledge base."

        # 4. Combine context
        context = "\n\n".join(
            [f"[Source {i+1} | score={score:.3f}]\n{text}"
             for i, (text, score) in enumerate(results)]
        )

        logger.info(f"Retrieved {len(results)} chunks (original={len(original_results)}, step-back={len(stepback_results)}), generating answer...")

        # 5. Generate answer
        answer = generate_answer(context, question)
        logger.info(f"Answer generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return f"Error processing question: {str(e)}"


def generate_answer(context: str, question: str) -> str:
    """Generate an answer from context and question using the LLM."""
    if not context or not context.strip():
        logger.warning("No context provided for answer generation")
        return "No relevant information found to answer your question."

    if not question or not question.strip():
        logger.error("Empty question provided")
        return "Please ask a valid question."

    prompt_text = f"""
    You are a history expert.

    Answer the question using ONLY the provided context. 
    And do not mention the context in the answer. 
    If the question is not answerable based on the context, say "I don't know based on the provided context." 
    Do not try to make up an answer. 
    Do not use any information that is not in the context. 
    Do not mention the context in your answer. 
    Do not say "based on the provided text" or similar phrases. 
    Just provide a direct answer to the question using only the information in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:
        answer = get_llm().generate(prompt_text)
        logger.info("Answer generated successfully")
        return answer
    except LLMError as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Error generating answer: {e}"
