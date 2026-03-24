"""Step-back prompting strategy for improved retrieval."""

from src.models.llm.ollama import get_llm
from src.utils.exceptions import LLMError
from src.core.prompts import STEPBACK_PROMPT
from src.logger_config import get_logger

logger = get_logger("stepback")


def generate_stepback(question: str) -> str:
    """Generate a broader step-back question to improve retrieval.

    Uses the LLM to rephrase the user's specific question into a more
    generic form that may retrieve additional relevant context.

    Args:
        question: The original user question.

    Returns:
        A broader step-back question, or the original question on failure.
    """
    prompt = f"""{STEPBACK_PROMPT}

Input: "{question}"
Output:"""

    try:
        stepback = get_llm().generate(prompt).strip().strip('"')
        logger.info(f"Step-back question: {stepback[:80]}")
        return stepback
    except LLMError as e:
        logger.warning(f"Step-back generation failed: {e}, using original question")
        return question
