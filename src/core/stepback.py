"""Step-back prompting strategy for improved retrieval."""

from src.models.llm.ollama import get_llm
from src.utils.exceptions import LLMError
from src.logger_config import get_logger

logger = get_logger("stepback")

STEPBACK_SYSTEM_PROMPT = """You are an expert at reformulating questions. Your task is to take a specific question and rewrite it as a broader, more general step-back question — one that retrieves foundational context useful for answering the original.

The step-back question should be:
- More general, but not so broad it loses relevance
- Focused on background knowledge, principles, or context
- Easier to answer from general knowledge

If the question is already general, conceptual, or broad enough that stepping back would not add useful context, return the original question unchanged.

Examples:
Input: "Could the members of The Police perform lawful arrests?"
Output: "What are the legal powers and occupations of The Police band members?"

Input: "Jan Sindel's was born in what country?"
Output: "What is Jan Sindel's biographical background?"

Input: "When did Napoleon invade Russia?"
Output: "What were Napoleon's major military campaigns and strategic ambitions?"

Input: "What is the boiling point of ethanol at high altitude?"
Output: "How does atmospheric pressure affect the boiling points of liquids?"

Input: "Which React hook should I use to avoid re-renders?"
Output: "What are React's hooks and how do they manage rendering and performance?"

Input: "What are the main causes of inflation?"
Output: "What are the main causes of inflation?"

Input: "How does photosynthesis work?"
Output: "How does photosynthesis work?"

Respond with ONLY the step-back question (or the original question if no step-back is needed), nothing else."""


def generate_stepback(question: str) -> str:
    """Generate a broader step-back question to improve retrieval.

    Uses the LLM to rephrase the user's specific question into a more
    generic form that may retrieve additional relevant context.

    Args:
        question: The original user question.

    Returns:
        A broader step-back question, or the original question on failure.
    """
    prompt = f"""{STEPBACK_SYSTEM_PROMPT}

Input: "{question}"
Output:"""

    try:
        stepback = get_llm().generate(prompt).strip().strip('"')
        logger.info(f"Step-back question: {stepback[:80]}")
        return stepback
    except LLMError as e:
        logger.warning(f"Step-back generation failed: {e}, using original question")
        return question
