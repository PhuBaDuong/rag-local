"""Core RAG pipeline logic."""

from typing import Tuple, List
from src.retrieval.vector_search import retrieve
from src.models.llm.ollama import get_llm
from src.utils.exceptions import LLMError
from src.core.stepback import generate_stepback
from src.core.prompts import ANSWER_CRITIQUE_PROMPT, ANSWER_GENERATION_PROMPT
import json
from src.config import MAX_QUESTION_LENGTH, MIN_QUESTION_LENGTH, MAX_RETRIEVAL_RETRIES, MIN_RETRIEVAL_SCORE, MAX_ANSWER_CRITIQUES
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


def _has_relevant_results(results: List[Tuple[str, float]]) -> bool:
    """Check if retrieval results contain relevant context above the score threshold."""
    return any(score >= MIN_RETRIEVAL_SCORE for _, score in results)


def _retrieve_with_critic(question: str) -> List[Tuple[str, float]]:
    """Retrieve context with iterative step-back fallback.

    Tries the original question first. If no relevant results are found,
    generates a step-back (broader) question and retries, up to
    MAX_RETRIEVAL_RETRIES total attempts.

    Returns:
        Sorted list of (text, score) tuples with the best results found.
    """
    current_question = question
    best_results: List[Tuple[str, float]] = []

    for attempt in range(1, MAX_RETRIEVAL_RETRIES + 1):
        logger.info(f"Retrieval attempt {attempt}/{MAX_RETRIEVAL_RETRIES} — query: {current_question[:80]}...")

        results = retrieve(current_question)

        if _has_relevant_results(results):
            best_results = _merge_results(best_results, results)
            logger.info(f"Found {len(best_results)} relevant chunks on attempt {attempt}")
            return best_results

        # Keep low-score results as fallback
        if results:
            best_results = _merge_results(best_results, results)

        if attempt < MAX_RETRIEVAL_RETRIES:
            logger.info("No relevant context found, generating step-back question...")
            print("No relevant context found, generating step-back question...")
            stepback = generate_stepback(current_question)
            if stepback == current_question:
                logger.info("Step-back returned same question, stopping retries")
                break
            current_question = stepback

    logger.warning(f"Retriever critic exhausted {attempt} attempts")
    return best_results


def _critique_answer(question: str, context: str) -> List[str]:
    """Critique whether the context fully answers the question.

    Returns a list of follow-up questions to retrieve missing information,
    or an empty list if the context is sufficient.
    """
    prompt = f"""{ANSWER_CRITIQUE_PROMPT}

Question: {question}

Provided information:
{context}

Respond ONLY with the JSON object, nothing else."""

    try:
        response = get_llm().generate(prompt).strip()
        # Strip markdown code fences if present
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(response)
        questions = parsed.get("questions", [])
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str) and q.strip()]
        return []
    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Answer critique failed: {e}")
        return []


def _merge_results(
    existing: List[Tuple[str, float]],
    new: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """Merge two result lists, keeping the best score per unique text."""
    seen = {text: score for text, score in existing}
    for text, score in new:
        if text not in seen or score > seen[text]:
            seen[text] = score
    return sorted(seen.items(), key=lambda x: x[1], reverse=True)


def rag_pipeline(question: str) -> str:
    """
    Run the complete RAG pipeline: validate → retrieve → critique → generate.

    The answer critic loop checks whether the retrieved context fully answers
    the question. If not, it uses the LLM to generate follow-up questions,
    retrieves additional context, and retries — up to MAX_ANSWER_CRITIQUES times.

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
        
        # 2. Retriever critic: try original question first, fall back to step-back
        results = _retrieve_with_critic(question)
        
        if not results:
            logger.warning(f"No results found for question: {question}")
            return "No relevant information found in the knowledge base."

        # 3. Answer critic loop
        for critique_round in range(1, MAX_ANSWER_CRITIQUES + 1):
            context = "\n\n".join(
                [f"[Source {i+1} | score={score:.3f}]\n{text}"
                 for i, (text, score) in enumerate(results)]
            )

            logger.info(f"Answer critique round {critique_round}/{MAX_ANSWER_CRITIQUES}")
            print(f"Answer critique round {critique_round}/{MAX_ANSWER_CRITIQUES}...")

            followup_questions = _critique_answer(question, context)

            if not followup_questions:
                logger.info("Answer critic: context is sufficient")
                break

            logger.info(f"Answer critic: {len(followup_questions)} follow-up question(s) — {followup_questions}")
            print(f"Retrieving additional context for {len(followup_questions)} follow-up question(s)...")

            for fq in followup_questions:
                extra = _retrieve_with_critic(fq)
                results = _merge_results(results, extra)

        # 4. Build final context
        context = "\n\n".join(
            [f"[Source {i+1} | score={score:.3f}]\n{text}"
             for i, (text, score) in enumerate(results)]
        )

        logger.info(f"Retrieved {len(results)} chunks, generating answer...")

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
    {ANSWER_GENERATION_PROMPT}

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
