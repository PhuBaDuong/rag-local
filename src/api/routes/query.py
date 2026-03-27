"""Query endpoint — ask questions to the RAG system."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import QueryRequest, QueryResponse, SourceItem
from src.core.pipeline import rag_pipeline, validate_question
from src.logger_config import get_logger

logger = get_logger("api.query")

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    """Run a question through the RAG pipeline and return the answer."""
    is_valid, error_msg = validate_question(body.question)
    if not is_valid:
        raise HTTPException(status_code=422, detail=error_msg)

    logger.info(f"API query: {body.question[:80]}...")

    try:
        answer = rag_pipeline(body.question)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return QueryResponse(
        answer=answer,
        question=body.question,
        sources=[],
    )
