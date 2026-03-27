"""Health check endpoint."""

import requests
from fastapi import APIRouter

from src.api.schemas import HealthResponse, ServiceStatus
from src.config import NEO4J_URI, OLLAMA_BASE_URL
from src.logger_config import get_logger

logger = get_logger("api.health")

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check connectivity to Neo4j and Ollama."""
    neo4j_status = _check_neo4j()
    ollama_status = _check_ollama()

    overall = "healthy" if neo4j_status.status == "ok" and ollama_status.status == "ok" else "degraded"

    return HealthResponse(status=overall, neo4j=neo4j_status, ollama=ollama_status)


def _check_neo4j() -> ServiceStatus:
    try:
        from src.db.database import get_driver
        driver = get_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        return ServiceStatus(status="ok", detail=NEO4J_URI)
    except Exception as e:
        logger.warning(f"Neo4j health check failed: {e}")
        return ServiceStatus(status="unavailable", detail=str(e))


def _check_ollama() -> ServiceStatus:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return ServiceStatus(status="ok", detail=OLLAMA_BASE_URL)
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return ServiceStatus(status="unavailable", detail=str(e))
