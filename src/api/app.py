"""FastAPI application factory with lifespan management."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, ingest, query
from src.db.database import close_driver
from src.logger_config import get_logger

logger = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI app."""
    logger.info("RAG API starting up")
    yield
    logger.info("RAG API shutting down — closing Neo4j driver")
    close_driver()


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API — ingest documents and query them.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in dev; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules under /api prefix
app.include_router(health.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
