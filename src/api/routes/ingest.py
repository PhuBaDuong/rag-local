"""Ingest endpoints — upload files or point to a local directory."""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import (
    FileDetail,
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileResponse,
)
from src.config import CHUNKING_STRATEGY, PARENT_SPLIT_METHOD
from src.ingestion.ingestion import ingest_directory, ingest_file
from src.logger_config import get_logger

logger = get_logger("api.ingest")

router = APIRouter(tags=["ingest"])

VALID_STRATEGIES = {"fixed", "parent_child"}
VALID_SPLIT_METHODS = {"fixed_size", "title", "tag"}


@router.post("/ingest/file", response_model=IngestFileResponse)
def ingest_upload(
    file: UploadFile = File(...),
    strategy: str = Form(default=CHUNKING_STRATEGY),
    split_method: str = Form(default=PARENT_SPLIT_METHOD),
) -> IngestFileResponse:
    """Upload a file and ingest it into the vector database."""
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=422, detail=f"strategy must be one of {VALID_STRATEGIES}")
    if split_method not in VALID_SPLIT_METHODS:
        raise HTTPException(status_code=422, detail=f"split_method must be one of {VALID_SPLIT_METHODS}")

    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a filename")

    tmp_dir = tempfile.mkdtemp(prefix="rag_ingest_")
    tmp_path = Path(tmp_dir) / file.filename

    try:
        # Save uploaded file to temp location
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"API ingest file: {file.filename} (strategy={strategy}, split_method={split_method})")
        chunks_created = ingest_file(tmp_path, strategy=strategy, split_method=split_method)

        return IngestFileResponse(
            filename=file.filename,
            chunks_created=chunks_created,
            strategy=strategy,
            split_method=split_method,
        )
    except Exception as e:
        logger.error(f"Ingest file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/ingest/directory", response_model=IngestDirectoryResponse)
def ingest_dir(body: IngestDirectoryRequest) -> IngestDirectoryResponse:
    """Ingest all supported files from a local directory."""
    if body.strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=422, detail=f"strategy must be one of {VALID_STRATEGIES}")
    if body.split_method not in VALID_SPLIT_METHODS:
        raise HTTPException(status_code=422, detail=f"split_method must be one of {VALID_SPLIT_METHODS}")

    dir_path = Path(body.path)
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {body.path}")

    logger.info(f"API ingest directory: {body.path} (recursive={body.recursive}, strategy={body.strategy})")

    try:
        results = ingest_directory(
            dir_path,
            recursive=body.recursive,
            strategy=body.strategy,
            split_method=body.split_method,
        )
    except Exception as e:
        logger.error(f"Ingest directory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    details = [
        FileDetail(file=str(fp), chunks=count)
        for fp, count in results.items()
        if count > 0
    ]

    return IngestDirectoryResponse(
        files_processed=len(details),
        total_chunks=sum(d.chunks for d in details),
        strategy=body.strategy,
        split_method=body.split_method,
        details=details,
    )
