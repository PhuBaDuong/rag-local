#!/usr/bin/env python3
"""
Multi-Modal Document Ingestion Script
Supports text files, images, PDFs, and directories
Run separately from the main RAG query interface
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.ingestion import ingest_file, ingest_directory
from src.config import DOCUMENT_PATH, CHUNKING_STRATEGY, PARENT_SPLIT_METHOD
from src.logger_config import get_logger

logger = get_logger("ingest")


def ingest_single_file(file_path: Path, strategy: str, split_method: str) -> int:
    """Ingest a single file using multi-modal processing."""
    print(f"\n📄 Processing file: {file_path}  (strategy={strategy}, split_method={split_method})")
    logger.info(f"Processing file: {file_path}")

    try:
        num_chunks = ingest_file(file_path, strategy=strategy, split_method=split_method)
        if num_chunks == 0:
            print(f"⚠️  No chunks created from {file_path.name}")
            return 1
        print(f"✅ Ingested {num_chunks} chunks from {file_path.name}")
        return 0
    except Exception as e:
        logger.error(f"Failed to ingest file: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1


def ingest_dir(directory_path: Path, recursive: bool, strategy: str, split_method: str) -> int:
    """Ingest all files from a directory."""
    print(f"\n📁 Processing directory: {directory_path}  (strategy={strategy}, split_method={split_method})")
    logger.info(f"Processing directory: {directory_path}")

    try:
        results = ingest_directory(
            directory_path, recursive=recursive,
            strategy=strategy, split_method=split_method,
        )
        total_files = len([f for f, c in results.items() if c > 0])
        total_chunks = sum(results.values())

        if total_chunks == 0:
            print("⚠️  No chunks created from any files")
            return 1

        print(f"\n✅ Ingested {total_chunks} chunks from {total_files} files")
        for file_path, num_chunks in results.items():
            if num_chunks > 0:
                print(f"   • {Path(file_path).name}: {num_chunks} chunks")
        return 0
    except Exception as e:
        logger.error(f"Failed to ingest directory: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1


def main():
    """Multi-modal document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                                              # Scan data/ folder with defaults
  python ingest.py --strategy parent_child --split-method title  # Parent-child with heading split
  python ingest.py --strategy parent_child --split-method tag    # Parent-child with HTML tag split
  python ingest.py --file image.png                             # Ingest single image
  python ingest.py --file document.pdf --strategy parent_child  # PDF with parent-child
  python ingest.py --directory ./docs                           # Ingest all files in directory
  python ingest.py --directory ./docs --no-recursive            # Non-recursive
        """
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single file to ingest (text, image, PDF)"
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Path to a directory to ingest (all supported files)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories (only with --directory)"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["fixed", "parent_child"],
        default=CHUNKING_STRATEGY,
        help=f"Chunking strategy (default: {CHUNKING_STRATEGY})"
    )
    parser.add_argument(
        "--split-method", "-m",
        type=str,
        choices=["fixed_size", "title", "tag"],
        default=PARENT_SPLIT_METHOD,
        help=f"Parent split method, used with parent_child strategy (default: {PARENT_SPLIT_METHOD})"
    )

    args = parser.parse_args()

    strategy = args.strategy
    split_method = args.split_method

    logger.info("🚀 Starting Multi-Modal Document Ingestion")
    print("📚 RAG Multi-Modal Ingestion Tool")
    print("="*60)

    try:
        # Directory ingestion
        if args.directory:
            dir_path = Path(args.directory)
            if not dir_path.exists():
                print(f"❌ Error: Directory not found: {args.directory}")
                return 1
            if not dir_path.is_dir():
                print(f"❌ Error: Not a directory: {args.directory}")
                return 1
            return ingest_dir(dir_path, recursive=not args.no_recursive, strategy=strategy, split_method=split_method)

        # Single file ingestion
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ Error: File not found: {args.file}")
                return 1
            return ingest_single_file(file_path, strategy=strategy, split_method=split_method)

        # Default: scan data/ folder
        default_path = Path(DOCUMENT_PATH)
        if not default_path.exists():
            print(f"❌ Error: Default path not found: {DOCUMENT_PATH}")
            return 1

        if default_path.is_dir():
            print(f"ℹ️  No file or directory specified, scanning default: {DOCUMENT_PATH}")
            return ingest_dir(default_path, recursive=True, strategy=strategy, split_method=split_method)
        else:
            print(f"ℹ️  No file or directory specified, using default: {DOCUMENT_PATH}")
            return ingest_single_file(default_path, strategy=strategy, split_method=split_method)

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"❌ Error: File not found: {str(e)}")
        return 1
    except PermissionError as e:
        logger.error(f"Permission denied: {str(e)}")
        print(f"❌ Error: Permission denied: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Failed to ingest: {str(e)}")
        print(f"❌ Error: Failed to ingest: {str(e)}")
        return 1
    finally:
        print("="*60 + "\n")


if __name__ == "__main__":
    sys.exit(main())
