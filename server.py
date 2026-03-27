#!/usr/bin/env python3
"""
RAG API Server
Start with: python server.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import sys
import os

# Add project root to Python path (same as main.py / ingest.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the RAG API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"🚀 Starting RAG API server on http://{args.host}:{args.port}")
    print(f"📖 Docs available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
