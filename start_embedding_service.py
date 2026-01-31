"""
Helper script to start the embedding service

Usage:
    python start_embedding_service.py
    python start_embedding_service.py --port 8001
"""

import subprocess
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start embedding service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    args = parser.parse_args()
    
    print(f"Starting embedding service on {args.host}:{args.port}")
    print("Press Ctrl+C to stop the service")
    
    try:
        subprocess.run([
            sys.executable,
            "embedding_service.py",
            "--host", args.host,
            "--port", str(args.port)
        ])
    except KeyboardInterrupt:
        print("\nEmbedding service stopped.")

