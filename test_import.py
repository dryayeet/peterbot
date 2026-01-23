import traceback

# Test each import
imports = [
    "import os",
    "import json",
    "import faiss",
    "import requests",
    "import numpy as np",
    "from typing import List, Dict, Optional",
    "from sentence_transformers import SentenceTransformer",
    "from src.config import Config",
    "from src.pdf_processor import PDFProcessor",
    "from src.utils import RateLimiter",
]

for imp in imports:
    try:
        exec(imp)
        print(f"OK: {imp}")
    except Exception as e:
        print(f"FAILED: {imp}")
        print(f"  Error: {e}")
        traceback.print_exc()
        break

print("\nNow trying to import the module and see what happens:")
try:
    import src.rag_system
    print(f"Module imported. Contents: {[x for x in dir(src.rag_system) if not x.startswith('_')]}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
