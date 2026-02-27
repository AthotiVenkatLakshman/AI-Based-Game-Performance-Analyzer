import sys
import time

def test_import(module_name, import_stmt):
    print(f"DEBUG: Importing {module_name}...", flush=True)
    start = time.time()
    try:
        exec(import_stmt)
        print(f"✅ Imported {module_name} in {time.time() - start:.2f}s", flush=True)
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}", flush=True)

if __name__ == "__main__":
    test_import("os", "import os")
    test_import("json", "import json")
    test_import("time", "import time")
    test_import("dotenv", "from dotenv import load_dotenv")
    test_import("huggingface_hub", "from huggingface_hub import InferenceClient")
    test_import("langchain_community.vectorstores", "from langchain_community.vectorstores import FAISS")
    test_import("langchain_community.embeddings", "from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings")
    test_import("langchain_huggingface.embeddings", "from langchain_huggingface import HuggingFaceEndpointEmbeddings")
    test_import("langchain_community.document_loaders", "from langchain_community.document_loaders import PyPDFLoader")
    print("\n--- Sequential Import Test Complete ---")
