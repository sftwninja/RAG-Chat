"""
Configuration settings for the RAG CLI application
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for RAG CLI application"""

    # Default paths
    DEFAULT_BASE_PATH = "./rag_systems"
    DEFAULT_CONFIG_FILE = "config.json"

    # Model settings
    DEFAULT_LLM_MODEL = "mistral:7b"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Text processing settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_SEARCH_K = 3

    # LLM settings
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9

    # File type mappings for text splitters
    CODE_FILE_EXTENSIONS = {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.go', '.rs', '.php'}

    @classmethod
    def get_text_splitter_config(cls, filename: str = "") -> Dict[str, Any]:
        """Get text splitter configuration based on file type"""
        file_ext = Path(filename).suffix.lower() if filename else ""

        if file_ext in cls.CODE_FILE_EXTENSIONS:
            return {
                'type': 'recursive',
                'chunk_size': cls.DEFAULT_CHUNK_SIZE,
                'chunk_overlap': cls.DEFAULT_CHUNK_OVERLAP,
                'separators': ["\n\n", "\n", " ", ""]
            }
        else:
            return {
                'type': 'character',
                'chunk_size': cls.DEFAULT_CHUNK_SIZE,
                'chunk_overlap': cls.DEFAULT_CHUNK_OVERLAP,
                'separator': "\n"
            }

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            'model': cls.DEFAULT_LLM_MODEL,
            'temperature': cls.DEFAULT_TEMPERATURE,
        }

    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            'model_name': cls.DEFAULT_EMBEDDING_MODEL
        }

    @classmethod
    def get_retriever_config(cls) -> Dict[str, Any]:
        """Get retriever configuration"""
        return {
            'search_kwargs': {'k': cls.DEFAULT_SEARCH_K}
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate that required dependencies are available"""
        try:
            import langchain
            import langchain_community
            import langchain_huggingface
            import langchain_chroma
            import langchain_ollama
            import chromadb
            import sentence_transformers
            import ollama
            import rich
            return True
        except ImportError as e:
            print(f"Missing required dependency: {e}")
            print("Please run: pip install -r requirements.txt\n\n")
            return False