"""
Configuration module for the AI Support Agent.

This module centralizes all configuration settings for the application.
It loads environment variables from a .env file and defines constants
for file paths, model names, and other parameters. Using a configuration
file like this makes the application more maintainable and easier to
configure without changing the source code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
load_dotenv()

# --- Project Root ---
# Use pathlib to ensure cross-platform compatibility for file paths.
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- LLM Provider ---
# Set our preferred LLM provider. Options: "openai" or "google"
# The default is "google" if not specified in the environment.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Model Names ---
# Centralizing model names makes it easy to update them later.
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# --- File Paths ---
DATA_DIR = PROJECT_ROOT / "sample_data"
DOCS_DIR = DATA_DIR / "docs"
ORDERS_CSV_PATH = DATA_DIR / "orders.csv"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
LOG_FILE_PATH = PROJECT_ROOT / "app.log"

# --- RAG Configuration ---
CHUNK_SIZE = 1000  # The number of characters per chunk
CHUNK_OVERLAP = 150  # The number of characters to overlap between chunks
TOP_K = 3  # The number of relevant chunks to retrieve

# --- Logging ---
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"