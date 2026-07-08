import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESUMES_DIR = BASE_DIR / "Resumes"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# Gemini API Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLM and Embedding Models (new google-genai SDK format — no "models/" prefix)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Chroma DB Collection Name
COLLECTION_NAME = "resumes_collection"

def validate_config():
    """Validates that crucial settings like Gemini API Key are set."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set in the environment or .env file. "
            "Please create a .env file and set GEMINI_API_KEY."
        )
    if not RESUMES_DIR.exists():
        raise FileNotFoundError(
            f"Resumes directory does not exist at {RESUMES_DIR}. "
            "Please create it and add resume files."
        )
