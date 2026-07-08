import argparse
import sys
import logging
from pathlib import Path
import chromadb
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext
)
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Indexer")

# Import project config and parser
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    RESUMES_DIR,
    COLLECTION_NAME,
    validate_config
)
from parser import parse_resume

def setup_llamaindex_settings(api_key: str = None):
    """
    Configures LlamaIndex to use OpenRouter for LLM and local HuggingFace for Embeddings.
    """
    key_to_use = api_key or OPENROUTER_API_KEY
    if not key_to_use or key_to_use == "your_openrouter_api_key_here":
        # If API key is missing or dummy, we will allow initialization with a dummy key for indexing
        # since local HuggingFaceEmbedding doesn't require an API key to run.
        logger.warning("OPENROUTER_API_KEY is not set or placeholder. Using dummy key for LLM init.")
        key_to_use = "dummy-openrouter-key"

    # Configure OpenRouter via OpenAI-like endpoint in LlamaIndex
    from llama_index.llms.openai import OpenAI
    Settings.llm = OpenAI(
        model=LLM_MODEL,
        api_key=key_to_use,
        api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        additional_headers={"HTTP-Referer": "https://github.com/Antigravity/resume-rag-llm", "X-Title": "Resume RAG LLM"}
    )

    # Configure local HuggingFace Embedding model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    logger.info(f"Initializing HuggingFace embedding model: {EMBEDDING_MODEL}...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    # Set up node parser with large chunks to capture whole document context if possible
    from llama_index.core.node_parser import SentenceSplitter
    Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=0)


def build_or_refresh_index(force_rebuild: bool = True):
    """
    Reads resumes from the Resumes directory, parses them,
    and indexes them in the local Chroma DB.
    Always rebuilds by deleting the collection first to avoid duplicate vectors.
    """
    logger.info("Validating configuration...")
    validate_config()

    logger.info("Setting up LLM and Embedding models (OpenRouter + Local BGE)...")
    setup_llamaindex_settings()

    logger.info(f"Initializing local Chroma DB client at {CHROMA_DB_PATH}...")
    db_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # To avoid duplicate indexing, we delete the collection if it exists
    if force_rebuild:
        try:
            logger.info(f"Deleting existing collection '{COLLECTION_NAME}' to ensure clean index...")
            db_client.delete_collection(COLLECTION_NAME)
        except Exception as e:
            logger.info(f"No existing collection '{COLLECTION_NAME}' to delete: {e}")

    chroma_collection = db_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Gather files
    resume_files = []
    for ext in ("*.pdf", "*.docx"):
        resume_files.extend(list(RESUMES_DIR.glob(ext)))

    logger.info(f"Found {len(resume_files)} resumes in {RESUMES_DIR}.")

    # Parse and construct LlamaIndex Documents
    documents = []
    candidate_names = []

    for i, filepath in enumerate(resume_files, 1):
        logger.info(f"[{i}/{len(resume_files)}] Parsing {filepath.name}...")
        try:
            candidate_name, text_content = parse_resume(filepath)

            doc = Document(
                text=text_content,
                metadata={
                    "candidate_name": candidate_name,
                    "source_file": filepath.name,
                },
                excluded_embed_metadata_keys=["source_file"],
                excluded_llm_metadata_keys=["source_file"],
            )
            documents.append(doc)
            candidate_names.append(candidate_name)
        except Exception as e:
            logger.error(f"  ✗ Failed to parse {filepath.name}: {e}")

    if not documents:
        logger.error("No documents were successfully parsed. Exiting.")
        sys.exit(1)

    logger.info(f"Indexing {len(documents)} documents into Chroma DB collection '{COLLECTION_NAME}'...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    # Verify that the vector collection contains correct amount of nodes
    collection_count = chroma_collection.count()
    logger.info("--- Verification ---")
    logger.info(f"Unique candidates parsed: {sorted(list(set(candidate_names)))}")
    logger.info(f"Total parsed resumes: {len(documents)}")
    logger.info(f"Collection Count (Chroma vectors): {collection_count}")
    
    # Warn or raise if there are duplicates or missing records
    if collection_count < len(documents):
        logger.error(f"Verification FAILED: Collection Count ({collection_count}) < Resume Count ({len(documents)})!")
    else:
        logger.info(f"Verification PASSED: Collection Count ({collection_count}) >= Resume Count ({len(documents)})")

    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume RAG LLM Indexer")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=True,
        help="Force rebuild the Chroma DB collection by deleting the existing one first (default: True)."
    )
    args = parser.parse_args()
    build_or_refresh_index(force_rebuild=args.rebuild)
