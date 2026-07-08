import argparse
import sys
from pathlib import Path
import chromadb
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext
)
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
from openai import OpenAI as OpenAIClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

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

class OpenRouterEmbedding(BaseEmbedding):
    _client: OpenAIClient = PrivateAttr()
    model_name: str
    api_key: str
    api_base: str

    def __init__(self, model_name: str, api_key: str, api_base: str, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, api_base=api_base, **kwargs)
        self._client = OpenAIClient(api_key=api_key, base_url=api_base)

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouterEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        response = self._client.embeddings.create(
            input=[query],
            model=self.model_name
        )
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

def setup_llamaindex_settings():
    """Configures LlamaIndex to use OpenRouter models for LLM and Embeddings."""
    # Configure Qwen LLM via OpenRouter OpenAI-compatible API
    Settings.llm = OpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        # OpenRouter-specific header to identify the app
        additional_headers={"HTTP-Referer": "https://github.com/Antigravity/resume-rag-llm", "X-Title": "Resume RAG LLM"}
    )
    
    # Configure Qwen Embeddings via custom OpenRouter Embedding wrapper
    Settings.embed_model = OpenRouterEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL
    )
    
    # Use our Large-Chunk / Whole-Document strategy
    Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=0)

def build_or_refresh_index(force_rebuild: bool = False):
    """
    Reads resumes from the Resumes directory, parses them,
    and indexes them in the local Chroma DB.
    """
    print("Validating configuration...")
    validate_config()
    
    print("Setting up LLM and Embedding models (OpenRouter)...")
    setup_llamaindex_settings()
    
    print(f"Initializing local Chroma DB client at {CHROMA_DB_PATH}...")
    db_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # If forcing rebuild, delete existing collection
    if force_rebuild:
        try:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            db_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
            
    chroma_collection = db_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Gather files
    resume_files = []
    for ext in ("*.pdf", "*.docx"):
        resume_files.extend(list(RESUMES_DIR.glob(ext)))
        
    print(f"Found {len(resume_files)} resumes in {RESUMES_DIR}.")
    
    # Parse and construct LlamaIndex Documents
    documents = []
    candidate_names = []
    
    for i, filepath in enumerate(resume_files, 1):
        print(f"[{i}/{len(resume_files)}] Parsing {filepath.name}...")
        try:
            candidate_name, text_content = parse_resume(filepath)
            
            # Create LlamaIndex Document with metadata
            doc = Document(
                text=text_content,
                metadata={
                    "candidate_name": candidate_name,
                    "source_file": filepath.name,
                },
                # Exclude from embeddings to avoid biasing retrieval based on filename metadata
                excluded_embed_metadata_keys=["source_file"],
                excluded_llm_metadata_keys=["source_file"],
            )
            documents.append(doc)
            candidate_names.append(candidate_name)
        except Exception as e:
            print(f"Failed to parse {filepath.name}: {e}")
            
    if not documents:
        print("No documents were successfully parsed. Exiting.")
        sys.exit(1)
        
    print(f"Indexing {len(documents)} documents into Chroma DB collection '{COLLECTION_NAME}'...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n--- Indexing Complete! ---")
    print(f"Unique candidates indexed: {sorted(list(set(candidate_names)))}")
    print(f"Total documents: {len(documents)}")
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume RAG LLM Indexer")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the Chroma DB collection by deleting the existing one first."
    )
    args = parser.parse_args()
    
    # Run the indexer
    build_or_refresh_index(force_rebuild=args.rebuild)
