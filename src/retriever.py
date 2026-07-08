import time
import logging
from typing import Dict, Any, List, Optional
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
import chromadb

# Import configurations and local services
from config import (
    OPENROUTER_API_KEY,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)
from indexer import setup_llamaindex_settings
from matcher import FuzzyNameMatcher
from classifier import QueryClassifier
# Add app/ directory to sys.path to avoid name collision with app.py file
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "app") not in sys.path:
    sys.path.append(str(ROOT_DIR / "app"))

from llm.openrouter_service import OpenRouterClient


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FilteredQueryEngine")

class FilteredQueryEngine:
    """
    FilteredQueryEngine coordinates the local query routing pipeline:
    1. Local Intent Detection (QueryClassifier)
    2. Candidate Name Extraction (FuzzyNameMatcher)
    3. Metadata Filtering (candidate_name == canonical_name)
    4. Vector Similarity Search
    5. Similarity Threshold filtering
    6. Top-K Selection
    7. Context Builder
    8. OpenRouter completion call (called exactly once)
    
    Resources are cached at initialization to optimize latency.
    """
    def __init__(self, api_key: str = None):
        # 1. Cache API key and client
        self.api_key = api_key or OPENROUTER_API_KEY
        self.is_llm_ready = False
        
        # Initialize local NLP modules
        self.matcher = FuzzyNameMatcher()
        self.classifier = QueryClassifier()
        
        # Initialize or reuse OpenRouter client
        try:
            self.openrouter_client = OpenRouterClient(api_key=self.api_key)
            if self.api_key and self.api_key != "your_openrouter_api_key_here":
                self.is_llm_ready = True
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None

        # 2. Cache LlamaIndex Settings
        try:
            setup_llamaindex_settings(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to configure LlamaIndex global settings: {e}")

        # 3. Cache Chroma DB Connection and Index
        self.db_client = None
        self.chroma_collection = None
        self.vector_store = None
        self.index = None
        
        try:
            logger.info(f"Connecting to persistent Chroma DB at {CHROMA_DB_PATH}...")
            self.db_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self.chroma_collection = self.db_client.get_or_create_collection(COLLECTION_NAME)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            logger.info("Chroma DB VectorStoreIndex loaded successfully and cached.")
        except Exception as e:
            logger.error(f"Failed to load Chroma DB: {e}. Index search is unavailable.")

    def query(self, query_str: str, similarity_threshold: float = 0.4, top_k: int = 3) -> dict:
        """
        Runs the full 10-step RAG pipeline.
        Returns a dict containing:
          - response (str)
          - candidate_name (str or None)
          - spelling_suggestion (str or None)
          - is_blocked (bool)
          - blocked_reason (str or None)
          - debug_info (dict containing logs for timing, prompt, filters, scores)
        """
        t_start = time.time()
        debug_info = {
            "query": query_str,
            "intent": None,
            "candidate_name": None,
            "metadata_filters": None,
            "retrieved_nodes": [],
            "prompt": None,
            "openrouter_response": None,
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 0.0
        }

        # Step 1: Local Intent Detection
        classification = self.classifier.classify(query_str)
        debug_info["intent"] = classification["reason"]
        
        if not classification["is_single_candidate"]:
            debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
            return {
                "response": "❌ Comparison, ranking, listing, and multi-candidate queries are blocked and not supported.",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": classification["reason"],
                "debug_info": debug_info
            }

        # Step 2: Candidate Name Extraction (Fuzzy matching)
        matches = self.matcher.match_query(query_str)

        if len(matches) > 1:
            names_found = [m["canonical_name"] for m in matches]
            debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
            return {
                "response": f"❌ Multiple candidates detected ({', '.join(names_found)}). Please ask about one candidate at a time.",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": f"Multiple candidates found: {names_found}",
                "debug_info": debug_info
            }

        if len(matches) == 0:
            debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
            return {
                "response": "❌ No candidate name detected. Please mention a candidate's name (e.g. 'What are Ashok's skills?').",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": "No candidate name found in query",
                "debug_info": debug_info
            }

        match = matches[0]
        canonical_name = match["canonical_name"]
        debug_info["candidate_name"] = canonical_name
        
        spelling_suggestion = None
        if match.get("is_spelling_mistake"):
            spelling_suggestion = f"Showing results for '{canonical_name}' (corrected from '{match['query_token']}')"

        # Mock Mode Fallback if API key is not configured or index is not loaded
        if not self.is_llm_ready or self.index is None:
            mock_res = f"[Mock Mode] Candidate resolved to '{canonical_name}'. Please configure your OPENROUTER_API_KEY to generate live answers."
            debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
            return {
                "response": mock_res,
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": False,
                "blocked_reason": None,
                "debug_info": debug_info
            }

        # Step 3 & 4: Metadata Filtering and Search
        t_retrieval_start = time.time()
        try:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="candidate_name", value=canonical_name)
                ]
            )
            debug_info["metadata_filters"] = {"candidate_name": canonical_name}
            
            # Cache the retriever for this specific metadata filtered query
            retriever = self.index.as_retriever(
                filters=filters,
                similarity_top_k=top_k * 2  # Retrieve more than top-k to filter by threshold
            )
            
            # Run similarity search
            retrieved_nodes = retriever.retrieve(query_str)
            
            # Step 5 & 6: Similarity Threshold & Top-K Filtering
            filtered_nodes = []
            for node in retrieved_nodes:
                score = getattr(node, "score", 0.0) or 0.0
                node_data = {
                    "text": node.node.get_content(),
                    "score": score,
                    "source": node.node.metadata.get("source_file", "unknown")
                }
                debug_info["retrieved_nodes"].append(node_data)
                
                # Check similarity threshold
                if score >= similarity_threshold:
                    filtered_nodes.append(node)
            
            # Limit to top_k
            filtered_nodes = filtered_nodes[:top_k]
            debug_info["retrieval_time_ms"] = (time.time() - t_retrieval_start) * 1000.0
            
            # Step 7: Context Builder
            if not filtered_nodes:
                # If no chunks exceed the similarity threshold, we return a standard not-found string
                logger.info(f"No retrieved nodes exceeded the similarity threshold of {similarity_threshold}.")
                not_found_res = "The requested information was not found in the retrieved resume."
                debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
                return {
                    "response": not_found_res,
                    "candidate_name": canonical_name,
                    "spelling_suggestion": spelling_suggestion,
                    "is_blocked": False,
                    "blocked_reason": None,
                    "debug_info": debug_info
                }
                
            context_text = "\n\n=== Context Chunk ===\n".join([n.node.get_content() for n in filtered_nodes])
            
            # Step 8: Prompt Builder
            system_prompt = (
                "You are an expert Technical Recruiter.\n"
                "Answer ONLY from the retrieved resume context.\n"
                "Never hallucinate.\n"
                "Never invent information.\n"
                "If information does not exist, reply exactly: \"The requested information was not found in the retrieved resume.\"\n\n"
                "Summaries should include:\n"
                "- Education\n"
                "- Skills\n"
                "- Experience\n"
                "- Projects\n"
                "- Certifications\n"
                "- Technologies"
            )
            
            user_prompt = f"Retrieved Resume Context for candidate '{canonical_name}':\n\n{context_text}\n\nQuestion: {query_str}\nAnswer:"
            debug_info["prompt"] = f"System: {system_prompt}\n\nUser: {user_prompt}"

            # Step 9: Call OpenRouter (Exactly once)
            t_generation_start = time.time()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response_text = self.openrouter_client.get_completion(
                messages=messages,
                temperature=0.1
            )
            
            debug_info["openrouter_response"] = response_text
            debug_info["generation_time_ms"] = (time.time() - t_generation_start) * 1000.0
            debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
            
            return {
                "response": response_text.strip(),
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": False,
                "blocked_reason": None,
                "debug_info": debug_info
            }
            
        except Exception as e:
            logger.error(f"Retrieval or generation failed: {e}")
            debug_info["total_time_ms"] = (time.time() - t_start) * 1000.0
            return {
                "response": f"⚠️ Retrieval error occurred: {e}",
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": True,
                "blocked_reason": f"Pipeline execution failure: {e}",
                "debug_info": debug_info
            }

if __name__ == "__main__":
    engine = FilteredQueryEngine()
    test_queries = [
        "What are the skills of Pawan?",
        "Compare Ashok and Pawan",
        "Summarize Yasasvi's achievements",
        "What is Python?"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Result:", engine.query(q))
