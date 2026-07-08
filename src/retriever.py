from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
import chromadb

# Import configuration and setup helpers
from config import (
    OPENROUTER_API_KEY,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    validate_config
)
from indexer import setup_llamaindex_settings
from matcher import FuzzyNameMatcher
from classifier import QueryClassifier

class FilteredQueryEngine:
    """
    FilteredQueryEngine acts as an intelligent router and gatekeeper.
    It:
    1. Classifies the query using LLM (or fallback) to block comparison queries.
    2. Resolves candidate names using FuzzyNameMatcher to detect typos/shortforms.
    3. Blocks queries that target 0 or >1 candidates.
    4. Routes the query to LlamaIndex with a candidate-specific metadata filter.
    """
    def __init__(self):
        self.matcher = FuzzyNameMatcher()
        self.classifier = QueryClassifier()
        self.is_llm_ready = False
        
        # Try to initialize LlamaIndex Settings
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            try:
                setup_llamaindex_settings()
                self.is_llm_ready = True
            except Exception as e:
                print(f"Warning: Failed to set up LlamaIndex settings: {e}. Using mock models.")
                Settings.embed_model = MockEmbedding(embed_dim=1536)
                Settings.llm = MockLLM()
        else:
            print("Warning: OPENROUTER_API_KEY is not set or is placeholder. LLM retrieval will run in Mock Mode.")
            Settings.embed_model = MockEmbedding(embed_dim=1536)
            Settings.llm = MockLLM()
            
        # Initialize Chroma DB client
        try:
            self.db_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self.chroma_collection = self.db_client.get_or_create_collection(COLLECTION_NAME)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        except Exception as e:
            print(f"Warning: Failed to load Chroma DB: {e}. Retrieval index is unavailable.")
            self.index = None

    def query(self, query_str: str) -> dict:
        """
        Runs the full filtered query pipeline.
        Returns a dict:
        {
            "response": str,
            "candidate_name": str or None,
            "spelling_suggestion": str or None,
            "is_blocked": bool,
            "blocked_reason": str or None
        }
        """
        # Step 1: Query Classification (Single vs Multi-candidate / Comparison)
        classification = self.classifier.classify(query_str)
        if not classification["is_single_candidate"]:
            return {
                "response": "Error: Multi-resume, comparison, ranking, or aggregation queries are not supported. Query blocked.",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": f"Classifier flagged comparison/multi-candidate: {classification['reason']}"
            }
            
        # Step 2: Name Extraction and Fuzzy Matching
        matches = self.matcher.match_query(query_str)
        
        # Block if multiple candidates are detected in the query text
        if len(matches) > 1:
            names_found = [m["canonical_name"] for m in matches]
            return {
                "response": f"Error: Comparison or multi-candidate query detected ({', '.join(names_found)}). Query blocked.",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": f"Fuzzy matcher detected multiple candidates: {names_found}"
            }
            
        # Block if no candidate is detected in the query text
        if len(matches) == 0:
            return {
                "response": "Error: No candidate name was detected in the query. Please specify a candidate's name (e.g. 'what are the skills of Ashok?').",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": "No candidate name found in query"
            }
            
        # Exactly one candidate resolved
        match = matches[0]
        canonical_name = match["canonical_name"]
        
        spelling_suggestion = None
        if match["is_spelling_mistake"]:
            spelling_suggestion = f"Showing results for '{canonical_name}' (corrected from '{match['query_token']}')"
            
        # If API key is not ready, return a mock response with match info (useful for testing/validation)
        if not self.is_llm_ready or not self.index:
            mock_res = f"[Mock Mode] Successfully resolved candidate to '{canonical_name}'. LLM retrieval is disabled."
            if spelling_suggestion:
                mock_res += f"\nAlert: {spelling_suggestion}"
            return {
                "response": mock_res,
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": False,
                "blocked_reason": None
            }
            
        # Step 3: Run retrieval with metadata filters applied
        try:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="candidate_name",
                        value=canonical_name,
                    )
                ]
            )
            
            # Setup LlamaIndex query engine with filter
            query_engine = self.index.as_query_engine(filters=filters)
            response_obj = query_engine.query(query_str)
            
            return {
                "response": str(response_obj),
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": False,
                "blocked_reason": None
            }
        except Exception as e:
            return {
                "response": f"Error executing index query: {e}",
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": True,
                "blocked_reason": f"Retrieval execution failure: {e}"
            }

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    
    # Simple self-test
    engine = FilteredQueryEngine()
    
    test_queries = [
        "What are the skills of Pawan?",
        "Compare Ashok and Pawan's python experience",
        "Who is the best candidate for Java?",
        "Summarize Yasasvi's achievements",
        "What is python?"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Result:", engine.query(q))
