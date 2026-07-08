from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
import chromadb

from config import (
    GEMINI_API_KEY,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)
from indexer import setup_llamaindex_settings
from matcher import FuzzyNameMatcher
from classifier import QueryClassifier


class FilteredQueryEngine:
    """
    FilteredQueryEngine acts as an intelligent router and gatekeeper.
    1. Classifies the query (single vs multi-candidate) using Gemini LLM or heuristic fallback.
    2. Resolves candidate names using FuzzyNameMatcher (handles typos/shortforms).
    3. Blocks queries targeting 0 or >1 candidates.
    4. Routes single-candidate queries to LlamaIndex with a metadata filter.

    api_key: optional Gemini API key override (e.g. provided via Streamlit sidebar at runtime).
    """
    def __init__(self, api_key: str = None):
        key_to_use = api_key or GEMINI_API_KEY
        self.api_key = key_to_use
        self.is_llm_ready = False

        self.matcher = FuzzyNameMatcher()
        self.classifier = QueryClassifier(api_key=key_to_use)

        # Try to initialize LlamaIndex Settings with the resolved key
        if key_to_use and key_to_use not in ("your_gemini_api_key_here", ""):
            try:
                setup_llamaindex_settings(api_key=key_to_use)
                self.is_llm_ready = True
                print(f"[OK] Gemini API key accepted -- LLM mode enabled.")
            except Exception as e:
                print(f"[WARN] Failed to set up LlamaIndex settings: {e}. Using mock models.")
                Settings.embed_model = MockEmbedding(embed_dim=768)
                Settings.llm = MockLLM()
        else:
            print("[WARN] GEMINI_API_KEY not set -- running in Mock Mode.")
            Settings.embed_model = MockEmbedding(embed_dim=768)
            Settings.llm = MockLLM()

        # Initialize Chroma DB connection
        self.index = None
        try:
            self.db_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self.chroma_collection = self.db_client.get_or_create_collection(COLLECTION_NAME)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        except Exception as e:
            print(f"[WARN] Failed to load Chroma DB: {e}. Retrieval index is unavailable.")

    def query(self, query_str: str) -> dict:
        """
        Runs the full filtered query pipeline.
        Returns a dict with keys:
          response, candidate_name, spelling_suggestion, is_blocked, blocked_reason
        """
        # Step 1: Classification — block comparison / multi-candidate queries
        classification = self.classifier.classify(query_str)
        if not classification["is_single_candidate"]:
            return {
                "response": "❌ Multi-resume, comparison, ranking, or aggregation queries are not supported.",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": f"Classifier flagged: {classification['reason']}"
            }

        # Step 2: Fuzzy name resolution
        matches = self.matcher.match_query(query_str)

        if len(matches) > 1:
            names_found = [m["canonical_name"] for m in matches]
            return {
                "response": f"❌ Multiple candidates detected ({', '.join(names_found)}). Please ask about one candidate at a time.",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": f"Multiple candidates found: {names_found}"
            }

        if len(matches) == 0:
            return {
                "response": "❌ No candidate name detected. Please mention a candidate's name (e.g. 'What are Ashok's skills?').",
                "candidate_name": None,
                "spelling_suggestion": None,
                "is_blocked": True,
                "blocked_reason": "No candidate name found in query"
            }

        match = matches[0]
        canonical_name = match["canonical_name"]
        spelling_suggestion = None
        if match.get("is_spelling_mistake"):
            spelling_suggestion = f"Showing results for '{canonical_name}' (corrected from '{match['query_token']}')"

        # Step 3: Mock Mode fallback if LLM / index is not available
        if not self.is_llm_ready or self.index is None:
            mock_res = f"[Mock Mode] Candidate resolved to '{canonical_name}'. Add your Gemini API key to enable live answers."
            return {
                "response": mock_res,
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": False,
                "blocked_reason": None
            }

        # Step 4: Live retrieval with metadata filter
        try:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="candidate_name", value=canonical_name)
                ]
            )
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
                "response": f"⚠️ Retrieval error: {e}",
                "candidate_name": canonical_name,
                "spelling_suggestion": spelling_suggestion,
                "is_blocked": True,
                "blocked_reason": f"Retrieval failure: {e}"
            }


if __name__ == "__main__":
    engine = FilteredQueryEngine()
    test_queries = [
        "What are the skills of Pawan?",
        "Compare Ashok and Pawan",
        "Summarize Yasasvi's achievements",
        "What is python?"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Result:", engine.query(q))
