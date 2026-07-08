import json
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL
)

class QueryClassifier:
    """
    QueryClassifier uses the configured OpenRouter LLM to classify if a query
    is about a single candidate or if it involves comparisons/aggregations/rankings
    of multiple candidates.
    """
    def __init__(self):
        self.llm = None
        # Only configure LLM if the API key is provided
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            try:
                Settings.llm = OpenAI(
                    model=LLM_MODEL,
                    api_key=OPENROUTER_API_KEY,
                    api_base=OPENROUTER_BASE_URL,
                    temperature=0.0,
                    additional_headers={
                        "HTTP-Referer": "https://github.com/Antigravity/resume-rag-llm",
                        "X-Title": "Resume RAG LLM"
                    }
                )
                self.llm = Settings.llm
            except Exception as e:
                print(f"Warning: Failed to initialize LLM client: {e}. Using fallback classifier.")

    def classify(self, query: str) -> dict:
        """
        Classifies the user query.
        Returns a dictionary:
        {
            "is_single_candidate": bool,
            "reason": str
        }
        """
        # If LLM is not initialized, use the fallback immediately
        if not self.llm:
            return self._fallback_classify(query, "LLM not initialized (missing API key)")

        # Define strict instructions for the classification task
        prompt = f"""You are a query classifier for a Resume Retrieval-Augmented Generation (RAG) system.
The system is built to search and retrieve details for exactly one single candidate at a time. It cannot handle questions that compare, list, rank, aggregate, or filter multiple candidates at once.

Classify the following query:
Query: "{query}"

Determine if the query is strictly about a single candidate (true), or if it asks for comparisons, rankings, listings, or aggregations across multiple candidates (false).

Respond strictly in valid JSON format. Do not write any conversational text or explanation outside the JSON.
Format:
{{
  "is_single_candidate": true or false,
  "reason": "Brief explanation of why the query was classified this way"
}}

JSON response:"""
        
        try:
            response = self.llm.complete(prompt)
            content = response.text.strip()
            
            # Clean up potential markdown formatting code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            return {
                "is_single_candidate": bool(result.get("is_single_candidate", True)),
                "reason": str(result.get("reason", ""))
            }
        except Exception as e:
            return self._fallback_classify(query, f"LLM error: {e}")

    def _fallback_classify(self, query: str, trigger_reason: str) -> dict:
        """Fallback local keyword-based heuristic classifier."""
        lower_query = query.lower()
        
        # List of triggers for multi-candidate / comparison
        comparison_triggers = {
            "compare", "comparison", "list", "rank", "ranking", "shortlist",
            "best", "who has", "which candidate", "difference", "differences",
            "similarities", "match", "between", "vs", "versus", "candidates", "resumes"
        }
        
        # If any trigger is present, flag as multi-candidate
        is_single = not any(trigger in lower_query for trigger in comparison_triggers)
        reason = f"[{trigger_reason}] Fallback heuristic triggered"
        if not is_single:
            reason += f" due to comparison/aggregation keywords detected."
        else:
            reason += "."
            
        return {
            "is_single_candidate": is_single,
            "reason": reason
        }

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    
    classifier = QueryClassifier()
    test_queries = [
        "What is Ashok's experience in Python?",
        "Compare Ashok and Pawan's python experience",
        "Who is the best candidate for Java?",
        "List all candidates with react experience",
        "Summarize Yasasvi's achievements"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Result:", classifier.classify(q))
