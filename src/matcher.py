import re
from pathlib import Path
from rapidfuzz import fuzz

# Import config and helper parser
from config import RESUMES_DIR
from parser import extract_candidate_name

class FuzzyNameMatcher:
    """
    FuzzyNameMatcher resolves potential candidate names in queries to canonical names
    extracted from the resumes. It uses rapidfuzz to perform spelling-insensitive
    matches and handles common spelling variants or shortforms.
    """
    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold
        self.candidates = self._load_candidates()
        self.candidate_terms = self._build_candidate_terms()

    def _load_candidates(self) -> list[str]:
        """Loads canonical candidate names from the Resumes directory."""
        if not RESUMES_DIR.exists():
            return []
        resume_files = []
        for ext in ("*.pdf", "*.docx"):
            resume_files.extend(list(RESUMES_DIR.glob(ext)))
        
        names = set()
        for filepath in resume_files:
            try:
                name = extract_candidate_name(filepath)
                if name:
                    names.add(name)
            except Exception as e:
                print(f"Error extracting name from {filepath.name}: {e}")
                
        return sorted(list(names))

    def _clean_token(self, token: str) -> str:
        """Strips everything except letters and numbers, returns lowercased string."""
        return re.sub(r'[^a-zA-Z0-9]', '', token).lower()

    def _build_candidate_terms(self) -> dict[str, list[str]]:
        """
        Builds a map from canonical candidate name to search term variations.
        These terms are used for fuzzy matching.
        """
        terms_map = {}
        for name in self.candidates:
            terms = [name.lower()]
            # Split the name into tokens and clean them
            tokens = name.split()
            cleaned_tokens = []
            for t in tokens:
                cleaned = self._clean_token(t)
                # Ignore small initials, common numbers, and academic suffixes/prefixes
                if cleaned and len(cleaned) > 1 and cleaned not in {"btech", "csd", "resume", "profile"}:
                    cleaned_tokens.append(cleaned)
            
            terms.extend(cleaned_tokens)
            
            # Combine pairs of cleaned tokens to capture first+last name combos
            if len(cleaned_tokens) >= 2:
                for i in range(len(cleaned_tokens) - 1):
                    terms.append(f"{cleaned_tokens[i]} {cleaned_tokens[i+1]}")
            
            # Add hardcoded common shortforms, spelling variations, and phonetic mappings
            # E.g., 'pawan' -> 'pavan' (pavanteja kamma)
            # 'yasasvi' -> 'yasaswi' (yasaswi kotha)
            # 'trinad' -> 'trinadh' (Trinadh Kumar Reddi)
            # 'krish' -> 'krishna' (Mungara Muddu Krishna yadav)
            # 'abdul' -> 'abduljaha' (parchuru abduljaha)
            # 'shirly' -> 'shirley' (21H51A6740-KANDRU SHIRLEY KATHERINE B.Tech-CSD)
            # 'rani' -> 'ranisri' (ranisri21 gundapaneni)
            for ct in cleaned_tokens:
                if ct == "pavanteja":
                    terms.extend(["pavan", "pawan", "teja"])
                elif ct == "yasaswi":
                    terms.append("yasasvi")
                elif ct == "trinadh":
                    terms.append("trinad")
                elif ct == "krishna":
                    terms.append("krish")
                elif ct == "abduljaha":
                    terms.append("abdul")
                elif ct == "shirley":
                    terms.append("shirly")
                elif ct == "ranisri21":
                    terms.extend(["ranisri", "rani"])
                    
            terms_map[name] = list(set(terms))
        return terms_map

    def match_query(self, query: str) -> list[dict]:
        """
        Parses potential candidate names from the search query.
        Returns a list of matching candidate details:
        [
            {
                "canonical_name": "M Ashok reddy",
                "matched_term": "ashok",
                "query_token": "ashok",
                "similarity": 100.0,
                "is_spelling_mistake": False
            }
        ]
        """
        cleaned_query = query.lower()
        # Find potential words in the query
        words = re.findall(r'[a-zA-Z0-9]+', cleaned_query)
        
        # Stop words to exclude from name matching
        stop_words = {
            "what", "where", "when", "who", "how", "why", "which",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
            "should", "now", "skills", "resume", "experience", "candidate", "candidates",
            "details", "profile", "projects", "education", "info", "information", "tell",
            "show", "give", "find", "get", "search", "query", "question", "please", "me", "him", "her",
            "about", "work", "role", "tech", "job", "career", "contact", "mail", "phone", "address",
            "pdf", "docx"
        }
        
        # Filtered tokens
        query_tokens = [w for w in words if w not in stop_words and len(w) > 1]
        
        # Generate sliding windows to capture multi-word candidate mentions
        query_windows = []
        # Single words
        for w in query_tokens:
            query_windows.append(w)
        # Pairs of adjacent words
        for i in range(len(query_tokens) - 1):
            query_windows.append(f"{query_tokens[i]} {query_tokens[i+1]}")
            
        matches_per_window = {}
        for qw in query_windows:
            best_match = None
            best_score = -1
            
            for canonical_name, terms in self.candidate_terms.items():
                for term in terms:
                    score = fuzz.ratio(qw, term)
                    if score >= self.threshold and score > best_score:
                        canonical_clean_tokens = [self._clean_token(t) for t in canonical_name.split() if len(self._clean_token(t)) > 1]
                        is_exact = (qw in canonical_clean_tokens) or (qw == canonical_name.lower())
                        is_spelling_mistake = not is_exact
                        
                        best_score = score
                        best_match = {
                            "canonical_name": canonical_name,
                            "matched_term": term,
                            "query_token": qw,
                            "similarity": score,
                            "is_spelling_mistake": is_spelling_mistake
                        }
            if best_match:
                matches_per_window[qw] = best_match
                
        # Deduplicate matches by canonical name
        unique_canonical_matches = {}
        for match in matches_per_window.values():
            name = match["canonical_name"]
            # If the same candidate is matched by multiple windows (e.g. "ashok" and "ashok reddy"),
            # keep the one with higher similarity score. If scores are equal, prefer the longer query token.
            if name not in unique_canonical_matches:
                unique_canonical_matches[name] = match
            else:
                existing = unique_canonical_matches[name]
                if match["similarity"] > existing["similarity"]:
                    unique_canonical_matches[name] = match
                elif match["similarity"] == existing["similarity"]:
                    if len(match["query_token"]) > len(existing["query_token"]):
                        unique_canonical_matches[name] = match
                        
        return list(unique_canonical_matches.values())

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    matcher = FuzzyNameMatcher()
    print("Candidates found:", len(matcher.candidates))
    for name in matcher.candidates[:5]:
        print(f"  {name}")
    
    test_queries = [
        "What are the skills of Pawan?",
        "Can you share Ashok Reddy's details?",
        "Compare Yasasvi and Trinad",
        "What is the contact number of Shirley?"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Matches:", matcher.match_query(q))
