import unittest
import sys
from pathlib import Path

# Add src/ to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from matcher import FuzzyNameMatcher
from classifier import QueryClassifier
from retriever import FilteredQueryEngine

class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        self.matcher = FuzzyNameMatcher()
        self.classifier = QueryClassifier()
        self.engine = FilteredQueryEngine()

    def test_fuzzy_name_matching_exact(self):
        # Exact match
        matches = self.matcher.match_query("Tell me about M Ashok reddy")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["canonical_name"], "M Ashok reddy")
        self.assertFalse(matches[0]["is_spelling_mistake"])

    def test_fuzzy_name_matching_misspelled(self):
        # Misspelled name: pawan -> pavanteja kamma
        matches = self.matcher.match_query("What are the skills of Pawan?")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["canonical_name"], "pavanteja kamma")
        self.assertTrue(matches[0]["is_spelling_mistake"])

        # Misspelled name: yasasvi -> yasaswi kotha
        matches2 = self.matcher.match_query("Summarize yasasvi's achievements")
        self.assertEqual(len(matches2), 1)
        self.assertEqual(matches2[0]["canonical_name"], "yasaswi kotha")
        self.assertTrue(matches2[0]["is_spelling_mistake"])

    def test_query_classification(self):
        # Test classification logic (tests fallback or LLM)
        # Single candidate query
        res1 = self.classifier.classify("What is Ashok's experience in Python?")
        self.assertTrue(res1["is_single_candidate"])

        # Multi candidate query
        res2 = self.classifier.classify("Compare Ashok and Pawan's python experience")
        self.assertFalse(res2["is_single_candidate"])

        # Aggregation/Comparison query
        res3 = self.classifier.classify("Who is the best candidate for Java?")
        self.assertFalse(res3["is_single_candidate"])

    def test_pipeline_routing(self):
        # Single candidate query
        res1 = self.engine.query("What are the skills of Pawan?")
        self.assertFalse(res1["is_blocked"])
        self.assertEqual(res1["candidate_name"], "pavanteja kamma")
        self.assertIsNotNone(res1["spelling_suggestion"])

        # Multi-candidate comparison query (should be blocked)
        res2 = self.engine.query("Compare Ashok and Pawan")
        self.assertTrue(res2["is_blocked"])
        self.assertIn("blocked", res2["response"].lower())

        # No candidate mentioned query (should be blocked)
        res3 = self.engine.query("What is python?")
        self.assertTrue(res3["is_blocked"])
        self.assertIn("no candidate name", res3["blocked_reason"].lower())

if __name__ == "__main__":
    unittest.main()
