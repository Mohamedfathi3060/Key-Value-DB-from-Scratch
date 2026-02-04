"""
Tests for semantic search (search by meaning via embeddings).
Run only when sentence-transformers is installed: pip install sentence-transformers
"""

import os
import sys
import unittest
import time
import threading
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from embedding_index import EmbeddingIndex, EmbeddingUnavailableError
    _EMBEDDING_AVAILABLE = True
except ImportError:
    _EMBEDDING_AVAILABLE = False

from server import KVServer
from client import KVClient


@unittest.skipIf(not _EMBEDDING_AVAILABLE, "sentence-transformers not installed")
class TestSemanticSearch(unittest.TestCase):
    """Test semantic search (requires sentence-transformers)."""

    def setUp(self):
        self.test_dir = "test_semantic_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.server = KVServer(host="localhost", port=9801, data_dir=self.test_dir)
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)
        self.server_thread.start()
        time.sleep(0.5)
        
        self.client = KVClient(host="localhost", port=9801)

    def tearDown(self):
        self.client.close()
        self.server.stop()
        time.sleep(0.3)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_semantic_search_top_k(self):
        """Semantic search returns at most k results above threshold."""
        self.client.set("doc:1", "The quick brown fox jumps over the lazy dog")
        self.client.set("doc:2", "A fast animal runs in the forest")
        self.client.set("doc:3", "Programming in Python is fun")
        
        results = self.client.semantic_search("fast fox animal", k=2, threshold=0.0)
        self.assertLessEqual(len(results), 2)
        for key, score in results:
            self.assertIn(key, ["doc:1", "doc:2", "doc:3"])
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_semantic_search_threshold(self):
        """Results have scores above the given threshold."""
        self.client.set("doc:a", "Machine learning and artificial intelligence")
        self.client.set("doc:b", "Cooking pasta and Italian food")
        
        results = self.client.semantic_search("AI and ML", k=10, threshold=0.2)
        for key, score in results:
            self.assertGreaterEqual(score, 0.2)
        # doc:a should be more similar to "AI and ML" than doc:b
        if len(results) >= 2:
            self.assertGreaterEqual(results[0][1], results[1][1])

    def test_semantic_search_ordering(self):
        """Results are ordered by score descending."""
        self.client.set("p1", "Dogs and cats are pets")
        self.client.set("p2", "Dogs are loyal animals")
        self.client.set("p3", "The weather is nice today")
        
        results = self.client.semantic_search("dogs and cats", k=5, threshold=0.0)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
