"""
Test indexing functionality: value index and full-text search.
"""

import os
import sys
import unittest
import time
import threading
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import KVServer
from client import KVClient


class TestIndexing(unittest.TestCase):
    """Test indexing features."""

    def setUp(self):
        self.test_dir = "test_index_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.server = KVServer(host="localhost", port=9800, data_dir=self.test_dir)
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)
        self.server_thread.start()
        time.sleep(0.5)
        
        self.client = KVClient(host="localhost", port=9800)

    def tearDown(self):
        self.client.close()
        self.server.stop()
        time.sleep(0.3)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_value_index(self):
        """Test searching by value."""
        # Set some values
        self.client.set("user:1", "Alice")
        self.client.set("user:2", "Bob")
        self.client.set("user:3", "Alice")
        self.client.set("user:4", "Charlie")
        
        # Search for keys with value "Alice"
        keys = self.client.search_by_value("Alice")
        self.assertEqual(len(keys), 2)
        self.assertIn("user:1", keys)
        self.assertIn("user:3", keys)
        
        # Search for "Bob"
        keys = self.client.search_by_value("Bob")
        self.assertEqual(len(keys), 1)
        self.assertIn("user:2", keys)
        
        # Search for non-existent value
        keys = self.client.search_by_value("David")
        self.assertEqual(len(keys), 0)

    def test_fulltext_search(self):
        """Test full-text search."""
        # Set text values
        self.client.set("doc:1", "Python is a programming language")
        self.client.set("doc:2", "Python and Java are languages")
        self.client.set("doc:3", "Java is also a programming language")
        self.client.set("doc:4", "JavaScript is different from Java")
        
        # Search for "Python"
        keys = self.client.fulltext_search("Python")
        self.assertEqual(len(keys), 2)
        self.assertIn("doc:1", keys)
        self.assertIn("doc:2", keys)
        
        # Search for "Java"
        keys = self.client.fulltext_search("Java")
        self.assertEqual(len(keys), 3)
        self.assertIn("doc:2", keys)
        self.assertIn("doc:3", keys)
        self.assertIn("doc:4", keys)
        
        # Search for multiple words (AND)
        keys = self.client.fulltext_search("Python programming")
        self.assertEqual(len(keys), 1)
        self.assertIn("doc:1", keys)
        
        # Search for "programming language"
        keys = self.client.fulltext_search("programming language")
        self.assertEqual(len(keys), 2)
        self.assertIn("doc:1", keys)
        self.assertIn("doc:3", keys)

    def test_index_update_on_delete(self):
        """Test that indexes are updated when keys are deleted."""
        self.client.set("key1", "value1")
        self.client.set("key2", "value1")
        
        keys = self.client.search_by_value("value1")
        self.assertEqual(len(keys), 2)
        
        self.client.delete("key1")
        
        keys = self.client.search_by_value("value1")
        self.assertEqual(len(keys), 1)
        self.assertIn("key2", keys)
        self.assertNotIn("key1", keys)

    def test_index_update_on_overwrite(self):
        """Test that indexes are updated when values are overwritten."""
        self.client.set("key1", "old value")
        self.client.set("key2", "old value")
        
        keys = self.client.search_by_value("old value")
        self.assertEqual(len(keys), 2)
        
        # Overwrite key1
        self.client.set("key1", "new value")
        
        # Old value should only have key2
        keys = self.client.search_by_value("old value")
        self.assertEqual(len(keys), 1)
        self.assertIn("key2", keys)
        
        # New value should have key1
        keys = self.client.search_by_value("new value")
        self.assertEqual(len(keys), 1)
        self.assertIn("key1", keys)

    def test_fulltext_case_insensitive(self):
        """Test that full-text search is case-insensitive."""
        self.client.set("doc:1", "Python Programming")
        self.client.set("doc:2", "python is great")
        
        keys = self.client.fulltext_search("python")
        self.assertEqual(len(keys), 2)
        
        keys = self.client.fulltext_search("PYTHON")
        self.assertEqual(len(keys), 2)
        
        keys = self.client.fulltext_search("Python")
        self.assertEqual(len(keys), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
