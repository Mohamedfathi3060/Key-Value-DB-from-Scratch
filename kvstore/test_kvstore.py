"""
Unit Tests for Key-Value Store

Tests cover:
1. Set then Get
2. Set then Delete then Get
3. Get without setting
4. Set then Set (same key) then Get
5. Set then exit (gracefully) then Get (persistence test)
6. Bulk Set operations
"""

import unittest
import threading
import time
import os
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import KVClient, KVClientError, ConnectionError
from server import KVServer, KVStore, WriteAheadLog


class TestWriteAheadLog(unittest.TestCase):
    """Tests for the WAL component."""
    
    def setUp(self):
        self.test_dir = "test_wal_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.wal = WriteAheadLog(f"{self.test_dir}/wal.log")
    
    def tearDown(self):
        self.wal.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_append_and_replay(self):
        """Test basic append and replay functionality."""
        self.wal.append("set", "key1", "value1")
        self.wal.append("set", "key2", "value2")
        self.wal.append("delete", "key1")
        
        self.wal.close()
        
        new_wal = WriteAheadLog(f"{self.test_dir}/wal.log")
        state = new_wal.replay()
        new_wal.close()
        
        self.assertNotIn("key1", state)
        self.assertEqual(state["key2"], "value2")
    
    def test_batch_append(self):
        """Test batch append functionality."""
        operations = [
            ("set", "batch1", "val1"),
            ("set", "batch2", "val2"),
            ("set", "batch3", "val3"),
        ]
        self.wal.append_batch(operations)
        
        self.wal.close()
        
        new_wal = WriteAheadLog(f"{self.test_dir}/wal.log")
        state = new_wal.replay()
        new_wal.close()
        
        self.assertEqual(state["batch1"], "val1")
        self.assertEqual(state["batch2"], "val2")
        self.assertEqual(state["batch3"], "val3")


class TestKVStoreLocal(unittest.TestCase):
    """Tests for the KVStore class directly (no network)."""
    
    def setUp(self):
        self.test_dir = "test_kvstore_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.store = KVStore(self.test_dir)
    
    def tearDown(self):
        self.store.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_set_get(self):
        """Test: Set then Get"""
        self.assertTrue(self.store.set("key1", "value1"))
        self.assertEqual(self.store.get("key1"), "value1")
    
    def test_set_delete_get(self):
        """Test: Set then Delete then Get"""
        self.store.set("key2", "value2")
        self.assertEqual(self.store.get("key2"), "value2")
        self.assertTrue(self.store.delete("key2"))
        self.assertIsNone(self.store.get("key2"))
    
    def test_get_nonexistent(self):
        """Test: Get without setting"""
        self.assertIsNone(self.store.get("nonexistent"))
    
    def test_overwrite(self):
        """Test: Set then Set (same key) then Get"""
        self.store.set("key3", "original")
        self.assertEqual(self.store.get("key3"), "original")
        self.store.set("key3", "updated")
        self.assertEqual(self.store.get("key3"), "updated")
    
    def test_bulk_set(self):
        """Test: Bulk Set"""
        items = [("bulk1", "v1"), ("bulk2", "v2"), ("bulk3", "v3")]
        self.assertTrue(self.store.bulk_set(items))
        self.assertEqual(self.store.get("bulk1"), "v1")
        self.assertEqual(self.store.get("bulk2"), "v2")
        self.assertEqual(self.store.get("bulk3"), "v3")
    
    def test_complex_values(self):
        """Test storing complex values (dicts, lists)."""
        self.store.set("dict_key", {"nested": {"value": 123}})
        self.store.set("list_key", [1, 2, 3, "four"])
        
        self.assertEqual(self.store.get("dict_key"), {"nested": {"value": 123}})
        self.assertEqual(self.store.get("list_key"), [1, 2, 3, "four"])


class TestKVStoreWithServer(unittest.TestCase):
    """Tests using the client-server architecture."""
    
    @classmethod
    def setUpClass(cls):
        """Start the server before all tests."""
        cls.test_dir = "test_server_data"
        cls.port = 9100
        
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        
        cls.server = KVServer(port=cls.port, data_dir=cls.test_dir)
        cls.server_thread = threading.Thread(target=cls.server.start, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)  # Wait for server to start
    
    @classmethod
    def tearDownClass(cls):
        """Stop the server after all tests."""
        cls.server.stop()
        time.sleep(0.3)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Create a new client for each test."""
        self.client = KVClient(port=self.port)
    
    def tearDown(self):
        """Close client after each test."""
        self.client.close()
    
    def test_set_then_get(self):
        """Test: Set then Get"""
        self.assertTrue(self.client.set("net_key1", "net_value1"))
        self.assertEqual(self.client.get("net_key1"), "net_value1")
    
    def test_set_delete_get(self):
        """Test: Set then Delete then Get"""
        self.client.set("net_key2", "net_value2")
        self.assertEqual(self.client.get("net_key2"), "net_value2")
        self.assertTrue(self.client.delete("net_key2"))
        self.assertIsNone(self.client.get("net_key2"))
    
    def test_get_without_setting(self):
        """Test: Get without setting"""
        self.assertIsNone(self.client.get("net_nonexistent_key"))
    
    def test_set_overwrite_get(self):
        """Test: Set then Set (same key) then Get"""
        self.client.set("net_key3", "original_value")
        self.assertEqual(self.client.get("net_key3"), "original_value")
        self.client.set("net_key3", "updated_value")
        self.assertEqual(self.client.get("net_key3"), "updated_value")
    
    def test_bulk_set(self):
        """Test: Bulk Set"""
        items = [
            ("net_bulk1", "val1"),
            ("net_bulk2", "val2"),
            ("net_bulk3", "val3"),
            ("net_bulk4", {"nested": True}),
        ]
        self.assertTrue(self.client.bulk_set(items))
        
        self.assertEqual(self.client.get("net_bulk1"), "val1")
        self.assertEqual(self.client.get("net_bulk2"), "val2")
        self.assertEqual(self.client.get("net_bulk3"), "val3")
        self.assertEqual(self.client.get("net_bulk4"), {"nested": True})
    
    def test_delete_nonexistent(self):
        """Test: Delete a key that doesn't exist"""
        self.assertFalse(self.client.delete("never_existed_key"))
    
    def test_ping(self):
        """Test: Ping server"""
        self.assertTrue(self.client.ping())
    
    def test_stats(self):
        """Test: Get server stats"""
        stats = self.client.stats()
        self.assertIn("keys", stats)
        self.assertIn("wal_size", stats)


class TestPersistence(unittest.TestCase):
    """
    Test persistence across server restarts.
    This is the critical test for ACID durability.
    """
    
    def test_persistence_after_restart(self):
        """Test: Set then exit (gracefully) then Get"""
        test_dir = "test_persistence_data"
        port = 9200
        
        # Clean up from previous runs
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        # === Phase 1: Start server, write data, stop server ===
        server1 = KVServer(port=port, data_dir=test_dir)
        server_thread1 = threading.Thread(target=server1.start, daemon=True)
        server_thread1.start()
        time.sleep(0.5)
        
        client1 = KVClient(port=port)
        
        # Write various types of data
        client1.set("persist_string", "hello world")
        client1.set("persist_number", 42)
        client1.set("persist_dict", {"key": "value", "nested": {"a": 1}})
        client1.set("persist_list", [1, 2, 3, "four"])
        client1.bulk_set([
            ("persist_bulk1", "bulk_value1"),
            ("persist_bulk2", "bulk_value2"),
        ])
        
        # Also test delete persistence
        client1.set("persist_to_delete", "will be deleted")
        client1.delete("persist_to_delete")
        
        # Verify data before shutdown
        self.assertEqual(client1.get("persist_string"), "hello world")
        
        client1.close()
        server1.stop()
        time.sleep(0.5)  # Wait for graceful shutdown
        
        # === Phase 2: Start new server, verify data persisted ===
        port2 = 9201  # Use different port to avoid binding issues
        server2 = KVServer(port=port2, data_dir=test_dir)
        server_thread2 = threading.Thread(target=server2.start, daemon=True)
        server_thread2.start()
        time.sleep(0.5)
        
        client2 = KVClient(port=port2)
        
        # Verify all data persisted correctly
        self.assertEqual(client2.get("persist_string"), "hello world")
        self.assertEqual(client2.get("persist_number"), 42)
        self.assertEqual(client2.get("persist_dict"), {"key": "value", "nested": {"a": 1}})
        self.assertEqual(client2.get("persist_list"), [1, 2, 3, "four"])
        self.assertEqual(client2.get("persist_bulk1"), "bulk_value1")
        self.assertEqual(client2.get("persist_bulk2"), "bulk_value2")
        
        # Verify deleted key stayed deleted
        self.assertIsNone(client2.get("persist_to_delete"))
        
        client2.close()
        server2.stop()
        
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    
    def test_persistence_after_many_writes(self):
        """Test persistence with many writes to verify WAL handles volume."""
        test_dir = "test_persistence_volume_data"
        port = 9300
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        # Phase 1: Write lots of data
        server1 = KVServer(port=port, data_dir=test_dir)
        server_thread1 = threading.Thread(target=server1.start, daemon=True)
        server_thread1.start()
        time.sleep(0.5)
        
        client1 = KVClient(port=port)
        
        # Write 100 keys with overwrites
        for i in range(100):
            client1.set(f"key_{i}", f"value_{i}_v1")
        
        # Overwrite half of them
        for i in range(0, 100, 2):
            client1.set(f"key_{i}", f"value_{i}_v2")
        
        client1.close()
        server1.stop()
        time.sleep(0.5)
        
        # Phase 2: Verify
        port2 = 9301
        server2 = KVServer(port=port2, data_dir=test_dir)
        server_thread2 = threading.Thread(target=server2.start, daemon=True)
        server_thread2.start()
        time.sleep(0.5)
        
        client2 = KVClient(port=port2)
        
        # Verify values
        for i in range(100):
            expected = f"value_{i}_v2" if i % 2 == 0 else f"value_{i}_v1"
            self.assertEqual(client2.get(f"key_{i}"), expected)
        
        client2.close()
        server2.stop()
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


class TestConcurrency(unittest.TestCase):
    """Tests for concurrent access."""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_concurrent_data"
        cls.port = 9400
        
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        
        cls.server = KVServer(port=cls.port, data_dir=cls.test_dir)
        cls.server_thread = threading.Thread(target=cls.server.start, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)
    
    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        time.sleep(0.3)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_concurrent_writes(self):
        """Test multiple clients writing concurrently."""
        num_threads = 5
        writes_per_thread = 20
        results = []
        errors = []
        
        def writer(thread_id):
            try:
                client = KVClient(port=self.port)
                for i in range(writes_per_thread):
                    key = f"concurrent_{thread_id}_{i}"
                    value = f"value_{thread_id}_{i}"
                    client.set(key, value)
                    results.append((key, value))
                client.close()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        # Verify all writes succeeded
        client = KVClient(port=self.port)
        for key, value in results:
            self.assertEqual(client.get(key), value)
        client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)