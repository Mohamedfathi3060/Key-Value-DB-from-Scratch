#!/usr/bin/env python3
"""
Helper script to run the KV Store server, tests, and benchmarks.

Usage:
    python run.py server              # Start the server
    python run.py test                # Run tests
    python run.py benchmark           # Run benchmarks (starts own server)
    python run.py benchmark --port X  # Run benchmarks against existing server
    python run.py benchmark-acid      # Run ACID benchmarks (isolation + atomicity kill)
    python run.py demo                # Run a quick demo
"""

import sys
import os
import subprocess
import time
import threading
import shutil


def run_server():
    """Run the KV store server."""
    from server import KVServer
    
    server = KVServer(host="localhost", port=9000, data_dir="data")
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def run_tests():
    """Run the test suite."""
    import unittest
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


def run_benchmarks(port=None):
    """Run benchmarks."""
    from benchmark import run_all_benchmarks, start_benchmark_server
    
    if port:
        run_all_benchmarks("localhost", port)
    else:
        from server import KVServer
        
        test_dir = "benchmark_data"
        port = 9500
        
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        server = KVServer(port=port, data_dir=test_dir)
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)
        
        try:
            run_all_benchmarks("localhost", port)
        finally:
            server.stop()
            time.sleep(0.3)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


def run_benchmark_acid(port=None):
    """Run ACID benchmarks (isolation + bulk atomicity under SIGKILL)."""
    from benchmark_ACID import run_all_acid_benchmarks
    
    if port is not None:
        run_all_acid_benchmarks("localhost", port, run_isolation=True, run_atomicity_kill=True)
        return
    
    from server import KVServer
    
    test_dir = "benchmark_acid_data"
    port = 9500
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    server = KVServer(port=port, data_dir=test_dir)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    
    try:
        run_all_acid_benchmarks("localhost", port, run_isolation=True, run_atomicity_kill=True)
    finally:
        server.stop()
        time.sleep(0.3)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def run_demo():
    """Run a quick demo of the KV store."""
    from server import KVServer
    from client import KVClient
    
    print("=" * 60)
    print("KEY-VALUE STORE DEMO")
    print("=" * 60)
    
    # Clean up and start server
    test_dir = "demo_data"
    port = 9600
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("\n1. Starting server...")
    server = KVServer(port=port, data_dir=test_dir)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    
    try:
        print("\n2. Creating client and connecting...")
        client = KVClient(port=port)
        
        print("\n3. Setting some values...")
        client.set("name", "Alice")
        client.set("age", 30)
        client.set("hobbies", ["reading", "coding", "hiking"])
        client.set("config", {"debug": True, "max_connections": 100})
        print("   Set: name='Alice', age=30, hobbies=[...], config={...}")
        
        print("\n4. Getting values...")
        print(f"   name = {client.get('name')}")
        print(f"   age = {client.get('age')}")
        print(f"   hobbies = {client.get('hobbies')}")
        print(f"   config = {client.get('config')}")
        
        print("\n5. Bulk setting...")
        client.bulk_set([
            ("user:1", {"id": 1, "name": "Bob"}),
            ("user:2", {"id": 2, "name": "Charlie"}),
            ("user:3", {"id": 3, "name": "Diana"}),
        ])
        print("   Set 3 user records")
        
        print("\n6. Getting bulk-set values...")
        print(f"   user:1 = {client.get('user:1')}")
        print(f"   user:2 = {client.get('user:2')}")
        print(f"   user:3 = {client.get('user:3')}")
        
        print("\n7. Deleting a key...")
        client.delete("age")
        print(f"   Deleted 'age', now age = {client.get('age')}")
        
        print("\n8. Overwriting a value...")
        client.set("name", "Alice Smith")
        print(f"   Updated name = {client.get('name')}")
        
        print("\n9. Server stats:")
        stats = client.stats()
        print(f"   Keys: {stats['keys']}")
        print(f"   WAL Size: {stats['wal_size']} bytes")
        
        print("\n10. Testing persistence...")
        client.close()
        server.stop()
        time.sleep(0.5)
        
        print("    Server stopped. Restarting...")
        server2 = KVServer(port=port+1, data_dir=test_dir)
        server_thread2 = threading.Thread(target=server2.start, daemon=True)
        server_thread2.start()
        time.sleep(0.5)
        
        client2 = KVClient(port=port+1)
        print(f"    After restart, name = {client2.get('name')}")
        print(f"    After restart, user:1 = {client2.get('user:1')}")
        
        client2.close()
        server2.stop()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "server":
        run_server()
    elif command == "test":
        return run_tests()
    elif command == "benchmark":
        port = None
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        run_benchmarks(port)
    elif command == "benchmark-acid":
        port = None
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        run_benchmark_acid(port)
    elif command == "demo":
        run_demo()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)