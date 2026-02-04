"""
Persistent Key-Value Store Server with Write-Ahead Logging (WAL)

Features:
- ACID compliant with fsync on every write
- Write-Ahead Logging for durability
- TCP-based communication with JSON protocol
- Thread-safe operations
- Recovery from WAL on startup
- Supports: Set, Get, Delete, BulkSet
"""

import socket
import json
import os
import threading
import time
import struct
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


class WriteAheadLog:
    """
    Write-Ahead Log for ensuring durability.
    All mutations are written to the log before being applied.
    Uses fsync to guarantee data is persisted to disk.
    """
    
    def __init__(self, path: str = "data/wal.log"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.lock = threading.Lock()
        self._open()
    
    def _open(self):
        """Open the WAL file in append mode with no buffering."""
        self.file = open(self.path, "ab+", buffering=0)
    
    def append(self, operation: str, key: str, value: Any = None) -> int:
        """
        Append a single operation to the WAL.
        Returns the log sequence number (LSN).
        """
        with self.lock:
            entry = {
                "op": operation,
                "key": key,
                "value": value,
                "ts": time.time_ns()
            }
            data = json.dumps(entry) + "\n"
            self.file.write(data.encode())
            self.file.flush()
            os.fsync(self.file.fileno())  # CRITICAL: Ensures durability
            return entry["ts"]
    
    def append_batch(self, operations: List[Tuple[str, str, Any]]) -> int:
        """
        Append multiple operations atomically.
        All operations share the same timestamp for atomicity.
        """
        with self.lock:
            timestamp = time.time_ns()
            entries = []
            for op, key, value in operations:
                entry = {
                    "op": op,
                    "key": key,
                    "value": value,
                    "ts": timestamp
                }
                entries.append(json.dumps(entry))
            
            data = "\n".join(entries) + "\n"
            self.file.write(data.encode())
            self.file.flush()
            os.fsync(self.file.fileno())  # CRITICAL: Ensures durability
            return timestamp
    
    def replay(self) -> Dict[str, Any]:
        """
        Replay the WAL to reconstruct the database state.
        Called during recovery after a restart.
        """
        data = {}
        if not self.path.exists():
            return data
        
        with open(self.path, "r") as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry["op"] == "set":
                        data[entry["key"]] = entry["value"]
                    elif entry["op"] == "delete":
                        data.pop(entry["key"], None)
                except json.JSONDecodeError as e:
                    print(f"Warning: Corrupt WAL entry at line {line_num}: {e}")
                    continue
        
        return data
    
    def get_size(self) -> int:
        """Get the current WAL file size in bytes."""
        return self.path.stat().st_size if self.path.exists() else 0
    
    def compact(self, current_state: Dict[str, Any]):
        """
        Compact the WAL by rewriting it with only the current state.
        This reduces file size by removing superseded entries.
        """
        with self.lock:
            temp_path = self.path.with_suffix(".tmp")
            
            # Write current state to temp file
            with open(temp_path, "w") as f:
                for key, value in current_state.items():
                    entry = {
                        "op": "set",
                        "key": key,
                        "value": value,
                        "ts": time.time_ns()
                    }
                    f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            
            # Atomically replace old WAL with compacted version
            self.file.close()
            os.replace(temp_path, self.path)
            self._open()
    
    def close(self):
        """Close the WAL file."""
        with self.lock:
            if self.file:
                self.file.flush()
                os.fsync(self.file.fileno())
                self.file.close()
                self.file = None


class KVStore:
    """
    In-memory key-value store with WAL-based persistence.
    
    Provides ACID guarantees:
    - Atomicity: Operations are atomic (especially bulk_set)
    - Consistency: Data is always in a valid state
    - Isolation: Thread-safe with RLock
    - Durability: fsync ensures data survives crashes
    """
    
    # Compact WAL when it exceeds this size (bytes)
    COMPACTION_THRESHOLD = 10 * 1024 * 1024  # 10MB
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.wal = WriteAheadLog(self.data_dir / "wal.log")
        self.data: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.operation_count = 0
        
        self._recover()
    
    def _recover(self):
        """Recover state from WAL after restart."""
        print(f"[KVStore] Starting recovery from {self.data_dir}...")
        start = time.time()
        self.data = self.wal.replay()
        elapsed = time.time() - start
        print(f"[KVStore] Recovered {len(self.data)} keys in {elapsed:.3f}s")
    
    def _maybe_compact(self):
        """Check if compaction is needed and perform it."""
        self.operation_count += 1
        if self.operation_count >= 1000:  # Check every 1000 operations
            self.operation_count = 0
            if self.wal.get_size() > self.COMPACTION_THRESHOLD:
                print("[KVStore] Starting WAL compaction...")
                self.wal.compact(self.data)
                print("[KVStore] WAL compaction complete")
    
    def set(self, key: str, value: Any) -> bool:
        """Set a key-value pair with durability guarantee."""
        with self.lock:
            # Write to WAL first (Write-Ahead)
            self.wal.append("set", key, value)
            # Then update in-memory state
            self.data[key] = value
            self._maybe_compact()
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key. Returns None if not found."""
        with self.lock:
            return self.data.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        with self.lock:
            if key in self.data:
                # Write to WAL first
                self.wal.append("delete", key)
                # Then update in-memory state
                del self.data[key]
                self._maybe_compact()
                return True
            return False
    
    def bulk_set(self, items: List[Tuple[str, Any]]) -> bool:
        """
        Set multiple key-value pairs atomically.
        All items are written to WAL in a single fsync for efficiency.
        """
        with self.lock:
            if not items:
                return True
            
            # Write all operations to WAL atomically
            operations = [("set", k, v) for k, v in items]
            self.wal.append_batch(operations)
            
            # Update in-memory state
            for key, value in items:
                self.data[key] = value
            
            self._maybe_compact()
            return True
    
    def keys(self) -> List[str]:
        """Get all keys."""
        with self.lock:
            return list(self.data.keys())
    
    def size(self) -> int:
        """Get the number of keys in the store."""
        with self.lock:
            return len(self.data)
    
    def close(self):
        """Close the store gracefully."""
        print("[KVStore] Closing...")
        with self.lock:
            self.wal.close()
        print("[KVStore] Closed")


class KVServer:
    """
    TCP Server for the Key-Value Store.
    
    Protocol: Newline-delimited JSON
    Request:  {"op": "...", "key": "...", "value": "...", "items": [...]}
    Response: {"status": "ok|error|not_found", "value": "...", "message": "..."}
    """
    
    def __init__(self, host: str = "localhost", port: int = 9000, data_dir: str = "data"):
        self.host = host
        self.port = port
        self.store = KVStore(data_dir)
        self.server_socket = None
        self.running = False
        self.clients: List[threading.Thread] = []
    
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request and return response."""
        op = request.get("op")
        
        try:
            if op == "set":
                key = request["key"]
                value = request["value"]
                self.store.set(key, value)
                return {"status": "ok"}
            
            elif op == "get":
                key = request["key"]
                value = self.store.get(key)
                if value is not None:
                    return {"status": "ok", "value": value}
                else:
                    return {"status": "not_found"}
            
            elif op == "delete":
                key = request["key"]
                if self.store.delete(key):
                    return {"status": "ok"}
                else:
                    return {"status": "not_found"}
            
            elif op == "bulk_set":
                items = request["items"]
                # Convert list of lists to list of tuples
                items = [(item[0], item[1]) for item in items]
                self.store.bulk_set(items)
                return {"status": "ok", "count": len(items)}
            
            elif op == "ping":
                return {"status": "ok", "message": "pong"}
            
            elif op == "stats":
                return {
                    "status": "ok",
                    "keys": self.store.size(),
                    "wal_size": self.store.wal.get_size()
                }
            
            else:
                return {"status": "error", "message": f"Unknown operation: {op}"}
        
        except KeyError as e:
            return {"status": "error", "message": f"Missing required field: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle a single client connection."""
        print(f"[Server] Client connected: {address}")
        buffer = b""
        
        try:
            while self.running:
                # Set socket timeout to allow checking self.running
                client_socket.settimeout(1.0)
                try:
                    data = client_socket.recv(4096)
                except socket.timeout:
                    continue
                
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages (newline-delimited)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    
                    try:
                        request = json.loads(line.decode("utf-8"))
                        response = self._handle_request(request)
                    except json.JSONDecodeError:
                        response = {"status": "error", "message": "Invalid JSON"}
                    except Exception as e:
                        response = {"status": "error", "message": str(e)}
                    
                    response_data = json.dumps(response) + "\n"
                    client_socket.sendall(response_data.encode("utf-8"))
        
        except ConnectionResetError:
            pass
        except Exception as e:
            print(f"[Server] Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"[Server] Client disconnected: {address}")
    
    def start(self):
        """Start the server (blocking)."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(128)
        self.server_socket.settimeout(1.0)
        self.running = True
        
        print(f"[Server] Listening on {self.host}:{self.port}")
        
        try:
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    thread.start()
                    self.clients.append(thread)
                except socket.timeout:
                    continue
        except Exception as e:
            if self.running:
                print(f"[Server] Error: {e}")
    
    def stop(self):
        """Stop the server gracefully."""
        print("[Server] Stopping...")
        self.running = False
        
        # Close store (flushes WAL)
        self.store.close()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        print("[Server] Stopped")


def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Key-Value Store Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    parser.add_argument("--data-dir", default="data", help="Data directory for persistence")
    args = parser.parse_args()
    
    server = KVServer(host=args.host, port=args.port, data_dir=args.data_dir)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Server] Received interrupt signal")
    finally:
        server.stop()


if __name__ == "__main__":
    main()