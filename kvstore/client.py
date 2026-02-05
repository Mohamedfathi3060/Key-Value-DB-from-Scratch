"""
Key-Value Store Client Library

Provides a simple interface to interact with the KV Store server.
"""

import socket
import json
from typing import Any, Optional, List, Tuple


class KVClientError(Exception):
    """Base exception for KV Client errors."""
    pass


class ConnectionError(KVClientError):
    """Connection-related errors."""
    pass


class OperationError(KVClientError):
    """Operation-related errors."""
    pass


class KVClient:
    """
    Client for the Key-Value Store.
    
    Usage:
        client = KVClient(host="localhost", port=9010)
        client.set("key", "value")
        value = client.get("key")
        client.close()
    
    Or with context manager:
        with KVClient() as client:
            client.set("key", "value")
    """
    
    def __init__(self, host: str = "localhost", port: int = 9010, timeout: float = 30.0):
        """
        Initialize the client.
        
        Args:
            host: Server hostname
            port: Server port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self._buffer = b""
        self._connect()
    
    def _connect(self):
        """Establish connection to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
    
    def _reconnect(self):
        """Reconnect to the server."""
        self.close()
        self._buffer = b""
        self._connect()
    
    def _send_request(self, request: dict) -> dict:
        """
        Send a request and receive the response.
        
        Args:
            request: Dictionary containing the request
            
        Returns:
            Response dictionary
        """
        if self.socket is None:
            self._connect()
        
        try:
            # Send request
            data = json.dumps(request) + "\n"
            self.socket.sendall(data.encode("utf-8"))
            
            # Receive response
            while b"\n" not in self._buffer:
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed by server")
                self._buffer += chunk
            
            # Parse response
            line, self._buffer = self._buffer.split(b"\n", 1)
            response = json.loads(line.decode("utf-8"))
            
            return response
        
        except socket.timeout:
            raise ConnectionError("Request timed out")
        except socket.error as e:
            raise ConnectionError(f"Socket error: {e}")
        except json.JSONDecodeError as e:
            raise OperationError(f"Invalid response from server: {e}")
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a key-value pair.
        
        Args:
            key: The key (string)
            value: The value (any JSON-serializable type)
            
        Returns:
            True if successful
            
        Raises:
            OperationError: If the operation fails
        """
        response = self._send_request({
            "op": "set",
            "key": str(key),
            "value": value
        })
        
        if response["status"] == "ok":
            return True
        else:
            raise OperationError(response.get("message", "Unknown error"))
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found, None otherwise
        """
        response = self._send_request({
            "op": "get",
            "key": str(key)
        })
        
        if response["status"] == "ok":
            return response["value"]
        elif response["status"] == "not_found":
            return None
        else:
            raise OperationError(response.get("message", "Unknown error"))
    
    def delete(self, key: str) -> bool:
        """
        Delete a key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if the key existed and was deleted, False otherwise
        """
        response = self._send_request({
            "op": "delete",
            "key": str(key)
        })
        
        if response["status"] == "ok":
            return True
        elif response["status"] == "not_found":
            return False
        else:
            raise OperationError(response.get("message", "Unknown error"))
    
    def bulk_set(self, items: List[Tuple[str, Any]]) -> bool:
        """
        Set multiple key-value pairs atomically.
        
        Args:
            items: List of (key, value) tuples
            
        Returns:
            True if successful
            
        Raises:
            OperationError: If the operation fails
        """
        response = self._send_request({
            "op": "bulk_set",
            "items": [[str(k), v] for k, v in items]
        })
        
        if response["status"] == "ok":
            return True
        else:
            raise OperationError(response.get("message", "Unknown error"))
    
    def ping(self) -> bool:
        """
        Check if the server is alive.
        
        Returns:
            True if server responds
        """
        try:
            response = self._send_request({"op": "ping"})
            return response["status"] == "ok"
        except Exception:
            return False
    
    def stats(self) -> dict:
        """
        Get server statistics.
        
        Returns:
            Dictionary with stats (keys, wal_size)
        """
        response = self._send_request({"op": "stats"})
        if response["status"] == "ok":
            return {
                "keys": response.get("keys", 0),
                "wal_size": response.get("wal_size", 0)
            }
        raise OperationError(response.get("message", "Unknown error"))
    
    def search_by_value(self, value: Any) -> List[str]:
        """
        Search for keys that have a specific value.
        
        Args:
            value: The value to search for
            
        Returns:
            List of keys that have this value
        """
        response = self._send_request({"op": "search_by_value", "value": value})
        if response["status"] == "ok":
            return response.get("keys", [])
        raise OperationError(response.get("message", "Unknown error"))
    
    def fulltext_search(self, query: str) -> List[str]:
        """
        Full-text search: find keys containing all words in the query.
        
        Args:
            query: Search query string (words separated by spaces)
            
        Returns:
            List of keys whose values contain all query words
        """
        response = self._send_request({"op": "fulltext_search", "query": query})
        if response["status"] == "ok":
            return response.get("keys", [])
        raise OperationError(response.get("message", "Unknown error"))
    
    def semantic_search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search by meaning using word embeddings (all-MiniLM-L6-v2).
        Returns top-k keys whose values are most similar to the query, above the threshold.
        Requires server to have sentence-transformers installed.
        
        Args:
            query: Natural language search query
            k: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of (key, score) tuples, sorted by score descending
        """
        response = self._send_request({
            "op": "semantic_search",
            "query": query,
            "k": k,
            "threshold": threshold,
        })
        if response["status"] == "ok":
            return [(r["key"], r["score"]) for r in response.get("results", [])]
        raise OperationError(response.get("message", "Unknown error"))
    
    def close(self):
        """Close the connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor."""
        self.close()