"""
Cluster mode for the KV Store: 3 nodes, 1 primary + 2 secondaries.

- Writes and reads go only to the primary.
- Primary replicates all writes to secondaries (sync: wait for ack).
- If the primary goes down, the two secondaries run an election; one becomes primary.
"""

import socket
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from server import KVStore


class Role(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


# Default 3-node cluster: node_id -> (host, port)
DEFAULT_NODES = {
    0: ("localhost", 9010),
    1: ("localhost", 9011),
    2: ("localhost", 9012),
}


def _send_msg(sock: socket.socket, msg: dict) -> None:
    sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))


def _recv_msg(sock: socket.socket, timeout: float = 5.0) -> Optional[dict]:
    sock.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            return None
        if not chunk:
            return None
        buf += chunk
    line, _ = buf.split(b"\n", 1)
    try:
        return json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        return None


class ClusterNode:
    """
    One node in a 3-node cluster. Either primary (accepts clients, replicates)
    or secondary (replicates from primary; can become primary after election).
    """

    def __init__(
        self,
        node_id: int,
        nodes: Dict[int, Tuple[str, int]],
        initial_primary_id: int,
        data_dir: str,
    ):
        self.node_id = node_id
        self.nodes = dict(nodes)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._store = KVStore(str(self.data_dir))
        self._role = Role.PRIMARY if node_id == initial_primary_id else Role.SECONDARY
        self._primary_id = initial_primary_id
        self._lock = threading.RLock()
        self._server_socket: Optional[socket.socket] = None
        self._running = False

        # Primary: list of (node_id, socket) for connected secondaries
        self._replication_clients: List[Tuple[int, socket.socket]] = []
        self._replication_lock = threading.Lock()

        # Secondary: connection to primary and heartbeat
        self._primary_socket: Optional[socket.socket] = None
        self._primary_socket_lock = threading.Lock()
        self._last_heartbeat = 0.0  # 0 = never received heartbeat
        self._ever_connected_to_primary = False
        self._heartbeat_interval = 1.0
        self._heartbeat_timeout = 3.0
        self._election_thread: Optional[threading.Thread] = None
        self._election_lock = threading.Lock()

    def _my_host_port(self) -> Tuple[str, int]:
        return self.nodes[self.node_id]

    @property
    def role(self) -> Role:
        with self._lock:
            return self._role

    @property
    def primary_id(self) -> int:
        with self._lock:
            return self._primary_id

    def _handle_kv_request(self, request: dict) -> dict:
        """Handle a single KV request (only valid when primary)."""
        op = request.get("op")
        try:
            if op == "set":
                self._store.set(request["key"], request["value"])
                self._replicate_op("set", key=request["key"], value=request["value"])
                return {"status": "ok"}
            elif op == "get":
                value = self._store.get(request["key"])
                if value is not None:
                    return {"status": "ok", "value": value}
                return {"status": "not_found"}
            elif op == "delete":
                if self._store.delete(request["key"]):
                    self._replicate_op("delete", key=request["key"])
                    return {"status": "ok"}
                return {"status": "not_found"}
            elif op == "bulk_set":
                items = [(x[0], x[1]) for x in request["items"]]
                self._store.bulk_set(items)
                self._replicate_op("bulk_set", items=request["items"])
                return {"status": "ok", "count": len(items)}
            elif op == "ping":
                return {"status": "ok", "message": "pong"}
            elif op == "stats":
                return {
                    "status": "ok",
                    "keys": self._store.size(),
                    "wal_size": self._store.wal.get_size(),
                    "role": self._role.value,
                    "primary_id": self._primary_id,
                }
            elif op == "search_by_value":
                value = request.get("value")
                keys = self._store.indexes.search_by_value(value)
                return {"status": "ok", "keys": keys, "count": len(keys)}
            elif op == "fulltext_search":
                query = request.get("query", "")
                keys = self._store.indexes.fulltext_search(query)
                return {"status": "ok", "keys": keys, "count": len(keys)}
            elif op == "semantic_search":
                query = request.get("query", "")
                k = request.get("k", 10)
                threshold = request.get("threshold", 0.0)
                if self._store.embedding_index is None:
                    return {"status": "error", "message": "Embedding index not available (install sentence-transformers)"}
                try:
                    from embedding_index import EmbeddingUnavailableError
                    results = self._store.embedding_index.semantic_search(query, k=int(k), threshold=float(threshold))
                    return {"status": "ok", "results": [{"key": key, "score": score} for key, score in results], "count": len(results)}
                except EmbeddingUnavailableError as e:
                    return {"status": "error", "message": str(e)}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": f"Unknown operation: {op}"}
        except KeyError as e:
            return {"status": "error", "message": f"Missing field: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _replicate_op(self, op: str, **kwargs) -> None:
        """Send replication message to all connected secondaries and wait for ack."""
        msg = {"type": "replicate", "op": op, **kwargs}
        with self._replication_lock:
            to_remove = []
            for node_id, sock in self._replication_clients:
                try:
                    sock.settimeout(2.0)
                    _send_msg(sock, msg)
                    ack = _recv_msg(sock, timeout=2.0)
                    if ack is None or ack.get("type") != "ack":
                        to_remove.append(node_id)
                except Exception:
                    to_remove.append(node_id)
            for nid in to_remove:
                self._replication_clients = [(n, s) for n, s in self._replication_clients if n != nid]

    def _handle_connection(self, client_socket: socket.socket, address: Tuple[str, int]) -> None:
        """First message identifies connection type: client, replication, or election."""
        try:
            client_socket.settimeout(5.0)
            first = _recv_msg(client_socket, timeout=5.0)
            if first is None:
                client_socket.close()
                return
        except Exception:
            client_socket.close()
            return

        msg_type = first.get("type")

        if msg_type == "replication":
            # A secondary is connecting to us (we are primary)
            node_id = first.get("node_id")
            if node_id is None or self._role != Role.PRIMARY:
                client_socket.close()
                return
            self._handle_replication_join(client_socket, node_id)
            return

        if msg_type == "election":
            # Another secondary is asking for our vote
            candidate_id = first.get("candidate_id")
            if candidate_id is not None and self._role == Role.SECONDARY:
                _send_msg(client_socket, {"type": "vote", "vote": candidate_id})
            client_socket.close()
            return

        if msg_type == "new_primary":
            # New primary is telling us to reconnect to them
            new_primary_id = first.get("primary_id")
            if new_primary_id is not None:
                with self._lock:
                    self._primary_id = new_primary_id
                with self._primary_socket_lock:
                    if self._primary_socket:
                        try:
                            self._primary_socket.close()
                        except Exception:
                            pass
                        self._primary_socket = None
                # _connect_to_primary loop will reconnect to self.nodes[self._primary_id]
            client_socket.close()
            return

        # Treat as KV client request
        if self._role != Role.PRIMARY:
            host, port = self.nodes.get(self._primary_id, ("", 0))
            _send_msg(client_socket, {
                "status": "error",
                "message": "Not primary",
                "redirect_host": host,
                "redirect_port": port,
            })
            client_socket.close()
            return

        # Primary: handle first request then loop
        try:
            response = self._handle_kv_request(first)
            _send_msg(client_socket, response)
            # Loop for more requests
            while self._running:
                msg = _recv_msg(client_socket, timeout=1.0)
                if msg is None:
                    break
                if msg.get("type"):
                    break  # control message, ignore
                response = self._handle_kv_request(msg)
                _send_msg(client_socket, response)
        except Exception:
            pass
        finally:
            client_socket.close()

    def _handle_replication_join(self, sock: socket.socket, node_id: int) -> None:
        """Send full snapshot to new secondary then add to replication list."""
        try:
            snapshot = self._store.get_snapshot()
            _send_msg(sock, {"type": "snapshot", "data": snapshot})
            ack = _recv_msg(sock, timeout=5.0)
            if ack and ack.get("type") == "ack":
                with self._replication_lock:
                    self._replication_clients.append((node_id, sock))
                    print(f"[Cluster] Secondary node {node_id} joined replication")
                    return
        except Exception as e:
            print(f"[Cluster] Replication join failed for node {node_id}: {e}")
        try:
            sock.close()
        except Exception:
            pass

    def _connect_to_primary(self, host: str, port: int) -> None:
        """Connect to primary as replication client and process stream."""
        while self._running and self._role == Role.SECONDARY:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(10.0)
                sock.connect((host, port))
                _send_msg(sock, {"type": "replication", "node_id": self.node_id})
                with self._primary_socket_lock:
                    self._primary_socket = sock
                self._ever_connected_to_primary = True
                self._last_heartbeat = time.monotonic()
                # Process replication stream
                while self._running and self._role == Role.SECONDARY:
                    msg = _recv_msg(sock, timeout=self._heartbeat_timeout)
                    if msg is None:
                        break
                    if msg.get("type") == "snapshot":
                        data = msg.get("data", {})
                        self._store.apply_snapshot(data)
                        _send_msg(sock, {"type": "ack"})
                    elif msg.get("type") == "replicate":
                        op = msg.get("op")
                        if op == "set":
                            self._store.set(msg["key"], msg["value"])
                        elif op == "delete":
                            self._store.delete(msg["key"])
                        elif op == "bulk_set":
                            items = [(x[0], x[1]) for x in msg.get("items", [])]
                            self._store.bulk_set(items)
                        _send_msg(sock, {"type": "ack"})
                    elif msg.get("type") == "heartbeat":
                        self._last_heartbeat = time.monotonic()
                        _send_msg(sock, {"type": "ack"})
            except Exception as e:
                if self._running:
                    print(f"[Cluster] Lost connection to primary: {e}")
            finally:
                with self._primary_socket_lock:
                    if self._primary_socket == sock:
                        self._primary_socket = None
                    try:
                        sock.close()
                    except Exception:
                        pass
            if self._running and self._role == Role.SECONDARY:
                time.sleep(0.5)

    def _heartbeat_loop(self) -> None:
        """Primary: send heartbeat to all secondaries periodically."""
        while self._running and self._role == Role.PRIMARY:
            time.sleep(self._heartbeat_interval)
            with self._replication_lock:
                to_remove = []
                for node_id, sock in self._replication_clients:
                    try:
                        sock.settimeout(1.0)
                        _send_msg(sock, {"type": "heartbeat"})
                        ack = _recv_msg(sock, timeout=1.0)
                        if ack is None or ack.get("type") != "ack":
                            to_remove.append(node_id)
                    except Exception:
                        to_remove.append(node_id)
                for nid in to_remove:
                    self._replication_clients = [(n, s) for n, s in self._replication_clients if n != nid]

    def _election_loop(self) -> None:
        """Secondary: if primary heartbeat times out (after we had a connection), run election."""
        while self._running:
            time.sleep(0.5)
            with self._lock:
                if self._role != Role.SECONDARY:
                    continue
            with self._primary_socket_lock:
                has_primary = self._primary_socket is not None
            if has_primary and self._last_heartbeat > 0:
                if time.monotonic() - self._last_heartbeat > self._heartbeat_timeout:
                    with self._primary_socket_lock:
                        if self._primary_socket:
                            try:
                                self._primary_socket.close()
                            except Exception:
                                pass
                            self._primary_socket = None
                    has_primary = False
            if not has_primary and self._role == Role.SECONDARY and self._ever_connected_to_primary:
                with self._election_lock:
                    if self._role != Role.SECONDARY:
                        continue
                    # Run election: we want to become primary
                    other_ids = [nid for nid in self.nodes if nid != self.node_id and nid != self._primary_id]
                    votes = 0
                    for other_id in other_ids:
                        host, port = self.nodes[other_id]
                        try:
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.settimeout(2.0)
                            s.connect((host, port))
                            _send_msg(s, {"type": "election", "candidate_id": self.node_id})
                            r = _recv_msg(s, timeout=2.0)
                            s.close()
                            if r and r.get("type") == "vote" and r.get("vote") == self.node_id:
                                votes += 1
                        except Exception:
                            pass
                    if votes >= 1:  # At least one other node voted for us (2 nodes total + dead primary)
                        with self._lock:
                            self._role = Role.PRIMARY
                            self._primary_id = self.node_id
                        print(f"[Cluster] Node {self.node_id} became PRIMARY (election)")
                        # Tell the other secondary to connect to us
                        my_host, my_port = self._my_host_port()
                        for other_id in other_ids:
                            try:
                                host, port = self.nodes[other_id]
                                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                s.settimeout(2.0)
                                s.connect((host, port))
                                _send_msg(s, {"type": "new_primary", "primary_id": self.node_id, "host": my_host, "port": my_port})
                                s.close()
                            except Exception as e:
                                print(f"[Cluster] Failed to notify node {other_id}: {e}")
                        # Connect to ourselves? No. Other secondary will connect to us.
                        return

    def start(self) -> None:
        """Start the node (blocking)."""
        host, port = self._my_host_port()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((host, port))
        self._server_socket.listen(128)
        self._server_socket.settimeout(1.0)
        self._running = True

        print(f"[Cluster] Node {self.node_id} listening on {host}:{port} (role={self._role.value})")

        if self._role == Role.SECONDARY:
            primary_id = self._primary_id
            phost, pport = self.nodes[primary_id]
            self._election_thread = threading.Thread(target=self._election_loop, daemon=True)
            self._election_thread.start()
            conn = threading.Thread(target=self._connect_to_primary, args=(phost, pport), daemon=True)
            conn.start()
        else:
            hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
            hb.start()

        try:
            while self._running:
                try:
                    client_socket, address = self._server_socket.accept()
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    t = threading.Thread(target=self._handle_connection, args=(client_socket, address), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
        except Exception as e:
            if self._running:
                print(f"[Cluster] Error: {e}")

    def stop(self) -> None:
        """Stop the node."""
        print(f"[Cluster] Node {self.node_id} stopping...")
        self._running = False
        with self._replication_lock:
            for _, sock in self._replication_clients:
                try:
                    sock.close()
                except Exception:
                    pass
            self._replication_clients.clear()
        with self._primary_socket_lock:
            if self._primary_socket:
                try:
                    self._primary_socket.close()
                except Exception:
                    pass
                self._primary_socket = None
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        self._store.close()
        print(f"[Cluster] Node {self.node_id} stopped")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="KV Store Cluster Node")
    parser.add_argument("--node-id", type=int, required=True, help="This node's ID (0, 1, or 2)")
    parser.add_argument("--host", default="localhost", help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 9010+node_id)")
    parser.add_argument("--nodes", default="localhost:9010,localhost:9011,localhost:9012",
                        help="Comma-separated host:port for nodes 0,1,2")
    parser.add_argument("--primary", type=int, default=0, help="Initial primary node ID")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    args = parser.parse_args()

    nodes = {}
    for i, part in enumerate(args.nodes.split(",")):
        part = part.strip()
        if ":" in part:
            h, p = part.rsplit(":", 1)
            nodes[i] = (h.strip(), int(p.strip()))
        else:
            nodes[i] = ("localhost", 9010 + i)
    if args.node_id not in nodes:
        nodes[args.node_id] = (args.host, args.port or (9010 + args.node_id))

    port = args.port or (9010 + args.node_id)
    nodes[args.node_id] = (args.host, port)

    data_dir = os.path.join(args.data_dir, f"node_{args.node_id}")
    node = ClusterNode(
        node_id=args.node_id,
        nodes=nodes,
        initial_primary_id=args.primary,
        data_dir=data_dir,
    )
    try:
        node.start()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()


if __name__ == "__main__":
    main()
