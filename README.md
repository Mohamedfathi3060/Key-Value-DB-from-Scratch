# Key-Value Store (KV Store)

A high-performance, persistent key-value store with ACID compliance, replication, and advanced search capabilities.

## Features

### Core Features
- **ACID Compliance**: Atomicity, Consistency, Isolation, Durability guarantees
- **Write-Ahead Logging (WAL)**: fsync on every write for 100% durability
- **Persistent Storage**: Automatic recovery from WAL after crashes
- **Thread-Safe Operations**: Safe concurrent access
- **TCP-based Communication**: JSON protocol over sockets

### Advanced Search Capabilities
- **Value-based Search**: Find keys by exact value match
- **Full-Text Search**: Token-based search within string values
- **Semantic Search**: Meaning-based search using embeddings (requires `sentence-transformers`)
- **Multi-index Support**: Value index and inverted index for efficient queries

### Cluster Mode (High Availability)
- **3-Node Cluster**: Primary + 2 secondaries configuration
- **Automatic Failover**: Leader election when primary fails
- **Synchronous Replication**: All writes replicated to secondaries
- **Data Consistency**: Strong consistency guarantees

### Performance Optimizations
- **Bulk Operations**: Atomic batch writes for efficiency
- **WAL Compaction**: Automatic log compaction to reduce disk usage
- **Connection Reuse**: Persistent client connections
- **Concurrent Client Handling**: Multi-threaded server architecture

## Quick Start

### Installation
```bash
# Clone or download the project
cd kvstore

# Install optional dependencies for semantic search
pip install sentence-transformers  # Optional: for semantic search
```

### Running the Server
```bash
# Basic single-node server
python run.py server

# Or run the server directly
python server.py --host localhost --port 9000 --data-dir data
```

### Running the Client
```python
from client import KVClient

# Basic usage
with KVClient(host="localhost", port=9000) as client:
    client.set("key1", "value1")
    value = client.get("key1")
    print(value)  # "value1"
```

## Usage Examples

### Basic Operations
```python
client = KVClient()

# Set values
client.set("name", "Alice")
client.set("config", {"debug": True, "max_conn": 100})

# Get values
name = client.get("name")
config = client.get("config")

# Delete keys
client.delete("temp_key")

# Bulk operations
items = [("user:1", "Alice"), ("user:2", "Bob"), ("user:3", "Charlie")]
client.bulk_set(items)
```

### Search Operations
```python
# Value-based search
keys_with_value = client.search_by_value("specific_value")

# Full-text search
keys_matching_query = client.fulltext_search("python database")

# Semantic search (requires sentence-transformers)
similar_keys = client.semantic_search("artificial intelligence", k=10, threshold=0.5)
```

### Server Statistics
```python
stats = client.stats()
print(f"Keys: {stats['keys']}, WAL Size: {stats['wal_size']} bytes")
```

## Cluster Mode

### Starting the Cluster
```bash
# Start 3-node cluster (primary + 2 secondaries)
python run.py cluster

# Or run individual nodes manually:
python cluster_server.py --node-id 0 --primary 0 --data-dir cluster_data/node_0
python cluster_server.py --node-id 1 --primary 0 --data-dir cluster_data/node_1  
python cluster_server.py --node-id 2 --primary 0 --data-dir cluster_data/node_2
```

### Cluster Features
- **Automatic Failover**: If primary node fails, secondaries elect a new primary
- **Client Redirection**: Clients automatically redirected to current primary
- **Data Sync**: Secondary nodes stay synchronized with primary
- **Heartbeat Monitoring**: Continuous health checking between nodes

## Benchmarking

### Performance Benchmarks
```bash
# Run all performance benchmarks (starts own server)
python run.py benchmark

# Run against existing server
python run.py benchmark --port 9000

# Run specific benchmarks
python benchmark.py --host localhost --port 9000
```

Benchmarks include:
- Single writes/reads with varying dataset sizes
- Bulk write operations
- Mixed read/write workloads
- Concurrent client performance
- Large value handling
- Durability crash tests

### ACID Compliance Tests
```bash
# Run ACID compliance benchmarks
python run.py benchmark-acid

# Run specific ACID tests
python benchmark_ACID.py --host localhost --port 9000
```

ACID tests verify:
- **Isolation**: Concurrent operations don't interfere
- **Atomicity**: Batch operations are all-or-nothing, even during crashes
- **Durability**: No data loss after server crashes

## Demo
```bash
# Run interactive demo
python run.py demo
```

The demo shows:
- Basic CRUD operations
- Bulk operations
- Search capabilities
- Persistence verification
- Server statistics

## Testing
```bash
# Run test suite
python run.py test

# Run specific test files
python -m unittest test_kvstore.py
```

## Configuration

### Server Options
```bash
python server.py --help
```
- `--host`: Bind address (default: localhost)
- `--port`: Port number (default: 9000)
- `--data-dir`: Data directory for persistence (default: data)

### Cluster Options
```bash
python cluster_server.py --help
```
- `--node-id`: Node identifier (0, 1, or 2)
- `--primary`: Initial primary node ID
- `--nodes`: Comma-separated host:port pairs for all nodes
- `--data-dir`: Node-specific data directory

### Client Configuration
```python
KVClient(
    host="localhost",      # Server hostname
    port=9000,            # Server port
    timeout=30.0          # Connection timeout in seconds
)
```

## Architecture

### Single-Node Architecture
```
Client → TCP Socket → KVServer → KVStore → WAL + Indexes
```

### Cluster Architecture
```
Clients → Primary Node → Replication → Secondary Nodes
                     ↓
              Failover Election
```

### Data Flow
1. Client sends JSON request over TCP
2. Server writes operation to WAL with fsync
3. Server updates in-memory store and indexes
4. Server sends response to client
5. (Cluster) Primary replicates to secondaries

## File Structure
```
kvstore/
├── client.py              # Client library
├── server.py              # Single-node server
├── cluster_server.py      # Cluster node implementation
├── embedding_index.py     # Semantic search with embeddings
├── run.py                 # Management script
├── benchmark.py           # Performance benchmarks
├── benchmark_ACID.py      # ACID compliance tests
└── README.md             # This file
```

## Requirements

### Required
- Python 3.7+
- Standard library only (no external dependencies for core functionality)

### Optional
- `sentence-transformers`: For semantic search capabilities
  ```bash
  pip install sentence-transformers
  ```

## Performance Characteristics

### Typical Performance (on modern hardware)
- **Single Writes**: 500-2,000 ops/sec (with fsync)
- **Single Reads**: 10,000-50,000 ops/sec
- **Bulk Writes**: 5,000-20,000 items/sec
- **Latency**: 1-10ms for most operations

### Durability Guarantee
- **100% Durability**: Every write is fsync'd to disk
- **Crash Recovery**: Automatic recovery from WAL
- **No Data Loss**: Acknowledged writes persist through crashes

## Troubleshooting

### Common Issues

**Connection Refused**
- Ensure server is running: `python run.py server`
- Check port availability: `netstat -an | grep 9000`

**Semantic Search Not Working**
- Install required package: `pip install sentence-transformers`
- Check server logs for embedding initialization errors

**Cluster Node Not Joining**
- Verify all nodes use same cluster configuration
- Check firewall/network connectivity between nodes
- Ensure initial primary is running first

**Performance Issues**
- Use bulk operations for multiple writes
- Consider WAL compaction threshold adjustments
- Monitor disk I/O performance

### Logging
Server outputs operational logs to console:
- Client connections/disconnections
- Cluster election events
- WAL compaction activities
- Error conditions

## Contributing

This is a demonstration key-value store implementation. For production use, consider:

1. Adding authentication/authorization
2. Implementing more sophisticated clustering
3. Adding backup/restore functionality
4. Implementing data compression
5. Adding monitoring/metrics collection

## License

This project is provided as a demonstration implementation. Feel free to use and modify for educational purposes.