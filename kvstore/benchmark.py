import os
import shutil
import sys
import time
import threading
import statistics
import subprocess
import random
from typing import List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import KVClient
from server import KVServer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    total_operations: int
    total_time: float
    ops_per_second: float
    latencies: List[float]
    
    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies) * 1000
    
    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies) * 1000
    
    @property
    def p95_latency_ms(self) -> float:
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[idx] * 1000
    
    @property
    def p99_latency_ms(self) -> float:
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[idx] * 1000
    
    def __str__(self) -> str:
        return f"""
Benchmark: {self.name}
{'=' * 50}
Total Operations: {self.total_operations:,}
Total Time:       {self.total_time:.3f}s
Throughput:       {self.ops_per_second:,.0f} ops/sec

Latency:
  Average:        {self.avg_latency_ms:.3f} ms
  P50:            {self.p50_latency_ms:.3f} ms
  P95:            {self.p95_latency_ms:.3f} ms
  P99:            {self.p99_latency_ms:.3f} ms
"""


@dataclass
class DurabilityResult:
    """Result of a durability crash test."""
    total_acknowledged: int
    lost_after_restart: int

    @property
    def loss_rate(self) -> float:
        if self.total_acknowledged == 0:
            return 0.0
        return self.lost_after_restart / self.total_acknowledged

    def __str__(self) -> str:
        return (
            "Durability Crash Test\n"
            + "=" * 50 + "\n"
            + f"Acknowledged writes: {self.total_acknowledged}\n"
            + f"Lost after restart: {self.lost_after_restart}\n"
            + f"Loss rate: {self.loss_rate * 100:.4f}%"
        )


class Benchmark:
    """Benchmark runner for the KV Store."""
    
    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
    
    def _run_operations(self, client: KVClient, operations: List[Tuple]) -> List[float]:
        """Run operations and return latencies."""
        latencies = []
        for op in operations:
            start = time.perf_counter()
            if op[0] == "set":
                client.set(op[1], op[2])
            elif op[0] == "get":
                client.get(op[1])
            elif op[0] == "delete":
                client.delete(op[1])
            latencies.append(time.perf_counter() - start)
        return latencies
    
    def benchmark_single_writes(self, num_operations: int = 1000, prepopulate: int = 0) -> BenchmarkResult:
        """
        Benchmark single SET operations with fsync (100% durability).

        Optionally pre-populates the store with additional keys before
        measuring throughput so we can see how performance changes as
        the dataset grows.
        """
        client = KVClient(self.host, self.port)

        # Optionally pre-populate the store to simulate a larger dataset
        if prepopulate > 0:
            print(f"  Pre-populating {prepopulate} keys before write benchmark...")
            for i in range(prepopulate):
                client.set(f"bench_prepop_{prepopulate}_{i}", f"value_{i}" * 5)
        
        # Prepare operations we actually time
        operations = [("set", f"bench_write_{prepopulate}_{i}", f"value_{i}" * 10) for i in range(num_operations)]
        
        # Warm up (does not count towards measured throughput)
        for i in range(min(100, num_operations // 10)):
            client.set(f"warmup_{prepopulate}_{i}", "warmup")
        
        # Run benchmark
        start = time.perf_counter()
        latencies = self._run_operations(client, operations)
        total_time = time.perf_counter() - start
        
        client.close()
        
        dataset_label = "empty dataset" if prepopulate == 0 else f"+{prepopulate} pre-populated keys"
        return BenchmarkResult(
            name=f"Single Writes ({dataset_label})",
            total_operations=num_operations,
            total_time=total_time,
            ops_per_second=num_operations / total_time,
            latencies=latencies,
        )
    
    def benchmark_single_reads(self, num_operations: int = 10000) -> BenchmarkResult:
        """Benchmark single GET operations."""
        client = KVClient(self.host, self.port)
        
        # Pre-populate data
        print("  Pre-populating data for read benchmark...")
        for i in range(num_operations):
            client.set(f"bench_read_{i}", f"value_{i}" * 10)
        
        # Prepare operations
        operations = [("get", f"bench_read_{i}", None) for i in range(num_operations)]
        
        # Run benchmark
        start = time.perf_counter()
        latencies = self._run_operations(client, operations)
        total_time = time.perf_counter() - start
        
        client.close()
        
        return BenchmarkResult(
            name="Single Reads",
            total_operations=num_operations,
            total_time=total_time,
            ops_per_second=num_operations / total_time,
            latencies=latencies
        )
    
    def benchmark_bulk_writes(self, num_batches: int = 100, batch_size: int = 100) -> BenchmarkResult:
        """Benchmark bulk SET operations."""
        client = KVClient(self.host, self.port)
        
        total_operations = num_batches * batch_size
        latencies = []
        
        # Run benchmark
        start = time.perf_counter()
        for batch_num in range(num_batches):
            items = [(f"bench_bulk_{batch_num}_{i}", f"value_{i}" * 10) for i in range(batch_size)]
            
            batch_start = time.perf_counter()
            client.bulk_set(items)
            latencies.append(time.perf_counter() - batch_start)
        
        total_time = time.perf_counter() - start
        
        client.close()
        
        return BenchmarkResult(
            name=f"Bulk Writes (batch_size={batch_size})",
            total_operations=total_operations,
            total_time=total_time,
            ops_per_second=total_operations / total_time,
            latencies=latencies  # Note: latencies are per batch, not per item
        )
    
    def benchmark_mixed_workload(self, num_operations: int = 5000, 
                                  read_ratio: float = 0.8) -> BenchmarkResult:
        """Benchmark mixed read/write workload."""
        
        client = KVClient(self.host, self.port)
        
        # Pre-populate some data
        print("  Pre-populating data for mixed workload...")
        for i in range(1000):
            client.set(f"bench_mixed_{i}", f"value_{i}")
        
        # Generate mixed operations
        operations = []
        for i in range(num_operations):
            if random.random() < read_ratio:
                key = f"bench_mixed_{random.randint(0, 999)}"
                operations.append(("get", key, None))
            else:
                key = f"bench_mixed_{random.randint(0, 999)}"
                operations.append(("set", key, f"updated_{i}"))
        
        # Run benchmark
        start = time.perf_counter()
        latencies = self._run_operations(client, operations)
        total_time = time.perf_counter() - start
        
        client.close()
        
        return BenchmarkResult(
            name=f"Mixed Workload ({int(read_ratio*100)}% reads)",
            total_operations=num_operations,
            total_time=total_time,
            ops_per_second=num_operations / total_time,
            latencies=latencies
        )
    
    def benchmark_concurrent_writes(self, num_threads: int = 4, 
                                     writes_per_thread: int = 250) -> BenchmarkResult:
        """Benchmark concurrent writes from multiple clients."""
        all_latencies = []
        latency_lock = threading.Lock()
        
        def writer(thread_id: int):
            client = KVClient(self.host, self.port)
            thread_latencies = []
            
            for i in range(writes_per_thread):
                start = time.perf_counter()
                client.set(f"bench_concurrent_{thread_id}_{i}", f"value_{i}" * 10)
                thread_latencies.append(time.perf_counter() - start)
            
            client.close()
            
            with latency_lock:
                all_latencies.extend(thread_latencies)
        
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.perf_counter() - start
        
        total_operations = num_threads * writes_per_thread
        
        return BenchmarkResult(
            name=f"Concurrent Writes ({num_threads} threads)",
            total_operations=total_operations,
            total_time=total_time,
            ops_per_second=total_operations / total_time,
            latencies=all_latencies
        )
    
    def benchmark_large_values(self, num_operations: int = 100, 
                                value_size_kb: int = 10) -> BenchmarkResult:
        """Benchmark writes with large values."""
        client = KVClient(self.host, self.port)
        
        # Create large value
        large_value = "x" * (value_size_kb * 1024)
        
        # Prepare operations
        operations = [("set", f"bench_large_{i}", large_value) for i in range(num_operations)]
        
        # Run benchmark
        start = time.perf_counter()
        latencies = self._run_operations(client, operations)
        total_time = time.perf_counter() - start
        
        client.close()
        
        return BenchmarkResult(
            name=f"Large Value Writes ({value_size_kb}KB values)",
            total_operations=num_operations,
            total_time=total_time,
            ops_per_second=num_operations / total_time,
            latencies=latencies
        )


def run_durability_crash_test() -> DurabilityResult:
    """
    Simulate a crash while writes are in-flight and measure durability.

    A writer thread continuously issues SET operations and records keys
    that were acknowledged by the server. A second thread randomly kills
    the server process. After restarting, we check which acknowledged
    keys are missing.
    """
    data_dir = "durability_bench_data"
    port = 9700

    # Clean up any previous run
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    server_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"),
        "--host",
        "localhost",
        "--port",
        str(port),
        "--data-dir",
        data_dir,
    ]

    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the server a moment to start
    time.sleep(0.5)

    acknowledged_keys: List[str] = []
    ack_lock = threading.Lock()
    stop_event = threading.Event()

    def writer():
        client = KVClient(host="localhost", port=port)
        i = 0
        try:
            while not stop_event.is_set():
                key = f"durability_key_{i}"
                value = f"value_{i}"
                try:
                    client.set(key, value)
                    with ack_lock:
                        acknowledged_keys.append(key)
                    i += 1
                except Exception:
                    # Most likely the server died mid-benchmark
                    break
        finally:
            client.close()

    def killer():
        # Wait for a random period, then kill the server process
        delay = random.uniform(0.5, 2.0)
        time.sleep(delay)
        if proc.poll() is None:
            proc.terminate()
        stop_event.set()

    writer_thread = threading.Thread(target=writer, daemon=True)
    killer_thread = threading.Thread(target=killer, daemon=True)

    writer_thread.start()
    killer_thread.start()

    writer_thread.join()
    proc.wait(timeout=5)

    # Restart server to recover from WAL
    proc2 = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    client = KVClient(host="localhost", port=port)

    with ack_lock:
        keys_to_check = list(acknowledged_keys)

    lost = 0
    for key in keys_to_check:
        try:
            if client.get(key) is None:
                lost += 1
        except Exception:
            # If something goes wrong while checking, treat as lost
            lost += 1

    client.close()
    if proc2.poll() is None:
        proc2.terminate()
        proc2.wait(timeout=5)

    time.sleep(0.2)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    result = DurabilityResult(
        total_acknowledged=len(keys_to_check),
        lost_after_restart=lost,
    )

    print("\n" + "=" * 60)
    print("DURABILITY CRASH TEST")
    print("=" * 60)
    print(result)

    return result


def run_all_benchmarks(host: str = "localhost", port: int = 9000):
    """Run all benchmarks and print results."""
    bench = Benchmark(host, port)
    
    print("\n" + "=" * 60)
    print("KEY-VALUE STORE BENCHMARKS")
    print("Optimized for 100% durability (fsync on every write)")
    print("=" * 60)
    
    # Verify connection
    client = KVClient(host, port)
    if not client.ping():
        print(f"ERROR: Cannot connect to server at {host}:{port}")
        return
    client.close()
    print(f"\nConnected to server at {host}:{port}")
    
    results = []
    
    # Write throughput as dataset grows
    print("\nRunning: Single Writes (empty dataset)...")
    results.append(bench.benchmark_single_writes(1000, prepopulate=0))
    
    print("Running: Single Writes (+10k pre-populated keys)...")
    results.append(bench.benchmark_single_writes(1000, prepopulate=10_000))
    
    print("Running: Single Writes (+50k pre-populated keys)...")
    results.append(bench.benchmark_single_writes(1000, prepopulate=50_000))
    
    print("Running: Single Reads...")
    results.append(bench.benchmark_single_reads(5000))
    
    print("Running: Bulk Writes...")
    results.append(bench.benchmark_bulk_writes(100, 100))
    
    print("Running: Mixed Workload...")
    results.append(bench.benchmark_mixed_workload(3000))
    
    print("Running: Concurrent Writes...")
    results.append(bench.benchmark_concurrent_writes(4, 250))
    
    print("Running: Large Value Writes...")
    results.append(bench.benchmark_large_values(50, 10))
    
    # Print all results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for result in results:
        print(result)
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<40} {'Ops/sec':>12} {'P99 (ms)':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r.name:<40} {r.ops_per_second:>12,.0f} {r.p99_latency_ms:>10.3f}")

    # Durability crash test (prints its own results)
    run_durability_crash_test()


def start_benchmark_server():
    """Start a server for benchmarking."""
    test_dir = "benchmark_data"
    port = 9500
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    server = KVServer(port=port, data_dir=test_dir)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    
    return server, port, test_dir


def main():
    """Main entry point for benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KV Store Benchmarks")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--standalone", action="store_true", 
                        help="Start a standalone server for benchmarks")
    args = parser.parse_args()
    
    if args.standalone:
        print("Starting standalone benchmark server...")
        server, port, test_dir = start_benchmark_server()
        try:
            run_all_benchmarks("localhost", port)
        finally:
            server.stop()
            time.sleep(0.3)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    else:
        port = args.port or 9000
        run_all_benchmarks(args.host, port)


if __name__ == "__main__":
    main()