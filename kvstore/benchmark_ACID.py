"""
ACID Benchmark for the Key-Value Store.

Tests:
1. Isolation: Concurrent bulk_set operations touching the same keys.
   Verifies that concurrent bulks do not corrupt each other (each key ends
   with a value from exactly one bulk).

2. Atomicity under crash: A single bulk_set is in-flight while the server
   is killed with SIGKILL (-9). After restart, the batch must be either
   completely applied or not applied at all (no partial batch).
"""

import os
import shutil
import sys
import time
import threading
import subprocess
import random
import signal
from typing import List, Tuple, Any
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import KVClient


def _kill_process_force(proc: subprocess.Popen) -> None:
    """Kill the process with SIGKILL (-9) on Unix; forceful terminate on Windows."""
    if proc.poll() is not None:
        return
    if hasattr(signal, "SIGKILL"):
        os.kill(proc.pid, signal.SIGKILL)
    else:
        # Windows: no SIGKILL; use Popen.kill() (TerminateProcess)
        proc.kill()


@dataclass
class IsolationResult:
    """Result of the concurrent bulk_set isolation test."""
    num_threads: int
    keys_per_bulk: int
    bulks_per_thread: int
    passed: bool
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "Isolation (concurrent bulk_set, same keys)",
            "=" * 50,
            f"Threads: {self.num_threads}, Keys per bulk: {self.keys_per_bulk}, Bulks per thread: {self.bulks_per_thread}",
            f"Result: {status}",
        ]
        for e in self.errors:
            lines.append(f"  - {e}")
        return "\n".join(lines)


@dataclass
class AtomicityResult:
    """Result of the bulk_set + kill atomicity test."""
    batch_size: int
    trials: int
    all_or_nothing_violations: int  # number of trials where batch was partial
    passed: bool
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "Atomicity (bulk_set + SIGKILL, all-or-nothing)",
            "=" * 50,
            f"Batch size: {self.batch_size}, Trials: {self.trials}",
            f"Partial-batch violations: {self.all_or_nothing_violations}",
            f"Result: {status}",
        ]
        for d in self.details:
            lines.append(f"  {d}")
        return "\n".join(lines)


def _start_server_subprocess(data_dir: str, port: int) -> subprocess.Popen:
    """Start the KV server as a subprocess. Returns the Popen instance."""
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    cmd = [
        sys.executable,
        server_script,
        "--host", "localhost",
        "--port", str(port),
        "--data-dir", data_dir,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.6)
    return proc


def run_isolation_test(host: str = "localhost", port: int = 9000,
                      num_threads: int = 4,
                      shared_keys_count: int = 20,
                      bulks_per_thread: int = 10,
                      keys_per_bulk: int = 20) -> IsolationResult:
    """
    Run concurrent bulk_set operations that touch the same keys.

    Multiple threads each perform several bulk_sets. Each bulk_set writes
    to a shared set of keys (same key names across threads) with
    thread-specific values. We verify that after all complete, every key
    has a value that is consistent (from one bulk only); no key may show
    mixed or corrupted state from two different bulks.
    """
    errors: List[str] = []
    # Shared key names (same keys written by all threads)
    shared_keys = [f"acid_shared_key_{i}" for i in range(shared_keys_count)]
    # We'll have each thread write bulks that include these keys with a unique
    # signature so we can verify no cross-thread corruption.
    results_per_key: List[Any] = [None] * len(shared_keys)
    results_lock = threading.Lock()

    def thread_worker(thread_id: int) -> None:
        client = KVClient(host=host, port=port)
        try:
            for b in range(bulks_per_thread):
                # Each bulk: same key set, value = (thread_id, batch_id, key_index)
                items = [
                    (k, {"thread": thread_id, "batch": b, "key": i})
                    for i, k in enumerate(shared_keys)
                ]
                client.bulk_set(items)
        except Exception as e:
            with results_lock:
                errors.append(f"Thread {thread_id}: {e}")
        finally:
            client.close()

    threads = [threading.Thread(target=thread_worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        return IsolationResult(
            num_threads=num_threads,
            keys_per_bulk=keys_per_bulk,
            bulks_per_thread=bulks_per_thread,
            passed=False,
            errors=errors,
        )

    # Verify: read each shared key; value must be a dict with thread, batch, key
    client = KVClient(host=host, port=port)
    try:
        for i, k in enumerate(shared_keys):
            val = client.get(k)
            if val is None:
                errors.append(f"Key {k} missing")
                continue
            if not isinstance(val, dict) or "thread" not in val or "batch" not in val:
                errors.append(f"Key {k} has invalid value: {val}")
                continue
            if val.get("key") != i:
                errors.append(f"Key {k} wrong key index: {val}")
        passed = len(errors) == 0
    finally:
        client.close()

    return IsolationResult(
        num_threads=num_threads,
        keys_per_bulk=keys_per_bulk,
        bulks_per_thread=bulks_per_thread,
        passed=passed,
        errors=errors,
    )


def run_atomicity_bulk_kill_test(port: int = 9700,
                                 batch_size: int = 50,
                                 trials: int = 15) -> AtomicityResult:
    """
    For each trial: start server, start a bulk_set, and kill the server
    at a random moment (SIGKILL / -9). Restart and check: the batch must
    be either fully present or fully absent (all-or-nothing).
    """
    data_dir = "acid_atomicity_data"
    details: List[str] = []
    violations = 0

    for trial in range(trials):
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        proc = _start_server_subprocess(data_dir, port)

        prefix = f"acid_atom_trial_{trial}"
        keys = [f"{prefix}_{i}" for i in range(batch_size)]
        values = [f"value_{trial}_{i}" for i in range(batch_size)]
        items = list(zip(keys, values))

        bulk_done = threading.Event()
        bulk_ack = threading.Event()
        kill_done = threading.Event()

        def do_bulk() -> None:
            try:
                client = KVClient(host="localhost", port=port)
                client.bulk_set(items)
                bulk_ack.set()
                client.close()
            except Exception:
                pass
            finally:
                bulk_done.set()

        def killer() -> None:
            delay = random.uniform(0.01, 0.25)
            time.sleep(delay)
            if proc.poll() is None:
                _kill_process_force(proc)
            kill_done.set()

        t_bulk = threading.Thread(target=do_bulk, daemon=True)
        t_kill = threading.Thread(target=killer, daemon=True)
        t_bulk.start()
        t_kill.start()
        t_bulk.join(timeout=5.0)
        t_kill.join(timeout=2.0)
        if proc.poll() is None:
            _kill_process_force(proc)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

        time.sleep(0.2)

        # Restart and count how many of the batch keys are present
        proc2 = _start_server_subprocess(data_dir, port)
        try:
            client = KVClient(host="localhost", port=port)
            found = 0
            for i, k in enumerate(keys):
                v = client.get(k)
                if v == values[i]:
                    found += 1
            client.close()

            if found != 0 and found != batch_size:
                violations += 1
                details.append(f"Trial {trial}: partial batch (found {found}/{batch_size})")
        finally:
            if proc2.poll() is None:
                proc2.terminate()
                try:
                    proc2.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc2.kill()

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    return AtomicityResult(
        batch_size=batch_size,
        trials=trials,
        all_or_nothing_violations=violations,
        passed=violations == 0,
        details=details if details else ["All trials: batch fully applied or fully absent."],
    )


def run_all_acid_benchmarks(host: str = "localhost", port: int = 9000,
                            run_isolation: bool = True,
                            run_atomicity_kill: bool = True) -> None:
    """Run all ACID benchmarks and print results."""
    print("\n" + "=" * 60)
    print("ACID BENCHMARKS")
    print("=" * 60)

    if run_isolation:
        print("\n--- Isolation: concurrent bulk_set (same keys) ---")
        # Use provided host/port (server must already be running, e.g. from run.py)
        client = KVClient(host=host, port=port)
        if not client.ping():
            print(f"ERROR: Cannot connect to server at {host}:{port}")
            client.close()
            return
        client.close()

        result = run_isolation_test(host=host, port=port,
                                    num_threads=4,
                                    shared_keys_count=20,
                                    bulks_per_thread=10,
                                    keys_per_bulk=20)
        print(result)

    if run_atomicity_kill:
        print("\n--- Atomicity: bulk_set + SIGKILL (-9), all-or-nothing ---")
        result = run_atomicity_bulk_kill_test(port=port if port != 9000 else 9700,
                                              batch_size=50,
                                              trials=15)
        print(result)

    print("\n" + "=" * 60)
    print("ACID BENCHMARKS DONE")
    print("=" * 60)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ACID benchmarks for KV Store")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=9000, help="Server port (for isolation); atomicity test uses its own process")
    parser.add_argument("--no-isolation", action="store_true", help="Skip isolation test")
    parser.add_argument("--no-atomicity", action="store_true", help="Skip atomicity kill test")
    args = parser.parse_args()

    run_all_acid_benchmarks(
        host=args.host,
        port=args.port,
        run_isolation=not args.no_isolation,
        run_atomicity_kill=not args.no_atomicity,
    )


if __name__ == "__main__":
    main()
