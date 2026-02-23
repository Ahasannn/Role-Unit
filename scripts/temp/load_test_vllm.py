#!/usr/bin/env python3
"""
vLLM Load Test: Send 4000 concurrent dummy requests across 5 models.
Monitors running/waiting queue depths and KV cache during the test.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# ── Model Configuration ──────────────────────────────────────────────────────
MODELS = [
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "short": "DeepSeek-32B",
        "port": 8001,
        "size": "32B",
    },
    {
        "name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "short": "Mistral-24B",
        "port": 8002,
        "size": "24B",
    },
    {
        "name": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "short": "Qwen-14B",
        "port": 8003,
        "size": "14B",
    },
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "Llama-8B",
        "port": 8004,
        "size": "8B",
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "short": "Llama-3B",
        "port": 8005,
        "size": "3B",
    },
]

API_KEY = "dummy-key-for-vllm"
TOTAL_REQUESTS = 4000
REQUESTS_PER_MODEL = TOTAL_REQUESTS // len(MODELS)  # 800 each

# ── Large prompt to blow out KV cache ────────────────────────────────────────
# max_model_len = 4096 tokens.  We want ~3000-3500 input tokens so that
# input + output ≈ 4096.  Rough rule: 1 token ≈ 3.5-4 chars for English.
# We build ~13 000 chars → ~3 300 tokens, then request 512 output tokens.

_PARAGRAPH = (
    "The infrastructure-aware multi-agent orchestration system continuously "
    "monitors the state of every deployed language model server in real time. "
    "It collects queue depth, KV cache utilization percentage, time to first "
    "token latency, inter-token latency, and end-to-end request latency from "
    "every vLLM endpoint through their Prometheus metrics interface. These "
    "observations form a high-dimensional state vector that is fed into a "
    "hierarchical Constrained Markov Decision Process. The planner network "
    "selects a collaboration topology and role assignment at the beginning of "
    "each task, while the executor network dynamically chooses which specific "
    "language model and reasoning strategy to assign to each role during "
    "execution. Under low system load the orchestrator favors larger models "
    "with deep chain-of-thought reasoning to maximize accuracy. Under high "
    "system load it gracefully degrades to smaller, faster models with concise "
    "reasoning strategies to maintain acceptable latency and avoid queue "
    "buildup. The reward function balances accuracy, cost, and latency through "
    "a multi-objective formulation with explicit budget and latency constraints "
    "enforced via Lagrangian relaxation. This adaptive behavior is what "
    "distinguishes the system from traditional static routing approaches that "
    "only consider task characteristics and ignore infrastructure state. "
)

# Repeat until we reach ~13000 chars ≈ ~3300 tokens
_BIG_PROMPT_BODY = ""
_counter = 0
while len(_BIG_PROMPT_BODY) < 13000:
    _counter += 1
    _BIG_PROMPT_BODY += f"[Block {_counter}] {_PARAGRAPH} "

BIG_PROMPT = (
    "Below is a long technical document. Read it carefully and then write a "
    "detailed multi-paragraph summary covering every major point.\n\n"
    + _BIG_PROMPT_BODY
    + "\n\nNow provide your comprehensive summary."
)

print(f"[Config] Prompt length: {len(BIG_PROMPT)} chars (~{len(BIG_PROMPT)//4} tokens est.)")

DUMMY_PAYLOAD = {
    "messages": [{"role": "user", "content": BIG_PROMPT}],
    "max_tokens": 512,
    "temperature": 0.0,
}

# Metrics polling interval (seconds)
METRICS_POLL_INTERVAL = 0.5


# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class ModelStats:
    """Tracks per-model statistics during the load test."""

    name: str
    short: str
    port: int
    size: str
    # Request outcomes
    total_sent: int = 0
    success: int = 0
    failed: int = 0
    errors: list = field(default_factory=list)
    # Latencies (seconds)
    latencies: list = field(default_factory=list)
    # Peak metrics observed from /metrics endpoint
    peak_running: float = 0
    peak_waiting: float = 0
    peak_kv_cache: float = 0
    # Time-series snapshots: list of (timestamp, running, waiting, kv_cache)
    metric_snapshots: list = field(default_factory=list)


# ── Metrics Polling ──────────────────────────────────────────────────────────
def parse_prometheus_metric(text: str, metric_substr: str) -> float:
    """Extract a numeric value from Prometheus text format by substring match."""
    for line in text.splitlines():
        if metric_substr in line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return 0.0


async def poll_metrics(
    session: aiohttp.ClientSession,
    stats_map: dict[int, ModelStats],
    stop_event: asyncio.Event,
    t0: float,
):
    """Continuously poll /metrics on each model until stop_event is set."""
    while not stop_event.is_set():
        for port, stats in stats_map.items():
            url = f"http://127.0.0.1:{port}/metrics"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    running = parse_prometheus_metric(text, "num_requests_running")
                    waiting = parse_prometheus_metric(text, "num_requests_waiting")
                    kv_cache = parse_prometheus_metric(text, "kv_cache_usage_perc")

                    stats.peak_running = max(stats.peak_running, running)
                    stats.peak_waiting = max(stats.peak_waiting, waiting)
                    stats.peak_kv_cache = max(stats.peak_kv_cache, kv_cache)

                    stats.metric_snapshots.append(
                        (time.time() - t0, running, waiting, kv_cache)
                    )
            except Exception:
                pass
        await asyncio.sleep(METRICS_POLL_INTERVAL)


# ── Request Sender ───────────────────────────────────────────────────────────
async def send_request(
    session: aiohttp.ClientSession,
    model: dict,
    stats: ModelStats,
    sem: asyncio.Semaphore,
):
    """Send a single dummy request to a vLLM endpoint."""
    url = f"http://127.0.0.1:{model['port']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {**DUMMY_PAYLOAD, "model": model["name"]}

    async with sem:
        stats.total_sent += 1
        t_start = time.time()
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=1800),
            ) as resp:
                body = await resp.text()
                elapsed = time.time() - t_start
                if resp.status == 200:
                    stats.success += 1
                    stats.latencies.append(elapsed)
                else:
                    stats.failed += 1
                    # Truncate error body
                    stats.errors.append(
                        f"HTTP {resp.status}: {body[:200]}"
                    )
        except Exception as e:
            elapsed = time.time() - t_start
            stats.failed += 1
            stats.errors.append(f"{type(e).__name__}: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 80)
    print("  vLLM LOAD TEST — 4000 Concurrent Dummy Requests (800 per model)")
    print("=" * 80)
    print()

    # Verify all models are healthy first
    print("[1/4] Checking model health...")
    async with aiohttp.ClientSession() as session:
        for m in MODELS:
            url = f"http://127.0.0.1:{m['port']}/health"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"  ✓ {m['short']:20s} (port {m['port']}) — healthy")
                    else:
                        print(f"  ✗ {m['short']:20s} (port {m['port']}) — HTTP {resp.status}")
                        print("    Aborting. Not all models are healthy.")
                        return
            except Exception as e:
                print(f"  ✗ {m['short']:20s} (port {m['port']}) — {e}")
                print("    Aborting. Not all models are healthy.")
                return

    print()
    print(f"[2/4] Launching {TOTAL_REQUESTS} requests ({REQUESTS_PER_MODEL} per model)...")
    print(f"       Prompt size: {len(BIG_PROMPT)} chars (~{len(BIG_PROMPT)//4} tokens)")
    print(f"       Max output tokens: {DUMMY_PAYLOAD['max_tokens']}")
    print()

    # Initialize stats
    stats_map: dict[int, ModelStats] = {}
    for m in MODELS:
        stats_map[m["port"]] = ModelStats(
            name=m["name"], short=m["short"], port=m["port"], size=m["size"]
        )

    # Use a connector with a high connection limit
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        stop_event = asyncio.Event()
        t0 = time.time()

        # Start metrics poller
        poller_task = asyncio.create_task(
            poll_metrics(session, stats_map, stop_event, t0)
        )

        # Create all request tasks — fire them ALL at once
        # Use a semaphore to limit file descriptor exhaustion but keep it very high
        sem = asyncio.Semaphore(2000)
        tasks = []
        for m in MODELS:
            for _ in range(REQUESTS_PER_MODEL):
                tasks.append(
                    asyncio.create_task(
                        send_request(session, m, stats_map[m["port"]], sem)
                    )
                )

        # Progress reporting
        total_tasks = len(tasks)
        print(f"       {total_tasks} tasks created. Waiting for completion...")
        print()

        # Wait with periodic progress updates
        done_count = 0
        while done_count < total_tasks:
            done_set, _ = await asyncio.wait(
                tasks, timeout=5, return_when=asyncio.FIRST_EXCEPTION
            )
            done_count = sum(1 for t in tasks if t.done())
            elapsed = time.time() - t0
            # Print current queue status
            status_parts = []
            for m in MODELS:
                s = stats_map[m["port"]]
                if s.metric_snapshots:
                    last = s.metric_snapshots[-1]
                    status_parts.append(
                        f"{s.short}: R={int(last[1])}/W={int(last[2])}/KV={last[3]*100:.0f}%"
                    )
            queue_str = " | ".join(status_parts) if status_parts else "polling..."
            print(
                f"  [{elapsed:6.1f}s] {done_count}/{total_tasks} done | {queue_str}"
            )

        total_elapsed = time.time() - t0

        # Stop metrics poller
        stop_event.set()
        await poller_task

    # ── Results ──────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"[3/4] RESULTS — Total elapsed: {total_elapsed:.1f}s")
    print("=" * 80)
    print()

    # Per-model summary table
    header = (
        f"{'Model':<22s} {'Size':>5s} │ {'Sent':>5s} {'OK':>5s} {'Fail':>5s} │ "
        f"{'Peak Run':>8s} {'Peak Wait':>9s} {'Peak KV%':>8s} │ "
        f"{'Avg Lat':>8s} {'P50 Lat':>8s} {'P95 Lat':>8s} {'P99 Lat':>8s}"
    )
    print(header)
    print("─" * len(header))

    for m in MODELS:
        s = stats_map[m["port"]]
        lats = sorted(s.latencies) if s.latencies else [0]
        avg_lat = sum(lats) / len(lats) if lats else 0
        p50 = lats[int(len(lats) * 0.50)] if lats else 0
        p95 = lats[int(len(lats) * 0.95)] if lats else 0
        p99 = lats[int(len(lats) * 0.99)] if lats else 0

        print(
            f"{s.short:<22s} {s.size:>5s} │ "
            f"{s.total_sent:>5d} {s.success:>5d} {s.failed:>5d} │ "
            f"{s.peak_running:>8.0f} {s.peak_waiting:>9.0f} {s.peak_kv_cache*100:>7.1f}% │ "
            f"{avg_lat:>7.2f}s {p50:>7.2f}s {p95:>7.2f}s {p99:>7.2f}s"
        )

    print()

    # ── Detailed Analysis ────────────────────────────────────────────────────
    print("=" * 80)
    print("[4/4] ANALYSIS")
    print("=" * 80)
    print()

    for m in MODELS:
        s = stats_map[m["port"]]
        print(f"── {s.short} ({s.name}) ──")
        print(f"   Port:              {s.port}")
        print(f"   Requests Sent:     {s.total_sent}")
        print(f"   Successes:         {s.success}")
        print(f"   Failures:          {s.failed}")
        print(f"   Peak Running:      {s.peak_running:.0f}")
        print(f"   Peak Waiting:      {s.peak_waiting:.0f}")
        print(f"   Peak Queue Total:  {s.peak_running + s.peak_waiting:.0f} (running + waiting)")
        print(f"   Peak KV Cache:     {s.peak_kv_cache*100:.1f}%")

        if s.latencies:
            lats = sorted(s.latencies)
            print(f"   Latency Min:       {lats[0]:.3f}s")
            print(f"   Latency Avg:       {sum(lats)/len(lats):.3f}s")
            print(f"   Latency P50:       {lats[int(len(lats)*0.50)]:.3f}s")
            print(f"   Latency P95:       {lats[int(len(lats)*0.95)]:.3f}s")
            print(f"   Latency P99:       {lats[int(len(lats)*0.99)]:.3f}s")
            print(f"   Latency Max:       {lats[-1]:.3f}s")
            throughput = s.success / total_elapsed if total_elapsed > 0 else 0
            print(f"   Throughput:        {throughput:.1f} req/s")

        if s.errors:
            unique_errors = {}
            for e in s.errors:
                key = e[:80]
                unique_errors[key] = unique_errors.get(key, 0) + 1
            print(f"   Errors ({len(s.errors)} total):")
            for err, count in sorted(unique_errors.items(), key=lambda x: -x[1])[:5]:
                print(f"     [{count}x] {err}")

        # Queue depth timeline (sampled)
        if s.metric_snapshots:
            print(f"   Queue Depth Timeline ({len(s.metric_snapshots)} samples):")
            # Show a few key snapshots
            snapshots = s.metric_snapshots
            indices = [0]
            step = max(1, len(snapshots) // 8)
            for i in range(step, len(snapshots), step):
                indices.append(i)
            if (len(snapshots) - 1) not in indices:
                indices.append(len(snapshots) - 1)
            for idx in indices:
                t, r, w, kv = snapshots[idx]
                print(
                    f"     t={t:6.1f}s  running={int(r):4d}  waiting={int(w):5d}  "
                    f"kv_cache={kv*100:5.1f}%"
                )
        print()

    # ── Overall Summary ──────────────────────────────────────────────────────
    total_success = sum(s.success for s in stats_map.values())
    total_failed = sum(s.failed for s in stats_map.values())
    overall_throughput = total_success / total_elapsed if total_elapsed > 0 else 0

    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Total Requests:     {TOTAL_REQUESTS}")
    print(f"  Total Successes:    {total_success}")
    print(f"  Total Failures:     {total_failed}")
    print(f"  Total Elapsed:      {total_elapsed:.1f}s")
    print(f"  Overall Throughput: {overall_throughput:.1f} req/s")
    print()

    # Ranking by peak queue capacity
    print("  Peak Queue Depth Ranking (Running + Waiting):")
    ranked = sorted(stats_map.values(), key=lambda s: s.peak_running + s.peak_waiting, reverse=True)
    for i, s in enumerate(ranked, 1):
        total_q = s.peak_running + s.peak_waiting
        print(
            f"    {i}. {s.short:<20s} — Peak Running: {s.peak_running:>4.0f}, "
            f"Peak Waiting: {s.peak_waiting:>5.0f}, Total: {total_q:>5.0f}"
        )
    print()

    # Ranking by KV cache
    print("  Peak KV Cache Usage Ranking:")
    ranked_kv = sorted(stats_map.values(), key=lambda s: s.peak_kv_cache, reverse=True)
    for i, s in enumerate(ranked_kv, 1):
        print(f"    {i}. {s.short:<20s} — {s.peak_kv_cache*100:.1f}%")
    print()

    # Ranking by throughput
    print("  Throughput Ranking (req/s):")
    ranked_tp = sorted(
        stats_map.values(),
        key=lambda s: s.success / total_elapsed if total_elapsed > 0 else 0,
        reverse=True,
    )
    for i, s in enumerate(ranked_tp, 1):
        tp = s.success / total_elapsed if total_elapsed > 0 else 0
        print(f"    {i}. {s.short:<20s} — {tp:.1f} req/s ({s.success} completed)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
