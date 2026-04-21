from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from sysadmin_env.sandbox import Sandbox
from sysadmin_env.tasks import hpc_outage


def run(iterations: int, verbose: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="hpc_bench_lower_") as lower_dir:
        lower = Path(lower_dir)
        hpc_outage.prepare_filesystem(lower)

        sandbox = Sandbox(
            lower,
            timeout=30.0,
            isolate_network=False,
            allow_nested_sandbox=True,
        )
        sandbox.create()
        try:
            latencies: list[float] = []
            for i in range(iterations):
                start = time.perf_counter()
                sandbox.reset()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed_ms)
                if verbose:
                    print(f"iter {i + 1:03d} {elapsed_ms:.3f} ms")
            summary = _summarize(latencies, sandbox.overlay.mount_type or "unknown")
        finally:
            sandbox.destroy()
    return summary


def _summarize(latencies: list[float], mount_type: str) -> dict:
    sorted_latencies = sorted(latencies)
    count = len(sorted_latencies)
    return {
        "count": count,
        "mount_type": mount_type,
        "min_ms": sorted_latencies[0],
        "p50_ms": statistics.median(sorted_latencies),
        "p95_ms": sorted_latencies[_pct_index(count, 0.95)],
        "p99_ms": sorted_latencies[_pct_index(count, 0.99)],
        "max_ms": sorted_latencies[-1],
        "mean_ms": statistics.fmean(sorted_latencies),
        "stdev_ms": statistics.pstdev(sorted_latencies),
    }


def _pct_index(count: int, quantile: float) -> int:
    idx = int(round(quantile * (count - 1)))
    return max(0, min(count - 1, idx))


def _print_report(summary: dict) -> None:
    print()
    print(f"mount_type  : {summary['mount_type']}")
    print(f"iterations  : {summary['count']}")
    print(f"min  ms     : {summary['min_ms']:.3f}")
    print(f"p50  ms     : {summary['p50_ms']:.3f}")
    print(f"p95  ms     : {summary['p95_ms']:.3f}")
    print(f"p99  ms     : {summary['p99_ms']:.3f}")
    print(f"max  ms     : {summary['max_ms']:.3f}")
    print(f"mean ms     : {summary['mean_ms']:.3f}")
    print(f"stdev ms    : {summary['stdev_ms']:.3f}")
    print()
    print("| mount | n | p50 ms | p95 ms | p99 ms | max ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {summary['mount_type']} | {summary['count']} | "
        f"{summary['p50_ms']:.2f} | {summary['p95_ms']:.2f} | "
        f"{summary['p99_ms']:.2f} | {summary['max_ms']:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--iterations", type=int, default=200)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    summary = run(args.iterations, args.verbose)
    _print_report(summary)


if __name__ == "__main__":
    main()
