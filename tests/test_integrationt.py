from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable
INFERENCE_PATH = PROJECT_ROOT / "inference.py"


def _read_rss_kb(pid: int) -> int:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return 0
    for line in status_path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return 0


def _read_cpu_ticks(pid: int) -> int:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return 0
    parts = stat_path.read_text().split()
    if len(parts) < 15:
        return 0
    return int(parts[13]) + int(parts[14])


def _wait_for_health(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = subprocess.run(
            [
                PYTHON_BIN,
                "-c",
                (
                    "import sys, urllib.request\n"
                    f"urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=2).read()\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return
        time.sleep(0.25)
    raise RuntimeError("server health check did not become ready")


def _run_full_cycle(port: int) -> dict[str, float | int | str]:
    env = os.environ.copy()
    env["SYSADMIN_ENV_SERVER_URL"] = f"ws://127.0.0.1:{port}/ws"
    env["SYSADMIN_ENV_HEALTHCHECK_URL"] = f"http://127.0.0.1:{port}/health"
    env["SYSADMIN_ENV_TASKS_URL"] = f"http://127.0.0.1:{port}/tasks"
    env["SYSADMIN_ENV_TASK_ID"] = ""
    env["SYSADMIN_ENV_DOTENV_PATH"] = str(Path(tempfile.gettempdir()) / f"sysadmin_env_missing_dotenv_{port}")

    server = subprocess.Popen(
        [
            PYTHON_BIN,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
    )

    peak_rss_kb = 0
    cpu_start = _read_cpu_ticks(server.pid)
    wall_start = time.perf_counter()

    try:
        _wait_for_health(port)

        inference = subprocess.Popen(
            [PYTHON_BIN, str(INFERENCE_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
        )

        while inference.poll() is None:
            peak_rss_kb = max(peak_rss_kb, _read_rss_kb(server.pid), _read_rss_kb(inference.pid))
            time.sleep(0.05)

        peak_rss_kb = max(peak_rss_kb, _read_rss_kb(server.pid))
        inference_output = inference.stdout.read() if inference.stdout is not None else ""
        if inference.returncode != 0:
            raise RuntimeError(inference_output)

        wall_elapsed = time.perf_counter() - wall_start
        cpu_end = _read_cpu_ticks(server.pid)
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cpu_seconds = max(cpu_end - cpu_start, 0) / ticks_per_second

        return {
            "output": inference_output,
            "wall_seconds": wall_elapsed,
            "peak_rss_kb": peak_rss_kb,
            "cpu_seconds": cpu_seconds,
        }
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def test_three_consecutive_clean_runs_stay_within_phase_eight_envelope():
    runs = []
    for index in range(3):
        port = 8200 + index
        run = _run_full_cycle(port)
        runs.append(run)

        output = str(run["output"])
        assert output.count("[START]") == 3
        assert output.count("[END]") == 3
        assert output.count("[STEP]") >= 3
        assert "nginx_crash" in output
        assert "disk_full" in output
        assert "network_broken" in output
        assert float(run["wall_seconds"]) < 1200.0
        assert int(run["peak_rss_kb"]) < 8 * 1024 * 1024
        assert float(run["cpu_seconds"]) < float(run["wall_seconds"]) * 2.2

    assert len(runs) == 3
