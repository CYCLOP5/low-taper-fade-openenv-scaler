import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from sysadmin_env.overlayfs import OverlayFSManager


@dataclass
class CommandResult:
    """raw output from a sandbox command execution"""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    timed_out: bool = False


class Sandbox:
    """
    manages the lifecycle of an isolated bubblewrap sandbox backed by overlayfs
    for sub second state resets between episodes
    """

    _HOST_RO_BINDS = [
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc/alternatives",
        "/etc/ld.so.cache",
    ]

    def __init__(
        self,
        lowerdir: str | Path,
        *,
        timeout: float = 30.0,
        isolate_network: bool = True,
        overlay_base_dir: str | None = None,
    ):
        """
        lowerdir is the pristine read only filesystem the agent interacts with
        timeout is the default per command timeout in seconds
        isolate network controls whether a new network namespace is created
        overlay base dir is an optional parent for overlay directories
        """
        self._lowerdir = Path(lowerdir).resolve()
        self._timeout = timeout
        self._isolate_network = isolate_network
        self._overlay = OverlayFSManager(base_dir=overlay_base_dir)
        self._created = False
        self._destroyed = False

    @property
    def is_created(self) -> bool:
        return self._created

    @property
    def is_destroyed(self) -> bool:
        return self._destroyed

    @property
    def overlay(self) -> OverlayFSManager:
        return self._overlay

    @property
    def merged_root(self) -> Path | None:
        return self._overlay.merged

    def create(self) -> None:
        """
        creates the sandbox by initializing and mounting the overlayfs stack
        must be called before execute
        """
        if self._created:
            raise RuntimeError("sandbox already created")
        if self._destroyed:
            raise RuntimeError("sandbox has been destroyed and cannot be recreated")

        self._verify_bwrap_available()
        self._overlay.create_stack(self._lowerdir)
        self._overlay.mount()
        self._created = True
        print("sandbox created")

    def _verify_bwrap_available(self) -> None:
        if shutil.which("bwrap") is None:
            raise FileNotFoundError("bwrap binary not found in path")

    def _build_bwrap_command(self, command: str) -> list[str]:
        """
        constructs the full bwrap command line binding host system dirs
        read only and the overlayfs merged dir as the task workspace
        """
        merged = str(self._overlay.merged)

        cmd = [
            "bwrap",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]

        if self._isolate_network:
            cmd.append("--unshare-net")

        for host_path in self._HOST_RO_BINDS:
            if Path(host_path).exists():
                cmd.extend(["--ro-bind", host_path, host_path])

        cmd.extend([
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--tmpfs", "/run",
            "--bind", merged, merged,
            "--chdir", merged,
            "--clearenv",
            "--setenv", "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--setenv", "HOME", "/root",
            "--setenv", "TERM", "xterm",
            "--setenv", "SANDBOX_ROOT", merged,
            "--hostname", "sandbox",
            "--die-with-parent",
            "--",
            "/bin/sh", "-c", command,
        ])

        return cmd

    def execute(self, command: str, *, timeout: float | None = None) -> CommandResult:
        """
        executes a command inside the sandbox and returns the captured output
        uses the default timeout if none is provided per call
        """
        if not self._created:
            raise RuntimeError("sandbox not created call create first")
        if self._destroyed:
            raise RuntimeError("sandbox has been destroyed")

        effective_timeout = timeout if timeout is not None else self._timeout
        bwrap_cmd = self._build_bwrap_command(command)

        result = CommandResult()
        start = time.perf_counter()

        try:
            proc = subprocess.run(
                bwrap_cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            result.stdout = proc.stdout
            result.stderr = proc.stderr
            result.exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            result.stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
            result.stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
            result.exit_code = -1
            result.timed_out = True

        result.execution_time = time.perf_counter() - start

        return result

    async def execute_async(self, command: str, *, timeout: float | None = None) -> CommandResult:
        """
        async variant of execute that runs the sandbox command without
        blocking the event loop
        """
        if not self._created:
            raise RuntimeError("sandbox not created call create first")
        if self._destroyed:
            raise RuntimeError("sandbox has been destroyed")

        effective_timeout = timeout if timeout is not None else self._timeout
        bwrap_cmd = self._build_bwrap_command(command)

        result = CommandResult()
        start = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                *bwrap_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout,
                )
                result.stdout = stdout_bytes.decode("utf-8", errors="replace")
                result.stderr = stderr_bytes.decode("utf-8", errors="replace")
                result.exit_code = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                result.stdout = ""
                result.stderr = ""
                result.exit_code = -1
                result.timed_out = True
        except OSError as exc:
            result.stderr = str(exc)
            result.exit_code = -1

        result.execution_time = time.perf_counter() - start

        return result

    def reset(self) -> float:
        """
        resets the sandbox filesystem state via overlayfs
        returns the reset latency in milliseconds
        """
        if not self._created:
            raise RuntimeError("sandbox not created call create first")
        if self._destroyed:
            raise RuntimeError("sandbox has been destroyed")

        latency = self._overlay.reset()
        print(f"sandbox reset {latency:.1f}ms")
        return latency

    def destroy(self) -> None:
        """releases all sandbox resources including the overlayfs stack"""
        if self._destroyed:
            return

        self._overlay.cleanup()
        self._created = False
        self._destroyed = True
        print("sandbox destroyed")

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False
