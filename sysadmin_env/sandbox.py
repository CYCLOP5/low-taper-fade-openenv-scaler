import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from sysadmin_env.overlayfs import OverlayFSManager


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    timed_out: bool = False


class Sandbox:
    _HOST_RO_BINDS = [
        "/usr/bin",
        "/usr/sbin",
        "/usr/lib",
        "/usr/lib64",
        "/usr/share",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
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
        allow_nested_sandbox: bool = False,
    ):
        self._lowerdir = Path(lowerdir).resolve()
        self._timeout = timeout
        self._isolate_network = isolate_network
        self._overlay = OverlayFSManager(base_dir=overlay_base_dir)
        self._allow_nested_sandbox = allow_nested_sandbox
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
    def merged_root(self) -> Path:
        return Path("/")

    @property
    def state_root(self) -> Path | None:
        return self._overlay.merged

    def create(self) -> None:
        if self._created:
            raise RuntimeError("sandbox already created")
        if self._destroyed:
            raise RuntimeError("sandbox has been destroyed and cannot be recreated")

        print("sandbox verify bwrap start")
        self._verify_bwrap_available()
        print("sandbox verify bwrap complete")
        print(f"sandbox create stack {self._lowerdir}")
        self._overlay.create_stack(self._lowerdir)
        print("sandbox overlay mount start")
        try:
            self._overlay.mount()
        except Exception as exc:
            print(f"sandbox overlay mount failed {type(exc).__name__.lower()}")
            raise
        print("sandbox overlay mount complete")
        print("sandbox runtime layout start")
        self._ensure_runtime_layout()
        print("sandbox runtime layout complete")
        self._created = True
        print("sandbox created")

    def _verify_bwrap_available(self) -> None:
        bwrap_bin = shutil.which("bwrap")
        if bwrap_bin is None:
            raise FileNotFoundError("bwrap binary not found in path")
        print(f"sandbox bwrap found {bwrap_bin}")

    def _ensure_runtime_layout(self) -> None:
        if self._overlay.merged is None:
            raise RuntimeError("overlay stack not ready")

        for relative in [
            Path("bin"),
            Path("sbin"),
            Path("lib"),
            Path("lib64"),
            Path("usr"),
            Path("usr/bin"),
            Path("usr/sbin"),
            Path("usr/lib"),
            Path("usr/lib64"),
            Path("usr/share"),
            Path("usr/local"),
            Path("usr/local/bin"),
            Path("etc"),
            Path("etc/alternatives"),
            Path("var"),
            Path("var/tmp"),
            Path("tmp"),
            Path("dev"),
            Path("proc"),
            Path("run"),
            Path("root"),
            Path("home"),
        ]:
            (self._overlay.merged / relative).mkdir(parents=True, exist_ok=True)

    def _build_bwrap_command(self, command: str) -> list[str]:
        if self._overlay.merged is None:
            raise RuntimeError("sandbox storage not ready")

        merged = str(self._overlay.merged)

        cmd = [
            "bwrap",
            "--bind",
            merged,
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--die-with-parent",
            "--hostname",
            "sandbox",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "--setenv",
            "HOME",
            "/root",
            "--setenv",
            "TERM",
            "xterm",
            "--uid",
            "0",
            "--gid",
            "0",
        ]

        if self._allow_nested_sandbox:
            cmd.extend([
                "--unshare-user",
                "--cap-add",
                "CAP_SYS_ADMIN",
            ])
        else:
            cmd.extend(["--cap-drop", "ALL"])

        if self._isolate_network:
            cmd.append("--unshare-net")

        for host_path in self._HOST_RO_BINDS:
            if Path(host_path).exists():
                cmd.extend(["--ro-bind", host_path, host_path])

        cmd.extend([
            "--chdir",
            "/",
            "--",
            "/bin/sh",
            "-c",
            command,
        ])

        return cmd

    def execute(self, command: str, *, timeout: float | None = None) -> CommandResult:
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
                result.exit_code = -1
                result.timed_out = True
        except OSError as exc:
            result.stderr = str(exc)
            result.exit_code = -1

        result.execution_time = time.perf_counter() - start
        return result

    def reset(self) -> float:
        if not self._created:
            raise RuntimeError("sandbox not created call create first")
        if self._destroyed:
            raise RuntimeError("sandbox has been destroyed")

        latency = self._overlay.reset()
        self._ensure_runtime_layout()
        print(f"sandbox reset {latency:.1f}ms")
        return latency

    def destroy(self) -> None:
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
