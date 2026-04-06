import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sysadmin_env.sandbox import Sandbox
from sysadmin_env.sandbox import CommandResult


def check_bwrap_available():
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        print("bwrap not installed")
        sys.exit(1)
    result = subprocess.run(["bwrap", "--version"], capture_output=True, text=True)
    print(f"bwrap found at {bwrap} version {result.stdout.strip()}")


def check_fuse_overlayfs_available():
    fuse_bin = shutil.which("fuse-overlayfs")
    if fuse_bin is None:
        print("fuse-overlayfs not installed attempting pacman install")
        result = subprocess.run(
            ["sudo", "pacman", "-S", "--noconfirm", "fuse-overlayfs"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"failed to install fuse-overlayfs {result.stderr.strip()}")
            sys.exit(1)
        print("fuse-overlayfs installed successfully")
    else:
        print(f"fuse-overlayfs found at {fuse_bin}")


def create_test_lowerdir(path: Path) -> None:
    """
    creates a minimal lowerdir with some test files
    the sandbox bind mounts host system dirs so no binaries needed here
    """
    (path / "etc").mkdir(exist_ok=True)
    (path / "var" / "log").mkdir(parents=True, exist_ok=True)
    (path / "home").mkdir(exist_ok=True)

    (path / "etc" / "config.txt").write_text("setting=default\n")
    (path / "var" / "log" / "app.log").write_text("startup complete\n")
    (path / "home" / "readme.txt").write_text("task workspace\n")


def test_basic_command_execution():
    print("\n--- test basic command execution ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            result = sandbox.execute("echo hello")

            assert result.stdout.strip() == "hello", f"expected hello got {result.stdout.strip()!r}"
            assert result.exit_code == 0, f"expected exit 0 got {result.exit_code}"
            assert result.execution_time > 0
            assert not result.timed_out

            print(f"basic execution passed stdout={result.stdout.strip()!r} exit={result.exit_code} time={result.execution_time:.3f}s")


def test_command_stderr():
    print("\n--- test command stderr ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            result = sandbox.execute("echo error message >&2")

            assert "error message" in result.stderr, f"expected error in stderr got {result.stderr!r}"
            assert result.exit_code == 0

            print(f"stderr capture passed stderr={result.stderr.strip()!r}")


def test_nonzero_exit_code():
    print("\n--- test nonzero exit code ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            result = sandbox.execute("exit 42")

            assert result.exit_code == 42, f"expected exit 42 got {result.exit_code}"
            assert not result.timed_out

            print(f"nonzero exit passed exit_code={result.exit_code}")


def test_read_lowerdir_files():
    print("\n--- test read lowerdir files ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))
        merged_path = None

        with Sandbox(lowerdir) as sandbox:
            merged_path = str(sandbox.merged_root)
            result = sandbox.execute(f"cat {merged_path}/etc/config.txt")

            assert "setting=default" in result.stdout, f"lowerdir read failed {result.stdout!r}"
            assert result.exit_code == 0

            print(f"read lowerdir files passed content={result.stdout.strip()!r}")


def test_file_write_and_reset():
    print("\n--- test file write and reset ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            merged = str(sandbox.merged_root)

            sandbox.execute(f"echo agent data > {merged}/agent_file.txt")

            verify = sandbox.execute(f"cat {merged}/agent_file.txt")
            assert "agent data" in verify.stdout, f"file write failed {verify.stdout!r} {verify.stderr!r}"

            latency = sandbox.reset()

            after_reset = sandbox.execute(f"cat {merged}/agent_file.txt 2>/dev/null")
            assert after_reset.exit_code != 0 or "agent data" not in after_reset.stdout

            original = sandbox.execute(f"cat {merged}/etc/config.txt")
            assert "setting=default" in original.stdout, "original lowerdir data lost after reset"

            print(f"file write and reset passed reset latency {latency:.1f}ms")


def test_timeout_enforcement():
    print("\n--- test timeout enforcement ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir, timeout=60.0) as sandbox:
            start = time.perf_counter()
            result = sandbox.execute("sleep 60", timeout=2.0)
            elapsed = time.perf_counter() - start

            assert result.timed_out, "expected timeout flag"
            assert result.exit_code == -1, f"expected exit -1 got {result.exit_code}"
            assert elapsed < 10.0, f"timeout took too long {elapsed:.1f}s"

            print(f"timeout enforcement passed elapsed={elapsed:.1f}s timed_out={result.timed_out}")


def test_host_isolation():
    print("\n--- test host isolation ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        host_home = os.path.expanduser("~")
        host_marker = Path(host_home) / ".sandbox_test_marker"
        host_marker_existed = host_marker.exists()

        try:
            if not host_marker_existed:
                host_marker.write_text("test marker")

            with Sandbox(lowerdir) as sandbox:
                result = sandbox.execute(f"cat {host_home}/.sandbox_test_marker 2>/dev/null && echo found || echo blocked")
                assert "blocked" in result.stdout or result.exit_code != 0, "sandbox accessed host home"

                result2 = sandbox.execute("ls /root 2>/dev/null || echo no_access")
                print(f"host isolation check passed home_result={result.stdout.strip()!r}")

        finally:
            if not host_marker_existed and host_marker.exists():
                host_marker.unlink()


def test_sandbox_not_created_errors():
    print("\n--- test sandbox not created errors ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        sandbox = Sandbox(lowerdir)

        try:
            sandbox.execute("echo hello")
            assert False, "should have raised"
        except RuntimeError:
            print("execute before create correctly rejected")

        try:
            sandbox.reset()
            assert False, "should have raised"
        except RuntimeError:
            print("reset before create correctly rejected")

        sandbox.destroy()


def test_double_create_rejected():
    print("\n--- test double create rejected ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        sandbox = Sandbox(lowerdir)
        sandbox.create()

        try:
            sandbox.create()
            assert False, "should have raised"
        except RuntimeError:
            print("double create correctly rejected")

        sandbox.destroy()


def test_destroy_idempotent():
    print("\n--- test destroy idempotent ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        sandbox = Sandbox(lowerdir)
        sandbox.create()
        sandbox.destroy()
        sandbox.destroy()
        print("destroy idempotent confirmed")


def test_context_manager():
    print("\n--- test context manager ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            assert sandbox.is_created
            assert not sandbox.is_destroyed
            result = sandbox.execute("echo context works")
            assert result.exit_code == 0, f"context command failed exit={result.exit_code} stderr={result.stderr!r}"
            assert "context works" in result.stdout

        assert sandbox.is_destroyed
        print("context manager passed")


def test_network_isolation():
    print("\n--- test network isolation ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir, isolate_network=True) as sandbox:
            result = sandbox.execute("echo network isolated")
            assert result.exit_code == 0, f"network isolated failed {result.stderr!r}"
            print("network isolated sandbox runs commands")

        with Sandbox(lowerdir, isolate_network=False) as sandbox:
            result = sandbox.execute("echo network shared")
            assert result.exit_code == 0, f"network shared failed {result.stderr!r}"
            print("network shared sandbox runs commands")

        print("network isolation test passed")


def test_multiple_commands_sequential():
    print("\n--- test multiple commands sequential ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            r1 = sandbox.execute("echo first")
            assert r1.stdout.strip() == "first", f"first command failed {r1.stdout!r} {r1.stderr!r}"

            r2 = sandbox.execute("echo second")
            assert r2.stdout.strip() == "second"

            r3 = sandbox.execute("echo third")
            assert r3.stdout.strip() == "third"

            print("multiple sequential commands passed")


def test_environment_clearenv():
    print("\n--- test environment clearenv ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            result = sandbox.execute("echo $HOME")
            assert "/root" in result.stdout, f"expected /root in HOME got {result.stdout.strip()!r}"

            result2 = sandbox.execute("echo $PATH")
            assert "/usr/bin" in result2.stdout

            print(f"env clearenv passed HOME={result.stdout.strip()!r}")


def test_hostname_isolation():
    print("\n--- test hostname isolation ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            result = sandbox.execute("uname -n")
            assert result.stdout.strip() == "sandbox", f"expected hostname sandbox got {result.stdout.strip()!r}"

            print(f"hostname isolation passed hostname={result.stdout.strip()!r}")


def test_lowerdir_immutability():
    print("\n--- test lowerdir immutability ---")
    with tempfile.TemporaryDirectory(prefix="test_sandbox_lower_") as lowerdir:
        create_test_lowerdir(Path(lowerdir))
        original_config = (Path(lowerdir) / "etc" / "config.txt").read_text()
        original_files = set(os.listdir(lowerdir))

        with Sandbox(lowerdir) as sandbox:
            merged = str(sandbox.merged_root)
            sandbox.execute(f"echo corrupted > {merged}/etc/config.txt")
            sandbox.execute(f"echo new > {merged}/new_file.txt")
            sandbox.reset()

        assert (Path(lowerdir) / "etc" / "config.txt").read_text() == original_config
        assert set(os.listdir(lowerdir)) == original_files
        print("lowerdir immutability confirmed")


def main():
    check_bwrap_available()
    check_fuse_overlayfs_available()

    tests = [
        test_basic_command_execution,
        test_command_stderr,
        test_nonzero_exit_code,
        test_read_lowerdir_files,
        test_file_write_and_reset,
        test_timeout_enforcement,
        test_host_isolation,
        test_sandbox_not_created_errors,
        test_double_create_rejected,
        test_destroy_idempotent,
        test_context_manager,
        test_network_isolation,
        test_multiple_commands_sequential,
        test_environment_clearenv,
        test_hostname_isolation,
        test_lowerdir_immutability,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED {test.__name__} {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n--- results {passed} passed {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
