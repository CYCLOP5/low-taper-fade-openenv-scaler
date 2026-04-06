import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sysadmin_env.overlayfs import OverlayFSManager


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


def test_create_stack():
    print("\n--- test create stack ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "base_file.txt").write_text("pristine content")
        Path(lowerdir, "subdir").mkdir()
        Path(lowerdir, "subdir", "nested.txt").write_text("nested content")

        manager = OverlayFSManager()
        merged = manager.create_stack(lowerdir)

        assert manager.lowerdir is not None
        assert manager.upperdir is not None
        assert manager.workdir is not None
        assert manager.merged is not None
        assert manager.upperdir.is_dir()
        assert manager.workdir.is_dir()
        assert merged.is_dir()
        assert not manager.is_mounted

        print("create stack passed")
        manager.cleanup()


def test_create_stack_invalid_lowerdir():
    print("\n--- test create stack invalid lowerdir ---")
    manager = OverlayFSManager()
    try:
        manager.create_stack("/nonexistent/path/that/does/not/exist")
        assert False
    except FileNotFoundError:
        print("invalid lowerdir correctly rejected")
    manager.cleanup()


def test_mount_and_read():
    print("\n--- test mount and read ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "readme.txt").write_text("hello from lowerdir")

        with OverlayFSManager() as manager:
            merged = manager.create_stack(lowerdir)
            manager.mount()

            assert manager.is_mounted
            assert manager.mount_type in ("kernel", "fuse")

            merged_file = merged / "readme.txt"
            assert merged_file.exists()
            assert merged_file.read_text() == "hello from lowerdir"

            print(f"mount and read passed using {manager.mount_type}")


def test_write_appears_in_upperdir():
    print("\n--- test write appears in upperdir ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "original.txt").write_text("original")

        with OverlayFSManager() as manager:
            merged = manager.create_stack(lowerdir)
            manager.mount()

            new_file = merged / "agent_created.txt"
            new_file.write_text("agent wrote this")

            upper_file = manager.upperdir / "agent_created.txt"
            assert upper_file.exists()
            assert upper_file.read_text() == "agent wrote this"

            print("write to upperdir confirmed")


def test_reset_clears_writes():
    print("\n--- test reset clears writes ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "base.txt").write_text("base content")

        with OverlayFSManager() as manager:
            merged = manager.create_stack(lowerdir)
            manager.mount()

            (merged / "ephemeral.txt").write_text("this should vanish")
            (merged / "ephemeral_dir").mkdir()
            (merged / "ephemeral_dir" / "inner.txt").write_text("also vanish")

            assert (merged / "ephemeral.txt").exists()

            latency = manager.reset()

            assert not (manager.upperdir / "ephemeral.txt").exists()
            assert not (manager.upperdir / "ephemeral_dir").exists()

            assert (merged / "base.txt").exists()
            assert (merged / "base.txt").read_text() == "base content"

            print(f"reset clears writes passed latency {latency:.1f}ms")


def test_lowerdir_immutability():
    print("\n--- test lowerdir immutability ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        original = Path(lowerdir, "immutable.txt")
        original.write_text("must not change")
        original_content = original.read_text()
        original_files = set(os.listdir(lowerdir))

        with OverlayFSManager() as manager:
            merged = manager.create_stack(lowerdir)
            manager.mount()

            (merged / "new_file.txt").write_text("new file")

            merged_immutable = merged / "immutable.txt"
            merged_immutable.write_text("CORRUPTED")

            manager.reset()

        assert original.read_text() == original_content
        assert set(os.listdir(lowerdir)) == original_files
        print("lowerdir immutability confirmed")


def test_reset_latency():
    print("\n--- test reset latency ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "base.txt").write_text("base")

        with OverlayFSManager() as manager:
            merged = manager.create_stack(lowerdir)
            manager.mount()

            for i in range(100):
                (merged / f"file_{i}.txt").write_text(f"content {i}")

            latency = manager.reset()

            assert latency < 1000.0
            print(f"sub second reset confirmed {latency:.1f}ms for 100 files")


def test_double_mount_rejected():
    print("\n--- test double mount rejected ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "f.txt").write_text("x")

        with OverlayFSManager() as manager:
            manager.create_stack(lowerdir)
            manager.mount()
            try:
                manager.mount()
                assert False
            except RuntimeError:
                print("double mount correctly rejected")


def test_mount_before_create_rejected():
    print("\n--- test mount before create rejected ---")
    manager = OverlayFSManager()
    try:
        manager.mount()
        assert False
    except RuntimeError:
        print("mount before create correctly rejected")
    manager.cleanup()


def test_reset_before_mount_rejected():
    print("\n--- test reset before mount rejected ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "f.txt").write_text("x")
        manager = OverlayFSManager()
        manager.create_stack(lowerdir)
        try:
            manager.reset()
            assert False
        except RuntimeError:
            print("reset before mount correctly rejected")
        manager.cleanup()


def test_context_manager_cleanup():
    print("\n--- test context manager cleanup ---")
    with tempfile.TemporaryDirectory(prefix="test_lower_") as lowerdir:
        Path(lowerdir, "f.txt").write_text("x")
        base_dir = None
        with OverlayFSManager() as manager:
            merged = manager.create_stack(lowerdir)
            manager.mount()
            base_dir = manager._base_dir
            assert base_dir.exists()

        assert not base_dir.exists()
        print("context manager cleanup passed")


def main():
    check_fuse_overlayfs_available()

    tests = [
        test_create_stack,
        test_create_stack_invalid_lowerdir,
        test_mount_and_read,
        test_write_appears_in_upperdir,
        test_reset_clears_writes,
        test_lowerdir_immutability,
        test_reset_latency,
        test_double_mount_rejected,
        test_mount_before_create_rejected,
        test_reset_before_mount_rejected,
        test_context_manager_cleanup,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED {test.__name__} {e}")
            failed += 1

    print(f"\n--- results {passed} passed {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
