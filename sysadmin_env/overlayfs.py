import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class OverlayFSManager:
    """manages overlayfs stacks for sub second filesystem state resets"""

    def __init__(self, base_dir: str | None = None):
        """
        base dir is the parent directory where overlay stack directories are created
        if none a temporary directory is used
        """
        if base_dir is not None:
            self._base_dir = Path(base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._owns_base_dir = False
        else:
            self._base_dir = Path(tempfile.mkdtemp(prefix="overlayfs_"))
            self._owns_base_dir = True

        self._lowerdir: Path | None = None
        self._upperdir: Path | None = None
        self._workdir: Path | None = None
        self._merged: Path | None = None
        self._mounted = False
        self._mount_type: str | None = None

    @property
    def lowerdir(self) -> Path | None:
        return self._lowerdir

    @property
    def upperdir(self) -> Path | None:
        return self._upperdir

    @property
    def workdir(self) -> Path | None:
        return self._workdir

    @property
    def merged(self) -> Path | None:
        return self._merged

    @property
    def is_mounted(self) -> bool:
        return self._mounted

    @property
    def mount_type(self) -> str | None:
        return self._mount_type

    def create_stack(self, lowerdir: str | Path) -> Path:
        """
        creates the overlay directory stack given a lowerdir path
        returns the path to the merged directory
        """
        lowerdir = Path(lowerdir).resolve()
        if not lowerdir.is_dir():
            raise FileNotFoundError(f"lowerdir does not exist {lowerdir}")

        self._lowerdir = lowerdir
        self._upperdir = self._base_dir / "upper"
        self._workdir = self._base_dir / "work"
        self._merged = self._base_dir / "merged"

        self._upperdir.mkdir(exist_ok=True)
        self._workdir.mkdir(exist_ok=True)
        self._merged.mkdir(exist_ok=True)

        print(f"overlay stack created at {self._base_dir}")
        return self._merged

    def mount(self) -> None:
        """
        mounts the overlay filesystem trying kernel overlayfs first
        then falling back to fuse overlayfs for unprivileged contexts
        """
        if self._mounted:
            raise RuntimeError("overlay already mounted")

        if self._merged is None:
            raise RuntimeError("create stack must be called before mount")

        try:
            self._mount_kernel()
            self._mount_type = "kernel"
            print("overlay mounted via kernel overlayfs")
        except (PermissionError, OSError, subprocess.CalledProcessError):
            self._mount_fuse()
            self._mount_type = "fuse"
            print("overlay mounted via fuse overlayfs")

        self._mounted = True

    def _mount_kernel(self) -> None:
        mount_opts = (
            f"lowerdir={self._lowerdir},"
            f"upperdir={self._upperdir},"
            f"workdir={self._workdir}"
        )
        result = subprocess.run(
            ["mount", "-t", "overlay", "overlay", "-o", mount_opts, str(self._merged)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise PermissionError(f"kernel mount failed {result.stderr.strip()}")

    def _mount_fuse(self) -> None:
        fuse_bin = shutil.which("fuse-overlayfs")
        if fuse_bin is None:
            raise FileNotFoundError("fuse-overlayfs binary not found in path")

        mount_opts = (
            f"lowerdir={self._lowerdir},"
            f"upperdir={self._upperdir},"
            f"workdir={self._workdir}"
        )
        result = subprocess.run(
            [fuse_bin, "-o", mount_opts, str(self._merged)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise OSError(f"fuse overlayfs mount failed {result.stderr.strip()}")

    def reset(self) -> float:
        """
        resets the overlay by clearing upperdir contents and recreating workdir
        returns the reset latency in milliseconds
        """
        if not self._mounted:
            raise RuntimeError("overlay is not mounted")

        start = time.perf_counter()

        for entry in self._upperdir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        if self._workdir.exists():
            shutil.rmtree(self._workdir)
        self._workdir.mkdir()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(f"overlay reset {elapsed_ms:.1f}ms")
        return elapsed_ms

    def unmount(self) -> None:
        """unmounts the overlay filesystem"""
        if not self._mounted:
            return

        if self._mount_type == "fuse":
            result = subprocess.run(
                ["fusermount", "-u", str(self._merged)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                subprocess.run(
                    ["fusermount3", "-u", str(self._merged)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
        else:
            subprocess.run(
                ["umount", str(self._merged)],
                capture_output=True,
                text=True,
                timeout=10,
            )

        self._mounted = False
        self._mount_type = None
        print("overlay unmounted")

    def cleanup(self) -> None:
        """unmounts if mounted and removes all overlay directories"""
        self.unmount()

        for d in [self._upperdir, self._workdir, self._merged]:
            if d is not None and d.exists():
                shutil.rmtree(d, ignore_errors=True)

        if self._owns_base_dir and self._base_dir.exists():
            shutil.rmtree(self._base_dir, ignore_errors=True)

        self._lowerdir = None
        self._upperdir = None
        self._workdir = None
        self._merged = None
        self._mount_type = None
        print("overlay cleanup complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
