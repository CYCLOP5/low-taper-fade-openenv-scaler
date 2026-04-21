import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


DEFAULT_VOLATILE_ROOT = "/dev/shm"


class OverlayFSManager:
    """manages overlayfs stacks for sub second filesystem state resets"""

    def __init__(
        self,
        base_dir: str | None = None,
        *,
        volatile_root: str | None = None,
    ):
        """
        base dir is the parent directory where the merged mount point is created
        volatile root is the ram backed filesystem where upperdir and workdir live
        defaults to /dev/shm so resets never hit persistent disk io
        """
        if base_dir is not None:
            self._base_dir = Path(base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._owns_base_dir = False
        else:
            self._base_dir = Path(tempfile.mkdtemp(prefix="overlayfs_"))
            self._owns_base_dir = True

        volatile_candidate = Path(volatile_root) if volatile_root is not None else Path(DEFAULT_VOLATILE_ROOT)
        self._volatile_base = self._select_volatile_base(volatile_candidate)
        self._volatile_dir = self._volatile_base / f"overlay_{uuid.uuid4().hex}"
        self._volatile_dir.mkdir(parents=True, exist_ok=True)
        print(f"overlay volatile root {self._volatile_dir}")

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

    @property
    def volatile_dir(self) -> Path:
        return self._volatile_dir

    def create_stack(self, lowerdir: str | Path) -> Path:
        """
        creates the overlay directory stack given a lowerdir path
        upperdir and workdir are pinned to the volatile ram disk
        returns the path to the merged directory
        """
        lowerdir = Path(lowerdir).resolve()
        if not lowerdir.is_dir():
            raise FileNotFoundError(f"lowerdir does not exist {lowerdir}")

        self._lowerdir = lowerdir
        self._upperdir = self._volatile_dir / "upper"
        self._workdir = self._volatile_dir / "work"
        self._merged = self._base_dir / "merged"

        self._upperdir.mkdir(exist_ok=True)
        self._workdir.mkdir(exist_ok=True)
        self._merged.mkdir(exist_ok=True)

        print(f"overlay stack created upper {self._upperdir} work {self._workdir} merged {self._merged}")
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
            print("overlay kernel mount start")
            self._mount_kernel()
            self._mount_type = "kernel"
            print("overlay mounted via kernel overlayfs")
        except (PermissionError, OSError, subprocess.CalledProcessError) as exc:
            print(f"overlay kernel mount failed {type(exc).__name__.lower()}")
            try:
                print("overlay fuse mount start")
                self._mount_fuse()
                self._mount_type = "fuse"
                print("overlay mounted via fuse overlayfs")
            except (FileNotFoundError, OSError, subprocess.CalledProcessError) as fuse_exc:
                print(f"overlay fuse mount failed {type(fuse_exc).__name__.lower()}")
                self._mount_copy()
                self._mount_type = "copy"
                print("overlay mounted via copy fallback")

        self._mounted = True

    def _mount_copy(self) -> None:
        if self._lowerdir is None or self._merged is None:
            raise RuntimeError("copy fallback requires lowerdir and merged path")

        self._clear_directory(self._merged)
        shutil.copytree(self._lowerdir, self._merged, dirs_exist_ok=True, symlinks=True)

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
        print(f"overlay fuse binary {fuse_bin}")

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
        upperdir/workdir live on tmpfs so this stays sub 10ms on warm kernels
        returns the reset latency in milliseconds
        """
        if not self._mounted:
            raise RuntimeError("overlay is not mounted")

        start = time.perf_counter()

        mount_type = self._mount_type

        if mount_type == "copy":
            if self._merged is None:
                raise RuntimeError("copy fallback merged path missing")
            self._mount_copy()
            self._mount_type = "copy"
        else:
            self.unmount()

            self._purge_volatile_pair()

            if self._merged is not None:
                self._merged.mkdir(exist_ok=True)

            if mount_type == "kernel":
                self._mount_kernel()
                self._mount_type = "kernel"
            else:
                self._mount_fuse()
                self._mount_type = "fuse"

        self._mounted = True

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(f"overlay reset {elapsed_ms:.1f}ms")
        return elapsed_ms

    def unmount(self) -> None:
        """unmounts the overlay filesystem"""
        if not self._mounted:
            return

        if self._mount_type == "copy":
            self._mounted = False
            self._mount_type = None
            print("overlay unmounted")
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

    def _purge_volatile_pair(self) -> None:
        """wipes upperdir and workdir trees from the volatile ram disk"""
        for target in (self._upperdir, self._workdir):
            if target is None:
                continue
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

    def _clear_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for entry in directory.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _select_volatile_base(self, preferred: Path) -> Path:
        """picks a ram backed root or falls back to the system temp dir"""
        candidates: list[Path] = [preferred]
        if preferred != Path(DEFAULT_VOLATILE_ROOT):
            candidates.append(Path(DEFAULT_VOLATILE_ROOT))
        candidates.append(Path(tempfile.gettempdir()))

        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / f".probe_{uuid.uuid4().hex}"
                probe.touch()
                probe.unlink()
                return candidate
            except OSError as exc:
                print(f"overlay volatile candidate rejected {candidate} {type(exc).__name__.lower()}")
                continue

        raise RuntimeError("no writable volatile root available")

    def cleanup(self) -> None:
        """unmounts if mounted and recursively deletes all overlay directories"""
        self.unmount()

        for d in [self._upperdir, self._workdir, self._merged]:
            if d is not None and d.exists():
                shutil.rmtree(d, ignore_errors=True)

        if self._volatile_dir.exists():
            shutil.rmtree(self._volatile_dir, ignore_errors=True)

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
