"""Subprocess lifecycle guard for the MAZ server.

Tracks every child subprocess we launch so we can kill the entire family
(grandchildren too) on graceful shutdown or via the admin /panic endpoint.
This prevents the "had to reboot the PC" scenario where orphan workers hold
the GPU, port bindings, or SQLite/diskcache locks after a crash.

psutil is used when available for recursive child enumeration; if it is
absent the guard falls back to terminating only the directly tracked PIDs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Set

try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False

log = logging.getLogger("maz.guard")

_tracked: Set[object] = set()


def register(proc: object) -> None:
    """Register a freshly spawned child for shutdown cleanup."""
    if proc is not None:
        _tracked.add(proc)


def unregister(proc: object) -> None:
    """Drop a finished child from tracking."""
    _tracked.discard(proc)


def _proc_pid(proc: object) -> int | None:
    pid = getattr(proc, "pid", None)
    return int(pid) if pid else None


def _alive(proc: object) -> bool:
    rc = getattr(proc, "returncode", None)
    return rc is None


def _kill_pid_tree(pid: int, timeout: float = 4.0) -> int:
    """Kill a process and all its descendants. Returns count terminated."""
    if not HAS_PSUTIL:
        try:
            os.kill(pid, 9)
            return 1
        except Exception:
            return 0
    killed = 0
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    family: List[psutil.Process] = parent.children(recursive=True) + [parent]
    for p in family:
        try:
            p.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(family, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    killed = len(family)
    return killed


def kill_all(timeout: float = 4.0) -> int:
    """Kill every tracked child plus its descendants. Returns count."""
    total = 0
    procs = list(_tracked)
    _tracked.clear()
    for proc in procs:
        if not _alive(proc):
            continue
        pid = _proc_pid(proc)
        if pid is None:
            continue
        total += _kill_pid_tree(pid, timeout=timeout)
    if total:
        log.warning(f"[guard] killed {total} tracked process(es) on shutdown")
    return total


_WORKER_SCRIPT_NAMES = {
    "voice_pipeline_worker.py",
    "voice_sections_worker.py",
    "rvc_training_worker.py",
}


def find_orphans() -> List[dict]:
    """Find python.exe processes running our worker scripts that we no longer track.

    Returned dicts have ``pid`` and ``name`` keys. Empty list when psutil is
    unavailable or no orphans match.
    """
    if not HAS_PSUTIL:
        return []
    tracked_pids = {_proc_pid(p) for p in _tracked if _proc_pid(p) is not None}
    orphans: List[dict] = []
    self_pid = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == self_pid:
                continue
            cmdline = p.info.get("cmdline") or []
            if not cmdline:
                continue
            for arg in cmdline:
                base = Path(str(arg)).name
                if base in _WORKER_SCRIPT_NAMES and p.info["pid"] not in tracked_pids:
                    orphans.append({"pid": p.info["pid"], "name": base})
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return orphans


def kill_orphans(timeout: float = 4.0) -> int:
    """Find and kill orphan worker processes left from a prior crash."""
    orphans = find_orphans()
    killed = 0
    for o in orphans:
        killed += _kill_pid_tree(o["pid"], timeout=timeout)
    if killed:
        log.warning(f"[guard] killed {killed} orphan worker process(es)")
    return killed
