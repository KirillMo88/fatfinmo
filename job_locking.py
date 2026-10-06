"""Crash-safe advisory file locks for scheduled and manual refresh jobs."""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def advisory_file_lock(path: Path) -> Iterator[None]:
    """Lock a stable file using an OS lock, which is released when a process exits.

    The file intentionally remains on disk after release. Its existence is not
    the lock; the kernel lock state is. This makes leftovers from killed jobs
    harmless while retaining useful metadata about the last owner.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    try:
        if os.name == "nt":
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise FileExistsError(f"Job lock is currently held: {path}") from exc
        else:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FileExistsError(f"Job lock is currently held: {path}") from exc
        acquired = True

        metadata = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        if os.name == "nt":
            # Keep byte zero as the stable Windows locking range.
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\n")
            os.lseek(descriptor, 1, os.SEEK_SET)
            os.write(descriptor, metadata)
            os.ftruncate(descriptor, 1 + len(metadata))
        else:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, metadata)
        os.fsync(descriptor)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def advisory_file_lock_held(path: Path) -> bool:
    """Check active lock state without treating a leftover file as a lock."""
    if not path.exists():
        return False
    try:
        descriptor = os.open(str(path), os.O_RDWR)
    except OSError:
        return True
    acquired = False
    try:
        if os.name == "nt":
            if os.fstat(descriptor).st_size == 0:
                return False
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
        else:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        acquired = True
        return False
    finally:
        if acquired:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
