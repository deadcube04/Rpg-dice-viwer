"""Local append-only segments; expiration deletes whole closed files only."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import re
import stat
from threading import Lock
import uuid

logger = logging.getLogger("dice_viewer.recognition.observability")


@dataclass(frozen=True)
class LogPolicy:
    max_bytes: int = 10 * 1024 * 1024
    retention_days: int = 30
    max_total_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.max_bytes, self.retention_days, self.max_total_bytes)):
            raise ValueError("Log limits must be positive integers")
        if self.max_total_bytes < self.max_bytes:
            raise ValueError("Total log limit must cover a segment")
        if self.retention_days > 36500:
            raise ValueError("Retention must not exceed 36500 days")


class AppendOnlyLog:
    """One writer/process per path. Flush/fsync/close each appended batch."""

    def __init__(self, path, policy=None, clock=None):
        self.path = Path(path)
        self.policy = policy or LogPolicy()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock = Lock()
        self.archive_pattern = re.compile(re.escape(self.path.stem) +
                                          r"\.\d{8}T\d{6}\.\d{6}Z\.[0-9a-f]{32}\.jsonl\Z")

    def append(self, payload):
        if len(payload) > self.policy.max_total_bytes:
            raise ValueError("Batch exceeds total log budget")
        if not payload:
            return
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            now = self.clock()
            if self.path.is_symlink():
                raise OSError("Active log must not be a symbolic link")
            if self.path.exists():
                info = self.path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise OSError("Active log must be a regular file")
                last_write = datetime.fromtimestamp(info.st_mtime, timezone.utc)
                incomplete = False
                if info.st_size:
                    with self.path.open("rb") as source:
                        source.seek(-1, os.SEEK_END)
                        incomplete = source.read(1) != b"\n"
                if info.st_size and (incomplete or last_write.date() != now.date()
                                     or info.st_size + len(payload) > self.policy.max_bytes):
                    stamp = last_write.strftime("%Y%m%dT%H%M%S.%fZ")
                    destination = self.path.with_name(f"{self.path.stem}.{stamp}.{uuid.uuid4().hex}.jsonl")
                    self.path.rename(destination)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # A retention failure must not retry or discard the appended batch.
            try:
                self._expire(now)
            except Exception:
                try:
                    logger.error("observability_retention_failed")
                except Exception:
                    pass

    def _expire(self, now):
        archives = []
        for candidate in self.path.parent.iterdir():
            if not self.archive_pattern.fullmatch(candidate.name):
                continue
            info = candidate.lstat()
            if stat.S_ISREG(info.st_mode):
                archives.append((info.st_mtime, candidate.name, candidate, info.st_size))
        archives.sort()
        cutoff = (now - timedelta(days=self.policy.retention_days)).timestamp()
        total = self.path.stat().st_size + sum(entry[3] for entry in archives)
        for modified, _, candidate, size in archives:
            if modified < cutoff or total > self.policy.max_total_bytes:
                # Skip links even if a file was replaced after the directory scan.
                if stat.S_ISREG(candidate.lstat().st_mode):
                    candidate.unlink()
                    total -= size
