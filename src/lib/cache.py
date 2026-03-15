"""Response caching with TTL-based LRU eviction and memory bounds."""

import json
import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("sfu_library_mcp")


class ResponseCache:
    """Thread-safe TTL-based LRU cache with memory bounds.

    Entries expire after ``ttl`` seconds. When the cache exceeds
    ``max_size`` entries or ``max_memory_mb`` megabytes, the oldest
    entries are evicted.
    """

    def __init__(
        self,
        ttl: int = 300,
        max_size: int = 100,
        max_memory_mb: int = 50,
    ):
        self.ttl = ttl
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._current_memory = 0

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        try:
            return sys.getsizeof(json.dumps(value))
        except (TypeError, ValueError):
            return sys.getsizeof(str(value))

    def get(self, key: str) -> Any | None:
        """Get a value from cache if it exists and hasn't expired.

        Returns None on miss or expiry.
        """
        with self._lock:
            if key not in self._cache:
                return None

            timestamp, value = self._cache[key]
            if time.time() - timestamp > self.ttl:
                # Expired — remove it
                self._evict(key)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        with self._lock:
            # If key exists, remove old entry first
            if key in self._cache:
                self._evict(key)

            entry_size = self._estimate_size(value)

            # Evict until we have room
            while (
                len(self._cache) >= self.max_size
                or (self._current_memory + entry_size > self.max_memory_bytes and self._cache)
            ):
                self._evict_oldest()

            self._cache[key] = (time.time(), value)
            self._current_memory += entry_size

    def _evict(self, key: str) -> None:
        """Remove a specific key (caller must hold lock)."""
        if key in self._cache:
            _, value = self._cache.pop(key)
            self._current_memory -= self._estimate_size(value)
            self._current_memory = max(0, self._current_memory)

    def _evict_oldest(self) -> None:
        """Remove the oldest entry (caller must hold lock)."""
        if self._cache:
            key, (_, value) = self._cache.popitem(last=False)
            self._current_memory -= self._estimate_size(value)
            self._current_memory = max(0, self._current_memory)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    @property
    def size(self) -> int:
        """Number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def make_key(self, *args: Any) -> str:
        """Create a cache key from arguments."""
        return ":".join(str(a) for a in args)
