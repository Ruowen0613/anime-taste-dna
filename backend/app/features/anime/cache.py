from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value")


class TtlCache(Generic[Key, Value]):
    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._values: dict[Key, tuple[float, Value]] = {}

    def get(self, key: Key) -> Value | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= self._clock():
            self._values.pop(key, None)
            return None
        return value

    def set(self, key: Key, value: Value) -> None:
        self._remove_expired()
        if key not in self._values and len(self._values) >= self._max_entries:
            self._values.pop(next(iter(self._values)))
        self._values[key] = (self._clock() + self._ttl_seconds, value)

    def clear(self) -> None:
        self._values.clear()

    def _remove_expired(self) -> None:
        now = self._clock()
        expired = [key for key, (expires_at, _) in self._values.items() if expires_at <= now]
        for key in expired:
            self._values.pop(key, None)
