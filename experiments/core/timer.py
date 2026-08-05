from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True, slots=True)
class TimerResult:
    name: str
    elapsed_time: float


class Timer:

    def __init__(self, name: str) -> None:
        self._name = name

    # Going to use this class as a context manager, so we can use it with the `with` statement.
    def __enter__(self) -> "Timer":
        self._start_time = perf_counter()
        return self

    # Going to use this class as a context manager, so we can use it with the `with` statement.
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._elapsed_time = perf_counter() - self._start_time

    def __str__(self) -> str:
        return f"{self._name}: {self._elapsed_time:.6f} seconds"

    @property
    def result(self) -> TimerResult:
        return TimerResult(
            name=self._name,
            elapsed_time=self._elapsed_time,
        )