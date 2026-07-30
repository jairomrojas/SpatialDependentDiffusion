"""Low-overhead terminal progress meters with elapsed time and ETA."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    description: str,
    unit: str,
    enabled: bool,
) -> tqdm[T]:
    return tqdm(
        iterable,
        total=total,
        desc=description,
        unit=unit,
        disable=not enabled,
        dynamic_ncols=True,
        smoothing=0.2,
        mininterval=0.15,
        leave=True,
    )
