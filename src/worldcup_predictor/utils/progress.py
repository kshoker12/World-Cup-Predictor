"""Progress bar utilities."""

from __future__ import annotations

import sys
from typing import Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    disable: bool | None = None,
) -> Iterable[T]:
    if disable is None:
        disable = not sys.stderr.isatty()
    return tqdm(iterable, desc=desc, total=total, disable=disable)
