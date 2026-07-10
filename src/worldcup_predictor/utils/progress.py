"""Progress bar utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def _on_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    disable: bool | None = None,
) -> Iterable[T]:
    if disable is None:
        disable = not sys.stderr.isatty() and not _on_kaggle()
    kwargs: dict[str, object] = {
        "desc": desc,
        "total": total,
        "disable": disable,
    }
    if _on_kaggle():
        kwargs["file"] = sys.stdout
        kwargs["mininterval"] = 10.0
    return tqdm(iterable, **kwargs)
