from __future__ import annotations

from typing import Iterable, Tuple

from numpy.typing import ArrayLike, NDArray

def eigh(
    a: ArrayLike,
    b: ArrayLike = ...,
    lower: bool = ...,
    eigvals_only: bool = ...,
    subset_by_index: Iterable[int] = ...,
    subset_by_value: Iterable[int] = ...,
    driver: str = ...,
    type: int = ...,
    overwrite_a: bool = ...,
    overwrite_b: bool = ...,
    check_finite: bool = ...,
) -> Tuple[NDArray, NDArray]:
    ...
