from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

class rv_continuous:
    def cdf(self, x: ArrayLike, *args: ArrayLike, loc: ArrayLike = ..., scale: ArrayLike = ...) -> NDArray: ...
    def pdf(self, x: ArrayLike, *args: ArrayLike, loc: ArrayLike = ..., scale: ArrayLike = ...) -> NDArray: ...

norm: rv_continuous
