from __future__ import annotations

from typing import Literal

import numpy
import numpy.random
from numpy.typing import ArrayLike, NDArray

class LatinHypercube:
    def __init__(
        self,
        d: int,
        scramble: bool = ...,
        optimization: Literal["random-cd", "lloyd"] = ...,
        strength: Literal[1, 2] = ...,
        seed: int | numpy.random.Generator = ...,
    ):...
    def random(self, n: int = ..., *, workers: int = ...) -> NDArray[numpy.double]: ...

def scale(sample: ArrayLike, l_bounds: ArrayLike, u_bounds: ArrayLike, reverse: bool = ...) -> NDArray[numpy.double]: ...
