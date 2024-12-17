from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import TypeVarTuple

_Args = TypeVarTuple("_Args")
_Method = Literal[
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "COBYQA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]

class Bounds: ...
class LinearConstraint: ...
class NonLinearConstraint: ...
class OptimizeResult:
    success: bool
    x: NDArray[np.double]

def minimize(
    fun: Callable[[NDArray[np.generic]], NDArray[np.generic] | float],
    x0: ArrayLike,
    method: _Method = ...,
    bounds: Bounds | Sequence[tuple[float, float]] = ...,
) -> OptimizeResult:
    ...

def fmin_l_bfgs_b(
    func: Callable,
    x0: NDArray,
    fprime: Callable = ...,
    args: Sequence[object] = ...,
    approx_grad: bool = ...,
    bounds: Sequence[tuple[float, float]] = ...,
    m: int = ...,
    factr: float = ...,
    pgtol: float = ...,
    epsilon: float = ...,
    iprint: int = ...,
    disp: int = ...,
    maxfun: int = ...,
    maxiter: int = ...,
    callback: Callable = ...,
    maxls: int = ...,
) -> tuple[NDArray, float, dict]:
    ...
