from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from numpy import float_
from numpy.typing import NDArray, ArrayLike
from typing_extensions import TypeAlias

from sklearn.gaussian_process.kernels import Kernel

_ObjFunc: TypeAlias = Callable[[float, bool], float]
_Optimizer: TypeAlias = Callable[[_ObjFunc, float, Iterable[Sequence[float]]], tuple[float, float]]

class GaussianProcessRegressor:
    def __init__(
        self,
        kernel: Kernel = ...,
        alpha: float | NDArray[float_] = ...,
        optimizer: Literal["fmin_l_bfgs_b"] | _Optimizer | None = ...,
        n_restarts_optimizer: int = ...,
        normalize_y: bool = ...,
        copy_X_train: bool = ...,
        n_targets: int = ...,
        random_state: int = ...,
    ): ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> GaussianProcessRegressor: ...
    def predict(
        self,
        X: ArrayLike,
        return_std: bool = ...,
        return_cov: bool = ...,
    ) -> tuple[NDArray[float_], NDArray[float_], NDArray[float_]]:
        ...
