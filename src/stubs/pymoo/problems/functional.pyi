from typing import Callable

from numpy.typing import NDArray

from pymoo.core.problem import Problem

class FunctionalProblem(Problem):
    def __init__(
        self,
        n_var: int,
        objs: list[Callable],
        xl: NDArray | float | int = ...,
        xu: NDArray | float | int = ...,
        constr_ieq: list = ...,
        constr_eq: list = ...,
        func_pf: Callable = ...,
        func_ps: Callable = ...,
    ): ...
