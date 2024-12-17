from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.core.termination import Termination
from pymoo.util.display.display import Display

def minimize(
    problem: Problem,
    algorithm: Algorithm,
    termination: Termination | tuple = ...,
    seed: int = ...,
    verbose: bool = ...,
    display: Display = ...,
    save_history: bool = ...,
    callback: Callback = ...,
    return_least_infeasible: bool = ...,
) -> Result:
    ...
