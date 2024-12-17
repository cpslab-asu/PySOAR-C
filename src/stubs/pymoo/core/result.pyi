from numpy.typing import NDArray

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population

class Result:
    X: NDArray
    F: NDArray
    G: NDArray
    CV: NDArray
    algorithm: Algorithm
    pop: Population
