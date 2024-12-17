from pymoo.core.algorithm import Algorithm
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
from pymoo.util.display.output import Output

class NSGA2(Algorithm):
    def __init__(
        self,
        pop_size: int = ...,
        sampling: Sampling = ...,
        selection: Selection = ...,
        crossover: Crossover = ...,
        mutation: Mutation = ...,
        survival: Survival = ...,
        output: Output = ...,
        eliminate_duplicates: DuplicateElimination | bool = ...,
    ):
        ...
