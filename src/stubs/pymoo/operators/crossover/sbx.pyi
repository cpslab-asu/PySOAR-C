from pymoo.core.crossover import Crossover

class SBX(Crossover):
    def __init__(
        self,
        prob: float = ...,
        prob_var: float = ...,
        eta: int = ...,
        prob_exch: float = ...,
        prob_bin: float = ...,
        n_offsprings: int = ...,
    ):
        ...
