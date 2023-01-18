import numpy as np
import numpy.typing as npt
from typing import List, Type
from .function import Fn


def compute_robustness(samples_in: npt.NDArray, mode:int, behavior:str, region, test_function: Type[Fn]) -> npt.NDArray:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in: Samples points for which the fitness is to be computed.
        test_function: Test Function insitialized with Fn
    Returns:
        Fitness (robustness) of the given sample(s)
    """
    if mode not in {0,1,2,3}:
        raise ValueError(f"Received mode = {mode}. Expected mode from set (0,1,2)")
    
    falsified = False
    if samples_in.shape[0] == 1:
        samples_out = np.array([test_function(samples_in[0], mode, region)])
        if samples_out < 0 and behavior == "Falsification":
            falsified = True
        print(f"{mode} --> {samples_out[0]}")
    else:
        samples_out = []
        for sample in samples_in:
            rob = test_function(sample, mode, region)
            samples_out.append(rob)
            print(f"{mode} --> {rob}")
            if rob < 0 and behavior == "Falsification":
                falsified = True
                break
            
        samples_out = np.array(samples_out)

    if behavior == "Minimization":
        falsified = False
    return samples_out, falsified