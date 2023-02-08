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
        rob, hd = test_function(samples_in[0], mode, region)
        samples_out = np.array([rob])
        hybrid_dist = np.array([hd])
        if samples_out < 0 and behavior == "Falsification":
            falsified = True
        print(f"{mode} --> {rob}, {hd}\t sample - {samples_in[0]}")
    else:
        samples_out = []
        hybrid_dist = []
        for sample in samples_in:
            rob, hd = test_function(sample, mode, region)
            samples_out.append(rob)
            hybrid_dist.append(hd)
            print(f"{mode} --> {rob}, {hd}\t sample - {sample}")
            if rob < 0 and behavior == "Falsification":
                falsified = True
                break
            
        samples_out = np.array(samples_out)
        hybrid_dist = np.array(hybrid_dist)

    if behavior == "Minimization":
        falsified = False
    return samples_out, hybrid_dist, falsified