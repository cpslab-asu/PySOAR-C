import math
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from .behavior import Behavior
from ..trustRegion import local_best_ei
from ..utils import Fn, EIcalc_kd, CrowdingDist_kd, pointsInTR
from ..sampling import lhs_sampling, uniform_sampling
from ..gprInterface import GPR, GaussianProcessRegressor

_logger = logging.getLogger("PySOAR-C")




@dataclass(frozen=True)
class LocalBest:
    local_best_x: Any
    local_best_y: Any

@dataclass(frozen=True)
class LocalPhase:
    region_support: Any
    local_phase_x: Any
    local_phase_y: Any
    

@dataclass(frozen=True)
class GlobalPhase:
    restart_point_x: Any
    restart_point_y: Any
    
    

@dataclass(frozen=True)
class InitializationPhase:
    initial_samples_x: NDArray[np.double]
    initial_samples_y: NDArray[np.double]

def _generate_dataset(output_type: int, *args):
    if output_type not in [0,1]:
        raise ValueError
    
    
    if type(args[0][0]) == InitializationPhase:
        x_train = args[0][0].initial_samples_x
        y_train = args[0][0].initial_samples_y[:,output_type]
    else:
        raise ValueError

    for arg in args[0][1:]:
        if type(arg) == GlobalPhase:
            x_train = np.vstack((x_train, arg.restart_point_x))
            y_train = np.hstack((y_train, arg.restart_point_y[:,output_type]))
        elif type(arg) == LocalPhase:
            x_train = np.vstack((x_train, arg.local_phase_x))
            y_train = np.hstack((y_train, arg.local_phase_y[:,output_type]))
        elif type(arg) == LocalBest:
            x_train = np.vstack((x_train, arg.local_best_x))
            y_train = np.hstack((y_train, arg.local_best_y[:,output_type]))
    
    return x_train, y_train


def _is_falsification(evaluation: NDArray) -> bool:
    if evaluation is None:
        return True
    return evaluation[0] == 0 and evaluation[1] < 0


class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *arg):
        self.count = self.count + 1
        
        hybrid_dist = self.func(*arg)
        print(self.count, arg[0], hybrid_dist)
        
        return hybrid_dist

def _evaluate_samples(
                samples: NDArray[np.double], 
                fn: Callable[[NDArray[np.double]], NDArray[np.double]],
                behavior: Behavior
) -> NDArray:

    evaluations = []
    
    modified_x_train = []
    for sample in samples:
        evaluation = np.array(fn(sample), dtype=np.double)
        
        if _is_falsification(evaluation): 
            if behavior is Behavior.FALSIFICATION:
                evaluations.append(evaluation)
                modified_x_train.append(sample)
                break
            elif behavior is Behavior.COVERAGE:
                evaluations.append(evaluation)
                modified_x_train.append(sample)
                break   
        else:
            evaluations.append(evaluation)
            modified_x_train.append(sample)

    return np.array(modified_x_train), np.array(evaluations)


##### v9 ####### add user defined parameters to input, break once falsified
def PySOARC(
    n_0: int,
    nSamples: int,
    trs_max_budget: int,
    max_loc_iter: int,
    inpRanges: ArrayLike, 
    alpha_lvl_set: float,
    eta0: float,
    eta1: float,
    delta: float,
    gamma: float,
    eps_tr: float,
    min_tr_size: float,
    TR_threshold: float,
    test_fn: Callable[[NDArray[np.double]], NDArray[np.double]],
    gpr_model: GaussianProcessRegressor, 
    seed: int, 
    local_search: str, 
    behavior: Behavior = Behavior.FALSIFICATION
):
    inpRanges = np.array(inpRanges)
    test_fn = Fn(test_fn)
    if inpRanges.ndim != 2:
        raise ValueError("input ranges should be 2-dimensional")

    if inpRanges.shape[1] != 2:
        raise ValueError("input range 2nd dimension should be equal to 2")

    rng = np.random.default_rng(seed)
    np.random.seed(seed+1000)

    tf_dim = inpRanges.shape[0]
    if n_0 > nSamples:
        raise ValueError(f"Received n_0({n_0}) > nSamples ({nSamples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (nSamples)")

    initial_samples = lhs_sampling(n_0, inpRanges, tf_dim, rng)
    # inital_samples_hd = initial_samples
    initial_samples, initial_sample_distances = _evaluate_samples(initial_samples, test_fn, behavior)
    
    initial_points = InitializationPhase(
                        initial_samples_x = initial_samples,
                        initial_samples_y = initial_sample_distances
                    )
    algo_journey = [initial_points]
    if any(_is_falsification(sd) for sd in initial_sample_distances) and (behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE):
        # TODO: Create return structure
        return algo_journey

    while test_fn.count < nSamples:
        
        
        x_train, y_train = _generate_dataset(1, algo_journey)
        print(f"{test_fn.count} Evaluations completed -> {x_train.shape}, {y_train.shape}")
        gpr = GPR(deepcopy(gpr_model))
        gpr.fit(x_train, y_train)

        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])
        
        problem = FunctionalProblem(
            n_var=inpRanges.shape[0],
            objs=[
                lambda x: -1 * EIcalc_kd(y_train, x, gpr),
                lambda x: -1 * CrowdingDist_kd(x, x_train)
            ],
            xl=lower_bound_theta,
            xu=upper_bound_theta
        )
        
        algorithm = NSGA2(
            pop_size=150,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob = 0.9, eta = 15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        ga_seed = rng.integers(low=1, high=100000, size=1)
        ga_result = minimize(
            problem = problem, 
            algorithm = algorithm, 
            termination = ('n_gen', 200), 
            seed = ga_seed[0], 
            verbose = False
        )
                    
        minNegEIindex = np.argmin(ga_result.F[:,0])
        minNegEI = ga_result.F[minNegEIindex, 0]
        global_rp_x = ga_result.X[minNegEIindex]
        best_crowd = math.inf
        
        for k in range(ga_result.F.shape[0]):
            if ga_result.F[k, 0] <= minNegEI * (1 - alpha_lvl_set):
                if ga_result.F[k, 1] < best_crowd:
                    _logger.debug(f"{ga_result.F[k, 0]} <= {minNegEI * (1-alpha_lvl_set)} -> {ga_result.F[k,0] <= minNegEI * (1-alpha_lvl_set)} \\ {ga_result.F[k,1]} < {best_crowd} -> {ga_result.F[k,1] < best_crowd} \n{global_rp_x}\n*************************************************")
                    best_crowd = ga_result.F[k, 1]
                    global_rp_x = ga_result.X[k, :]
        global_rp_x = np.array([global_rp_x])
        global_rp_x, global_rp_y = _evaluate_samples(global_rp_x, test_fn, behavior)
        algo_journey.append(GlobalPhase(global_rp_x, global_rp_y))
        
        if _is_falsification(global_rp_y[0]) and (behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE):
            # TODO
            return algo_journey
        
        local_sample_x, local_samples_y = _generate_dataset(0, algo_journey)
        
        TR_Bounds = np.vstack(
            [    global_rp_x[0,:]- inpRanges[:, 0], 
                inpRanges[:, 1] - global_rp_x[0,:], 
                (inpRanges[:, 1] - inpRanges[:, 0]) / min_tr_size
            ]
        ).flatten()

        
        TR_size = np.min(np.abs(TR_Bounds[TR_Bounds>=TR_threshold]))
        
        trust_region = np.empty((inpRanges.shape))
        for d in range(tf_dim): 
            trust_region[d, 0] = max(global_rp_x[0,d] - TR_size, inpRanges[d,0])
            trust_region[d, 1] = min(global_rp_x[0,d] + TR_size, inpRanges[d,1])

        local_sample_x_subset, local_sample_y_subset = pointsInTR(local_sample_x, local_samples_y, trust_region)
        num_points_present = local_sample_x_subset.shape[0]
        
        local_counter = 0
        restart_point_x, restart_point_y = deepcopy(global_rp_x), deepcopy(global_rp_y)

        if local_search == "gp_local_search":
            
            while (local_counter < max_loc_iter 
                    and TR_size > eps_tr * np.min(inpRanges[:, 1] - inpRanges[:,0])
                    and test_fn.count + (max(trs_max_budget - num_points_present,0) + 1) < nSamples):
                
                # print(f"Needed: {trs_max_budget}, present: {num_points_present}, More {num_samples_needed} points needed")
                if trs_max_budget - num_points_present > 0:
                    num_samples_needed = trs_max_budget - num_points_present
                    
                    local_additional_x = lhs_sampling(num_samples_needed, trust_region, tf_dim, rng)
                    local_additional_x, local_additional_y = _evaluate_samples(local_additional_x, test_fn, behavior)
                    
                    algo_journey.append(LocalPhase(trust_region, local_additional_x, local_additional_y))
                    
                    if any(_is_falsification(sd) for sd in local_additional_y) and (behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE):
                        return algo_journey               
                
                x_train_hd, y_train_hd = _generate_dataset(0, algo_journey)
                local_sample_x_subset, local_sample_y_subset = pointsInTR(x_train_hd, y_train_hd, trust_region)
                
                # Fit Gaussian Process Meta Model Locally
                
                local_best_x, local_best_y, rho = local_best_ei(
                                                    restart_point_x, 
                                                    restart_point_y, 
                                                    _evaluate_samples,
                                                    test_fn, 
                                                    tf_dim, 
                                                    trust_region, 
                                                    local_sample_x_subset, 
                                                    local_sample_y_subset, 
                                                    behavior, 
                                                    gpr_model, 
                                                    rng
                )
                algo_journey.append(LocalBest(local_best_x, local_best_y))
                if _is_falsification(local_best_y[0]) and (behavior is Behavior.FALSIFICATION):
                    return algo_journey

                max_indicator = np.max(np.abs(local_best_x - restart_point_x)) / TR_size
                test = rng.random()
                if max_indicator < test:
                    break

                # execute RC testing and TR control
                if rho < eta0:
                    TR_size *= delta
                    trust_region = np.empty((inpRanges.shape))
                    for d in range(tf_dim): 
                        trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
                        trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])
                else:
                    if eta0 < rho < eta1:
                        # low pass of RC test
                        restart_point_x = local_best_x
                        restart_point_y = local_best_y
                        
                        valid_bound = np.array([
                                        np.min(np.abs(restart_point_x[0,:] - inpRanges[:, 0])), 
                                        np.min(np.abs(inpRanges[:, 1] - restart_point_x[0,:])), 
                                        TR_size
                                    ]).flatten()
                        TR_size = np.min(valid_bound[valid_bound>=TR_threshold])
                        trust_region = np.empty((inpRanges.shape))
                        
                        for d in range(tf_dim): 
                            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
                            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])
                    else:
                        # high pass of RC test
                        restart_point_x = local_best_x
                        restart_point_y = local_best_y
                        valid_bound = np.array([
                                        np.min(np.abs(restart_point_x[0,:] - inpRanges[:, 0])), 
                                        np.min(np.abs(inpRanges[:, 1] - restart_point_x[0,:])), 
                                        TR_size*gamma
                                    ]).flatten()
                        # TR_size *= gamma
                        TR_size = np.min(valid_bound[valid_bound>=TR_threshold])
                        trust_region = np.empty((inpRanges.shape))
                        
                        for d in range(tf_dim): 
                            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
                            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])

                local_counter += 1
                x_train_hd, y_train_hd = _generate_dataset(0,algo_journey)

                local_sample_x_subset, local_sample_y_subset = pointsInTR(x_train_hd, y_train_hd, trust_region)
                num_points_present = local_sample_x_subset.shape[0]
                
                # print(f"{TR_size} ---- {eps_tr * np.min(inpRanges[:, 1] - inpRanges[:,0])}")
                
                # check if budget has been exhausted

    return algo_journey