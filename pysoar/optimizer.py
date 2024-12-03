from __future__ import annotations

import pathlib
import pickle
import time
from copy import deepcopy
from typing import Any, Callable, Tuple, Type, Sequence

import numpy as np
import numpy.typing as npt
from attrs import frozen
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pyswarms.single import LocalBestPSO, GlobalBestPSO
from pyswarm import pso
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import NonlinearConstraint
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from .regions import local_best_ei
from .sampling import lhs_sampling, uniform_sampling
from .gpr import GPR


class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0
        self.point_history = []
        self.simultation_time = []
        self.mode_count = 0
        self.modes = []

    def __call__(self, sample, mode, region):
        self.count = self.count + 1
        sim_time_start = time.perf_counter()
        rob_val = self.func(sample)
        time_elapsed = time.perf_counter() - sim_time_start
        self.simultation_time.append(time_elapsed)

        if mode == 1:
            self.mode_count += 1

        self.modes.append(self.mode_count)
        self.point_history.append([self.count, sample, self.mode_count, mode, region, rob_val])
        return rob_val


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
        # print(f"{mode} --> {samples_out[0]}")
    else:
        samples_out = []
        for sample in samples_in:
            rob = test_function(sample, mode, region)
            samples_out.append(rob)
            # print(f"{mode} --> {rob}")
            if rob < 0 and behavior == "Falsification":
                falsified = True
                break
            
        samples_out = np.array(samples_out)

    if behavior == "Minimization":
        falsified = False
    return samples_out, falsified


def _surrogate(gpr_model: Callable, x_train: npt.NDArray):
    """_surrogate Model function

    Args:
        model: Gaussian process model
        X: Input points

    Returns:
        Predicted values of points using gaussian process model
    """

    return gpr_model.predict(x_train)


def EIcalc_kd(y_train: npt.NDArray, sample: npt.NDArray, gpr_model: Callable) -> npt.NDArray:
    """Acquisition Model: Expected Improvement

    Args:
        y_train: corresponding robustness values
        sample: Sample(s) whose EI is to be calculated
        gpr_model: GPR model
        sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

    Returns:
        EI of samples
    """
    curr_best = np.min(y_train)
    # print(sample.shape)
    if len(sample.shape) == 2:
        mu, std = _surrogate(gpr_model, sample)
        ei_list = []
        for mu_iter, std_iter in zip(mu, std):
            pred_var = std_iter
            if pred_var > 0:
                var_1 = curr_best - mu_iter
                var_2 = var_1 / pred_var

                ei = (var_1 * norm.cdf(var_2)) + (
                    pred_var * norm.pdf(var_2)
                )
            else:
                ei = 0.0

            ei_list.append(ei)
        return_ei = np.array(ei_list)
    elif len(sample.shape) == 1:
        
        mu, std = _surrogate(gpr_model, sample.reshape(1, -1))
        pred_var = std[0]
        if pred_var > 0:
            var_1 = curr_best - mu[0]
            var_2 = var_1 / pred_var

            ei = (var_1 * norm.cdf(var_2)) + (
                pred_var * norm.pdf(var_2)
            )
        else:
            ei = 0.0
        return_ei = ei
        

    return return_ei


def CrowdingDist_kd(x_0, x):
   
    # cd = np.sum(np.min(np.abs(x_0-x), 0))
    
    # cd = np.array(cd)
    for sample in x_0:
        cd = np.sum(np.sqrt(np.sum((sample - x)**2, 1)))
    return cd


def ei_cd(samples, x_train, y_train, gpr, alpha_lvl_set, EI_star):

    if len(samples.shape) == 1:
        c = EIcalc_kd(y_train, samples, gpr) - (alpha_lvl_set * (EI_star))
        if c >= 0:
            ret = -1 * CrowdingDist_kd(samples, x_train)
        else:
            ret = 0
    elif len(samples.shape) > 1:
        ret = []
        # print(x)
        for sample in samples:
            c = EIcalc_kd(y_train, sample, gpr) - (alpha_lvl_set * (EI_star))
            # print(c[i])
            if c >= 0:
                ret.append(-1 * CrowdingDist_kd(sample, x_train))
            else:
                ret.append(0)
        ret = np.array(ret)
    return ret


def pointsInTR(samples_in: np.array, samples_out:np.array, subregion: list) -> list:
    """

    Args:
        samples_in: Samples from Training set.
        samples_out: Evaluated values of samples from Training set.
        region_support: Min and Max of all dimensions

    Returns:
        list: Divided samples
    """    
    regionSamples = []
    corresponding_robustenss = []
    if samples_in.shape[0] == samples_out.shape[0] and  samples_out.shape[0] != 0:

        
        boolArray = []
        for dimension in range(len(subregion)):
            subArray = samples_in[:, dimension]
            logical_subArray = np.logical_and(subArray >= subregion[dimension, 0], subArray <= subregion[dimension, 1])
            boolArray.append(np.squeeze(logical_subArray))
        corresponding_robustenss = samples_out[(np.all(boolArray, axis = 0))]
        regionSamples = samples_in[(np.all(boolArray, axis = 0)),:]
    else:
        
        corresponding_robustenss = np.array([])
        regionSamples = np.array([[]])
            
    return regionSamples, corresponding_robustenss


##### v9 ####### add user defined parameters to input, break once falsified
def PySOAR(n_0, nSamples, trs_max_budget, max_loc_iter, inpRanges, alpha_lvl_set, eta0, eta1, delta, gamma, eps_tr, prob, gpr_model, seed, local_search, behavior = "Minimization"):  

    t = time.time()
    rng = np.random.default_rng(seed)
    np.random.seed(seed+1000)

    tf_dim = inpRanges.shape[0]
    tf_wrapper = Fn(prob)
    falsified = False
    if n_0 > nSamples:
        raise ValueError(f"Received n_0({n_0}) > nSamples ({nSamples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (nSamples)")

    x_train = lhs_sampling(n_0, inpRanges, tf_dim, rng)
    y_train, falsified = compute_robustness(x_train, 0, behavior, inpRanges, tf_wrapper)

    if falsified:
        # with open(benchmark_directory.joinpath(f"{benchmark_name}_seed_{seed}.pkl"), "wb") as f:
        #     pickle.dump((tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time), f)
        return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)

    
    while (tf_wrapper.count < nSamples) and (not falsified):
        print(f"{tf_wrapper.count} Evaluations completed -> {x_train.shape}, {y_train.shape}")
        gpr = GPR(deepcopy(gpr_model))
        gpr.fit(x_train, y_train)

        
        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])
        
        objs = [lambda x: -1 * EIcalc_kd(y_train, x, gpr), lambda x: -1 * CrowdingDist_kd(x, x_train)]
        problem = FunctionalProblem(inpRanges.shape[0], objs, xl = lower_bound_theta, xu = upper_bound_theta)
        
        algorithm = NSGA2(
            pop_size = 50,
            sampling = FloatRandomSampling(),
            crossover = SBX(prob = 0.9, eta = 15),
            mutation = PM(eta=20),
            eliminate_duplicates = True
        )
        res = minimize(problem, algorithm, ('n_gen', 50), seed = rng.integers(low = 1,high = 100000, size = 1)[0], verbose = False)
        F = res.F
        X = res.X
        minNegEIindex = np.argmin(res.F[:,0])
        minNegEI = F[minNegEIindex, 0]
        x0 = X[minNegEIindex]
        best_crowd = float("inf")
        
        for k in range(F.shape[0]):
            
            if F[k,0] <= minNegEI * (1-alpha_lvl_set):
                if F[k,1] < best_crowd:
                    best_crowd = F[k,1]
                    x0 = X[k,:]
        
        
        pred_sample_x = np.array([np.array(x0)])
        
        pred_sample_y, falsified = compute_robustness(pred_sample_x, 1, behavior, inpRanges, tf_wrapper)
        
        if falsified:
            # with open(benchmark_directory.joinpath(f"{benchmark_name}_seed_{seed}.pkl"), "wb") as f:
            #     pickle.dump((tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time), f)
            return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)

        x_train = np.vstack((x_train, pred_sample_x))
        y_train = np.hstack((y_train, pred_sample_y))
        
        ######### LOCAL SEARCH PHASE ###########
        restart_point_x, restart_point_y = deepcopy(pred_sample_x), deepcopy(pred_sample_y)

        # Initialize TR Bounds
        TR_Bounds = np.vstack(
            [restart_point_x[0,:] - inpRanges[:, 0], inpRanges[:, 1] - restart_point_x[0,:], (inpRanges[:, 1] - inpRanges[:, 0]) / 10]).flatten()
        
        

        TR_size = np.min(np.abs(TR_Bounds[TR_Bounds!=0]))
        
        trust_region = np.empty((inpRanges.shape))
        for d in range(tf_dim): 
            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])

        
        x_train_subset, y_train_subset = pointsInTR(x_train, y_train, trust_region)
        num_points_present = x_train_subset.shape[0]
        ####### Enter TR Meta Model Loop ######
        local_counter = 0
        
        

        if local_search == "gp_local_search":
            
            # print("LS")
            
            while (local_counter < max_loc_iter 
                     and TR_size > eps_tr * np.min(inpRanges[:, 1] - inpRanges[:,0])
                    and tf_wrapper.count + (max(trs_max_budget - num_points_present,0) + 1) < nSamples):
                
                if trs_max_budget - num_points_present-1 > 0:
                    
                    num_samples_needed = trs_max_budget - num_points_present
                    
                    # draw a new lhs over the current TR
                    x0_local = lhs_sampling(num_samples_needed, trust_region, tf_dim, rng)
                    y0_local, falsified = compute_robustness(x0_local, 2, behavior, trust_region, tf_wrapper)
                    if falsified:
                        
                        return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)
                    
                    x_train = np.vstack((x_train, x0_local))
                    y_train = np.hstack((y_train, y0_local))

                    x_train_subset = np.vstack((x_train_subset, x0_local))
                    y_train_subset = np.hstack((y_train_subset, y0_local))
                    
                # Fit Gaussian Process Meta Model Locally
                
                xk, fk, rho, falsified = local_best_ei(restart_point_x, restart_point_y, tf_wrapper, tf_dim, trust_region, x_train_subset, y_train_subset, behavior, gpr_model, rng)
                print(xk, fk)
                if falsified:
                    # with open(benchmark_directory.joinpath(f"{benchmark_name}_seed_{seed}.pkl"), "wb") as f:
                    #     pickle.dump((tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time), f)
                    return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)
                x_train = np.vstack((x_train, xk))
                y_train = np.hstack((y_train, fk))

                x_train_subset = np.vstack((x_train_subset, xk))
                y_train_subset = np.hstack((y_train_subset, fk))
                
                # print(xk, fk, rho, falsified)
                
                
                max_indicator = np.max(np.abs(xk - restart_point_x)) / TR_size
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
                        restart_point_x = xk
                        restart_point_y = fk
                        
                        valid_bound = np.array([np.min(np.abs(restart_point_x[0,:] - inpRanges[:, 0])), np.min(np.abs(inpRanges[:, 1] - restart_point_x[0,:])), TR_size]).flatten()
                        TR_size = np.min(valid_bound[valid_bound!=0])
                        trust_region = np.empty((inpRanges.shape))
                        
                        for d in range(tf_dim): 
                            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
                            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])
                    else:
                        # high pass of RC test
                        restart_point_x = xk
                        restart_point_y = fk
                        valid_bound = np.array([np.min(np.abs(restart_point_x[0,:] - inpRanges[:, 0])), np.min(np.abs(inpRanges[:, 1] - restart_point_x[0,:])), TR_size*gamma]).flatten()
                        # TR_size *= gamma
                        TR_size = np.min(valid_bound[valid_bound!=0])
                        trust_region = np.empty((inpRanges.shape))
                        
                        for d in range(tf_dim): 
                            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
                            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])

                local_counter += 1
                x_train_subset, y_train_subset = pointsInTR(x_train, y_train, trust_region)
                num_points_present = x_train_subset.shape[0]
                
                
                # check if budget has been exhausted

    # with open(benchmark_directory.joinpath(f"{benchmark_name}_seed_{seed}.pkl"), "wb") as f:
    #     pickle.dump((tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time), f)
    print(f"{tf_wrapper.count} Evaluations completed -> {x_train.shape}, {y_train.shape}")
    return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)


##### v9 ####### add user defined parameters to input, break once falsified
def PySOAR_Timed(n_0, nSamples, trs_max_budget, inpRanges, alpha_lvl_set, eta0, eta1, delta, gamma, eps_tr, prob, gpr_model, seed, local_search, folder_name, benchmark_name, behavior = "Minimization"):
    t = time.time()
    rng = np.random.default_rng(seed)
    np.random.seed(seed+1000)

    tf_dim = inpRanges.shape[0]
    tf_wrapper = Fn(prob)
    falsified = False
    if n_0 > nSamples:
        raise ValueError(f"Received n_0({n_0}) > nSamples ({nSamples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (nSamples)")

    x_train = lhs_sampling(n_0, inpRanges, tf_dim, rng)
    
    y_train, falsified = compute_robustness(x_train, 0, behavior, inpRanges, tf_wrapper)

    if falsified:
        return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)

    
    while (tf_wrapper.count < nSamples) and (not falsified):
        print(f"{tf_wrapper.count} Evaluations completed")
        gpr = GPR(deepcopy(gpr_model))
        gpr.fit(x_train, y_train)

        print("*********************************************************************")
        print("*********************************************************************")
        print("optimize EI")

        EI_obj = lambda x: -1*EIcalc_kd(y_train, x, gpr)
        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])

        max_bound = inpRanges[:, 1]
        min_bound = inpRanges[:, 0]
        bounds = (min_bound, max_bound)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        t = time.time()
        optimizer = GlobalBestPSO(n_particles=n_0, dimensions=tf_dim, options=options, bounds=bounds)

        gpso_val, gpso_x = optimizer.optimize(EI_obj, iters=200)
        t_gpso = time.time() - t
        
        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])
        t = time.time()
        random_samples = uniform_sampling(2000, inpRanges, tf_dim, rng)
        min_bo_val = EI_obj(random_samples)
        
        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])
        min_bo_val = np.min(min_bo_val)
        # print(min_bo_val)
        for _ in range(9):
            new_params = minimize(
                EI_obj,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            # if not new_params.success:
            #     continue

            if min_bo is None or EI_obj(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = EI_obj(min_bo)
            
            # print(min_bo_val)
        new_params = minimize(
            EI_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        local_x = new_params.x
        local_val = EI_obj(min_bo)
        t_local = time.time() - t

        print("**************************************************************")
        print(f"Values from GPSO: {gpso_val}, {gpso_x}")
        print(f"Time for GPSO = {t_gpso}")
        print("**************************************************************")
        print(f"Values from LOCAL: {local_val}, {local_x}")
        print(f"Time for Local = {t_local}")
        print("**************************************************************")

        EI_obj = lambda x: -1*EIcalc_kd(y_train, x, gpr)
        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])
        
        random_samples = uniform_sampling(2000, inpRanges, tf_dim, rng)
        min_bo_val = EI_obj(random_samples)

        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])
        min_bo_val = np.min(min_bo_val)

        for _ in range(9):
            new_params = minimize(
                EI_obj,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            

            if min_bo is None or EI_obj(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = EI_obj(min_bo)

            if not new_params.success:
                continue

        new_params = minimize(
            EI_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        EI_star_x = new_params.x
        EI_star = -1 * EI_obj(EI_star_x) #actual problem is that of maximization, so earlier we multipled by -1
        # print("********************************************************")
        # print(EI_star, EI_star_x)

        ########################################################################################
        print("*********************************************************************")
        print("*********************************************************************")
        print("optimize EI CD")

        const = lambda x: EIcalc_kd(y_train, x, gpr) - (alpha_lvl_set * (EI_star))
        CD_obj = lambda x: -1 * CrowdingDist_kd(x, x_train)
        
        opt_obj = lambda x: ei_cd(x, x_train, y_train, gpr, alpha_lvl_set, EI_star)
        lb = inpRanges[:, 0]
        ub = inpRanges[:, 1]
        t = time.time()
        x0, x_val = pso(CD_obj, lb, ub, f_ieqcons=const, maxiter=200)
        t_original = time.time() - t

        max_bound = inpRanges[:, 1]
        min_bound = inpRanges[:, 0]
        bounds = (min_bound, max_bound)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        t = time.time()
        optimizer = GlobalBestPSO(n_particles=n_0, dimensions=tf_dim, options=options, bounds=bounds)

        gpso_val, gpso_x = optimizer.optimize(opt_obj, iters=200)
        t_gpso = time.time() - t
        
        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])
        t = time.time()
        random_samples = uniform_sampling(2000, inpRanges, tf_dim, rng)
        min_bo_val = opt_obj(random_samples)
        
        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])
        min_bo_val = np.min(min_bo_val)
        # print(min_bo_val)
        for _ in range(9):
            new_params = minimize(
                opt_obj,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            # if not new_params.success:
            #     continue

            if min_bo is None or opt_obj(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = opt_obj(min_bo)
            
            # print(min_bo_val)
        new_params = minimize(
            opt_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        local_x = new_params.x
        local_val = opt_obj(min_bo)
        t_local = time.time() - t

        print("**************************************************************")
        print(f"Values from original: {x_val}, {x0}")
        print(f"Time for original = {t_original}")
        print("**************************************************************")
        print(f"Values from GPSO: {gpso_val}, {gpso_x}")
        print(f"Time for GPSO = {t_gpso}")
        print("**************************************************************")
        print(f"Values from LOCAL: {local_val}, {local_x}")
        print(f"Time for GPSO = {t_local}")
        print("**************************************************************")
        ########################################################################################
        opt_obj = lambda x: ei_cd(x, x_train, y_train, gpr, alpha_lvl_set, EI_star)
        min_eicd_val = opt_obj(random_samples)
        
        min_eicd = np.array([random_samples[np.argmin(min_eicd_val), :]])
        min_eicd_val = np.min(min_eicd_val)

        for _ in range(9):
            new_params = minimize(
                opt_obj,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_eicd,
            )

            if min_eicd is None or opt_obj(new_params.x) < min_eicd_val:
                min_eicd = new_params.x
                min_eicd_val = opt_obj(min_eicd)
            
            if not new_params.success:
                continue
                
        new_params = minimize(
            opt_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_eicd
        )
        pred_sample_x = np.array([np.array(new_params.x)])
        
        pred_sample_y, falsified = compute_robustness(pred_sample_x, 1, behavior, inpRanges, tf_wrapper)
        if falsified:
            return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)
        x_train = np.vstack((x_train, pred_sample_x))
        y_train = np.hstack((y_train, pred_sample_y))
        
        ######### LOCAL SEARCH PHASE ###########
        restart_point_x, restart_point_y = deepcopy(pred_sample_x), deepcopy(pred_sample_y)

        # Initialize TR Bounds
        TR_Bounds = np.vstack(
            [restart_point_x[0,:] - inpRanges[:, 0], inpRanges[:, 1] - restart_point_x[0,:], (inpRanges[:, 1] - inpRanges[:, 0]) / 10])
        """
        x = [[1,2]]
        inpRanges = [[-6,0],[-6,6]]
        min([7,8], [1,-4] [0.6, 1.2])

        """
        

        TR_size = np.min(np.abs(TR_Bounds))
        trust_region = np.empty((inpRanges.shape))
        for d in range(tf_dim): 
            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])

        
        x_train_subset, y_train_subset = pointsInTR(x_train, y_train, trust_region)
        num_points_present = x_train_subset.shape[0]
        ####### Enter TR Meta Model Loop ######
        local_counter = 0
        max_loc_iter = trs_max_budget
        

        if local_search == "gp_local_search":
            while (local_counter <= max_loc_iter 
                    and TR_size > eps_tr * np.min((inpRanges[:, 1] - inpRanges[:,0])) 
                    and tf_wrapper.count + (max(max_loc_iter - num_points_present,0) + 1) <= nSamples):
                
                
                if max_loc_iter - num_points_present > 0:
                    num_samples_needed = max_loc_iter - num_points_present
                    # draw a new lhs over the current TR
                    x0_local = lhs_sampling(num_samples_needed, trust_region, tf_dim, rng)
                    y0_local, falsified = compute_robustness(x0_local, 2, behavior, trust_region, tf_wrapper)
                    if falsified:
                        return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)

                    x_train = np.vstack((x_train, x0_local))
                    y_train = np.hstack((y_train, y0_local))

                    x_train_subset = np.vstack((x_train_subset, x0_local))
                    y_train_subset = np.hstack((y_train_subset, y0_local))
                
                # Fit Gaussian Process Meta Model Locally
                
                xk, fk, rho, falsified = local_best_ei(restart_point_x, restart_point_y, tf_wrapper, tf_dim, trust_region, x_train_subset, y_train_subset, behavior, gpr_model, rng)
                if falsified:
                    return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)
                x_train = np.vstack((x_train, xk))
                y_train = np.hstack((y_train, fk))

                x_train_subset = np.vstack((x_train_subset, xk))
                y_train_subset = np.hstack((y_train_subset, fk))
                # print(xk, fk, rho, falsified)
                
                # """ What the use of this?
                max_indicator = np.max(np.abs(xk - restart_point_x)) / TR_size
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
                        restart_point_x = xk
                        restart_point_y = fk
                        
                        # valid_bound = np.min([np.min(np.abs(restart_point_x[0,:] - inpRanges[:, 0])), np.min(np.abs(inpRanges[:, 1] - restart_point_x[0,:])), TR_size])
                    else:
                        # high pass of RC test
                        restart_point_x = xk
                        restart_point_y = fk
                        # valid_bound = np.min([np.min(np.abs(restart_point_x[0,:] - inpRanges[:, 0])), np.min(np.abs(inpRanges[:, 1] - restart_point_x[0,:])), TR_size*gamma])
                        TR_size *= gamma
                        # TR_size = np.min(valid_bound)
                        trust_region = np.empty((inpRanges.shape))
                        
                        for d in range(tf_dim): 
                            trust_region[d, 0] = max(restart_point_x[0,d] - TR_size, inpRanges[d,0])
                            trust_region[d, 1] = min(restart_point_x[0,d] + TR_size, inpRanges[d,1])

                local_counter += 1
                x_train_subset, y_train_subset = pointsInTR(x_train, y_train, trust_region)
                num_points_present = x_train_subset.shape[0]
                # check if budget has been exhausted

    base_path = pathlib.Path()
    results_directory = base_path.joinpath(folder_name)
    results_directory.mkdir(exist_ok=True)

    benchmark_directory = results_directory.joinpath(benchmark_name)
    benchmark_directory.mkdir(exist_ok=True)

    with open(benchmark_directory.joinpath(f"{benchmark_name}_seed_{seed}.pkl"), "wb") as f:
        pickle.dump((tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time), f)

    return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)


Bounds = Sequence[Interval]


@frozen(slots=True)
class PySOARResult:
    history: Any
    modes: Any
    simulation_times: Any


@frozen()
class run_pysoar(Optimizer[PySOARResult, None]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""

    # inpRanges: 
    
    n_0: int
    trs_max_budget: int
    max_loc_iter:int
    alpha_lvl_set: float
    eta0: float
    eta1: float
    delta: float
    gamma: float
    eps_tr: float
    gpr_model: Callable
    local_search: str
    behavior: str

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> PySOARResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))[0]
        
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))
        
        history, modes, sim_times = PySOAR(n_0=self.n_0, 
                            nSamples=budget, 
                            trs_max_budget=self.trs_max_budget,
                            max_loc_iter=self.max_loc_iter,
                            inpRanges= region_support,
                            alpha_lvl_set=self.alpha_lvl_set, 
                            eta0=self.eta0, 
                            eta1=self.eta1, 
                            delta=self.delta, 
                            gamma=self.gamma, 
                            eps_tr=self.eps_tr, 
                            prob= test_function,
                            gpr_model=self.gpr_model,
                            seed = seed,
                            local_search=self.local_search,
                            behavior=self.behavior
                        )

        return PySOARResult(history, modes, sim_times)
