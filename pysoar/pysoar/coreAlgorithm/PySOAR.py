from copy import deepcopy
import numpy as np

from ..trustRegion import local_best_ei
from ..utils import Fn, compute_robustness, EIcalc_kd, CrowdingDist_kd, ei_cd, pointsInTR
from ..sampling import lhs_sampling, uniform_sampling
from ..gprInterface import GPR


import pathlib
import time
import pickle
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import NonlinearConstraint

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
# from pymoo.visualization.scatter import Scatter



##### v9 ####### add user defined parameters to input, break once falsified
def PySOAR(n_0, nSamples, trs_max_budget, max_loc_iter, inpRanges, alpha_lvl_set, eta0, eta1, delta, gamma, eps_tr, prob, gpr_model, seed, local_search, folder_name, benchmark_name, behavior = "Minimization"):
    base_path = pathlib.Path()
    results_directory = base_path.joinpath(folder_name)
    results_directory.mkdir(exist_ok=True)

    benchmark_directory = results_directory.joinpath(benchmark_name)
    benchmark_directory.mkdir(exist_ok=True)

    

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