"""

"""



from copy import deepcopy
import numpy as np
import math

from ..trustRegion import local_gp_tr, gradient_based_tr, local_best_ei
from ..utils import Fn, compute_robustness, EIcalc_kd, CrowdingDist_kd, ei_cd, pointsInTR
from ..sampling import lhs_sampling, uniform_sampling
from ..gprInterface import GPR

from pyswarms.single import LocalBestPSO, GlobalBestPSO
from pyswarm import pso

import pathlib
import time
import pickle
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import NonlinearConstraint


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