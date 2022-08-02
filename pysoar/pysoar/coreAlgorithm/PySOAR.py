"""

"""



from copy import deepcopy
import numpy as np
import math

from ..trustRegion import local_gp_tr, gradient_based_tr
from ..utils import Fn, compute_robustness, EIcalc_kd, CrowdingDist_kd, ei_cd
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
def PySOAR(n_0, nSamples, trs_max_budget, inpRanges, alpha_lvl_set, eta0, eta1, delta, gamma, eps_tr, prob, gpr_model, seed, local_search, folder_name, benchmark_name, behavior = "Minimization"):
    t = time.time()
    rng = np.random.default_rng(seed)
    np.random.seed(seed+1000)

    tf_dim = inpRanges.shape[0]
    tf_wrapper = Fn(prob)
    falsified = False
    if n_0 > nSamples:
        raise ValueError(f"Received n_0({n_0}) > nSamples ({nSamples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (nSamples)")

    x_train = lhs_sampling(n_0, inpRanges, tf_dim, rng)
    
    y_train, falsified = compute_robustness(x_train, 0, behavior, tf_wrapper)

    if falsified:
        return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)

    
    while (tf_wrapper.count < nSamples) and (not falsified):
        
        gpr = GPR(deepcopy(gpr_model))
        gpr.fit(x_train, y_train)

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

            if not new_params.success:
                continue

            if min_bo is None or EI_obj(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = EI_obj(min_bo)
        new_params = minimize(
            EI_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        EI_star_x = new_params.x
        EI_star = -1 * EI_obj(EI_star_x) #actual problem is that of maximization, so earlier we multipled by -1
        
        # print("*********************************************************************")
        # optimize CD using constrained pso
        # const = lambda x: EIcalc_kd(y_train, x, gpr) - (alpha_lvl_set * (EI_star))
        # CD_obj = lambda x: -1 * CrowdingDist_kd(x, x_train)
        
        # opt_obj = lambda x: ei_cd(x, x_train, y_train, gpr, alpha_lvl_set, EI_star)
        # lb = inpRanges[:, 0]
        # ub = inpRanges[:, 1]
        # t = time.time()
        # x0, x_val = pso(CD_obj, lb, ub, f_ieqcons=const, maxiter=200)
        # t_original = time.time() - t

        # max_bound = inpRanges[:, 1]
        # min_bound = inpRanges[:, 0]
        # bounds = (min_bound, max_bound)
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        # t = time.time()
        # optimizer = GlobalBestPSO(n_particles=n_0, dimensions=tf_dim, options=options, bounds=bounds)

        # gpso_val, gpso_x = optimizer.optimize(opt_obj, iters=200)
        # t_gpso = time.time() - t
        
        # lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        # upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])
        # t = time.time()
        # random_samples = uniform_sampling(2000, inpRanges, tf_dim, rng)
        # min_bo_val = opt_obj(random_samples)
        
        # min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])
        # min_bo_val = np.min(min_bo_val)
        # # print(min_bo_val)
        # for _ in range(9):
        #     new_params = minimize(
        #         opt_obj,
        #         bounds=list(zip(lower_bound_theta, upper_bound_theta)),
        #         x0=min_bo,
        #     )

        #     # if not new_params.success:
        #     #     continue

        #     if min_bo is None or opt_obj(new_params.x) < min_bo_val:
        #         min_bo = new_params.x
        #         min_bo_val = opt_obj(min_bo)
            
        #     # print(min_bo_val)
        # new_params = minimize(
        #     opt_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        # )
        # local_x = new_params.x
        # local_val = opt_obj(min_bo)
        # t_local = time.time() - t

        # print("**************************************************************")
        # print(f"Values from original: {x_val}, {x0}")
        # print(f"Time for original = {t_original}")
        # print("**************************************************************")
        # print(f"Values from GPSO: {gpso_val}, {gpso_x}")
        # print(f"Time for GPSO = {t_gpso}")
        # print("**************************************************************")
        # print(f"Values from LOCAL: {local_val}, {local_x}")
        # print(f"Time for GPSO = {t_local}")
        # print("**************************************************************")
        
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
                
        new_params = minimize(
            opt_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_eicd
        )
        pred_sample_x = np.array([np.array(new_params.x)])
        
        pred_sample_y, falsified = compute_robustness(pred_sample_x, 1, behavior, tf_wrapper)
        if falsified:
            return (tf_wrapper.point_history, tf_wrapper.modes, tf_wrapper.simultation_time)
        
        x_train = np.vstack((x_train, pred_sample_x))
        y_train = np.hstack((y_train, pred_sample_y))
        
        print(pred_sample_x)
        print(inpRanges[:, 0])
        print(inpRanges[:, 1])
        ######### LOCAL SEARCH PHASE ###########
        # Initialize TR Bounds
        TR_Bounds = np.vstack(
            [pred_sample_x[0,:] - inpRanges[:, 0], inpRanges[:, 1] - pred_sample_x[0,:], (inpRanges[:, 1] - inpRanges[:, 0]) / 10])

        
        TR_size = np.min(TR_Bounds)
        TR = np.vstack([pred_sample_x[0,:] - TR_size, pred_sample_x[0,:] + TR_size])
        print(TR_Bounds)
        print(TR_size)
        print(TR)
        print(f"time elapsed = {time.time() - t}")
        print(fvs)
        all_local_x = x0
        all_local_y = f0
        local_in_TR_idx = np.all(np.vstack([np.all(all_local_x >= TR[0, :].reshape(-1, 1).T),
                                            np.all(all_local_x <= TR[1, :].reshape(-1, 1).T)]), axis=0)
        m = sum(local_in_TR_idx)
        xTrain_local = x0
        yTrain_local = f0

        ####### Enter TR Meta Model Loop ######
        local_counter = 0
        max_loc_iter = 10
        if local_search == "gp_local_search":
            while local_counter <= max_loc_iter \
                    and TR_size > eps_tr * min((inpRanges[0][:, 1] - inpRanges[0][:,0]).flatten()) \
                    and dimOK4budget \
                    and sim_count + n_0 - m <= nSamples \
                    and run['falsified'] != 1:
                start = sim_count
                # initialize local samples and values
                print('starting localGPs TR search...')
                if n_0 - m > 0:
                    # draw a new lhs over the current TR
                    x0_local = lhs_sampling(n_0 - m, inpRanges, nInputs, rng)[0]
                    for i in range(n_0 - m):
                        curSample_local = x0_local[i, :]
                        curVal_local = calculate_robustness(curSample_local, prob)
                        sim_count += 1
                        # store as necessary
                        history['cost'] = np.vstack((history['cost'], curVal_local))
                        history['rob'] = np.vstack((history['rob'], curVal_local))
                        history['samples'] = np.vstack((history['samples'], curSample_local))
                        # find and store the best value seen so far
                        minmax_val = minmax(history['rob'])
                        minmax_idx = np.where(history['rob'] == minmax_val)[0][0]
                        if fcn_cmp(minmax_val, bestCost):
                            bestCost = minmax_val
                            history['f_star'] = np.vstack([history['f_star'], minmax_val])
                            run['bestCost'] = minmax_val
                            run['bestSample'] = history['samples'][minmax_idx, :]
                            run['bestRob'] = minmax_val
                            print('Best ==>' + str(run['bestSample']) + str(minmax_val))
                        else:
                            history['f_star'] = np.vstack([history['f_star'], bestCost])
                            print('Best ==>' + str(bestCost))
                        if fcn_cmp(bestCost, 0):  # and StopCond:
                            run['falsified'] = 1
                            run['nTests'] = sim_count
                            print('SOAR_Taliro: FALSIFIED!')
                            print('FinGlobalMod_: ', seed, 'GPmod', 'yTrain')
                    # add newly drawn points to list of all local points
                    curVal_local = history['rob'][-n_0 + m:]
                    # add newly drawn points to list of all local points
                    all_local_x = np.vstack([all_local_x, x0_local])
                    all_local_y = np.vstack([all_local_y, curVal_local])
                    ############## store the local samples separately
                    history['local_samples'] = np.vstack([history['local_samples'], x0_local])
                    all_x = np.vstack([all_x, x0_local])
                    all_y = np.vstack([all_y, curVal_local])
                    xTrain_local = np.vstack([xTrain_local, x0_local])
                    yTrain_local = np.vstack([yTrain_local, curVal_local])
                # Fit Gaussian Process Meta Model Locally
                xk, fk, rho = local_gp_tr(x0, f0, n_0, nInputs, prob, TR, xTrain_local, yTrain_local)
                # store to current sample and value
                curSample = np.vstack((curSample, xk.reshape(-1, 1).T))
                curVal = np.vstack((curVal, fk.T))
                sim_count += 1
                # store as necessary
                history['cost'] = np.vstack((history['cost'], curVal[-1]))
                history['rob'] = np.vstack((history['rob'], curVal[-1]))
                history['samples'] = np.vstack((history['samples'], curSample[-1, :]))
                history['local_samples'] = np.vstack([history['local_samples'], curSample[-1, :]])
                # find and store the best value seen so far
                minmax_val = minmax(curVal)
                minmax_idx = np.where(curVal == minmax_val)[0][0]
                if fcn_cmp(minmax_val, bestCost):
                    bestCost = minmax_val
                    history['f_star'] = np.vstack([history['f_star'], minmax_val])
                    run['bestCost'] = minmax_val
                    run['bestRob'] = minmax_val
                    run['bestSample'] = curSample[minmax_idx, :]
                    print('Best ==>' + str(run['bestSample']) + str(minmax_val))
                else:
                    history['f_star'] = np.vstack([history['f_star'], bestCost])
                    # best_idx = np.where(curVal == bestCost)
                    print('Best ==>' + str(bestCost))
                # check if best value is falsifying, if so, exit as necessary
                if fcn_cmp(bestCost, 0):  # and StopCond:
                    run['falsified'] = 1
                    run['nTests'] = sim_count
                    print('SOAR_Taliro: FALSIFIED!')
                    print('FinGlobalMod_: ', seed, 'GPmod', 'yTrain')

                # store xk candidates to all samples and all local samples
                all_x = np.vstack([all_x, xk])
                all_y = np.vstack([all_y, fk])
                all_local_x = np.vstack([all_local_x, xk])
                all_local_y = np.vstack([all_local_y, fk])

                ############# temporary
                traj['restart'] = np.vstack([traj['restart'], restart])
                traj['xk'] = np.vstack([traj['xk'], xk])
                print('@@@@@@@@@@@@@@@@@@@@', local_counter, restart, xk)
                ############# temporary

                max_indicator = max(np.abs(xk - x0)) / TR_size
                test = np.random.rand()
                if max_indicator < test:
                    local_counter += 1
                    local_budget = sim_count - start
                    local_budget_used.append(local_budget)
                    history['TR_budget_used'] = np.array(local_budget_used)
                    # break
                local_counter += 1
                local_budget = sim_count - start
                local_budget_used.append(local_budget)
                history['TR_budget_used'] = np.array(local_budget_used)

                # execute RC testing and TR control
                if rho < eta0:
                    x0 = x0
                    TR_size *= delta
                    TR = np.vstack([x0 - TR_size, x0 + TR_size])
                else:
                    if eta0 < rho < eta1:
                        # low pass of RC test
                        x0 = xk
                        f0 = fk
                        valid_bound = np.hstack([x0 - inpRanges[0][:, 0], inpRanges[0][:, 1] - x0, TR_size])
                    else:
                        # high pass of RC test
                        x0 = xk
                        f0 = fk
                        valid_bound = np.hstack([x0 - inpRanges[0][:, 0], inpRanges[0][:, 1] - x0, TR_size * gamma])
                    TR_size = min(valid_bound.flatten())
                    TR = np.vstack([x0 - TR_size, x0 + TR_size])

                # check if budget has been exhausted
                if sim_count >= nSamples:
                    run['nTests'] = sim_count
                    print('SOAR_Taliro: Samples Exhausted!')
                    print('FinGlobalMod_: ', seed, 'GPmod', 'yTrain')
                    break

                # check old local points in new TR, build local training set
                local_in_TR_idx = np.all(np.vstack([np.all(all_local_x >= TR[0, :].reshape(-1, 1).T, axis=1),
                                                    np.all(all_local_x <= TR[1, :].reshape(-1, 1).T, axis=1)]), axis=0)
                m = sum(local_in_TR_idx)
                xTrain_local = all_local_x[local_in_TR_idx, :]
                yTrain_local = all_local_y[local_in_TR_idx, 0].reshape(-1, 1)

        # add to global samples separately
        if x0 in history['global_samples']:
            print('no new point')
        else:
            history['global_samples'] = np.vstack([history['global_samples'], x0])
        history['local_samples'] = np.delete(history['local_samples'], -1, 0)

        # add EI point to the global set and local set
        xTrain = np.vstack([xTrain, x0])
        yTrain = np.vstack([yTrain, f0])

        print(sim_count)

    print(history['samples'].shape, history['rob'].shape, history['f_star'].shape, history['global_samples'].shape,
          history['local_samples'].shape)
    print('SOAR_Taliro: Samples Exhausted!')
    run['nTests'] = nSamples

    base_path = pathlib.Path()
    results_directory = base_path.joinpath(folder_name)
    results_directory.mkdir(exist_ok=True)

    benchmark_directory = results_directory.joinpath(benchmark_name)
    benchmark_directory.mkdir(exist_ok=True)

    with open(benchmark_directory.joinpath(f"{benchmark_name}_seed_{seed}.pkl"), "wb") as f:
        pickle.dump(history, f)

    return run, history, traj