from ..kriging_gpr.interface import OK_Rmodel_kd_nugget
from ..trustRegion import local_gp_tr, gradient_based_tr
from ..utils import calculate_robustness, callCounter, EIcalc_kd, neg_EIcalc_kd, CrowdingDist_kd, neg_CrowdingDist_kd
from ..sampling import lhs_sampling

from pyswarms.single import LocalBestPSO, GlobalBestPSO
from pyswarm import pso
import numpy as np
import math

##### v8 ####### combining two methods
def PySOAR(inpRanges, test_function, repNum, local_search):
    seed = 1000 + repNum
    print(f"Running PySOAR with {local_search} local search - Replication {repNum} with seed {seed}.")
    prob = callCounter(test_function)
    rng = np.random.default_rng(seed)
    np.random.seed(repNum)
    nSamples = 100
    # initialize result dictionary
    run = dict()
    history = dict()
    # get polarity and set the fcn_cmp
    fcn_cmp = lambda x, y: True if x <= y else False
    minmax = lambda x: min(x)

    # start SOAR here
    nInputs = inpRanges.shape[1]
    n_0 = 10 * nInputs
    dimOK4budget = True
    if n_0 >= nSamples:
        n_0 = max(nSamples - 10, math.ceil(nSamples / 2))
        dimOK4budget = False
    # Crowded EI level set threshold
    alpha_lvl_set = .05
    # parameters for the TR algorithm, user defined
    # for RC test and TR control
    eta0 = .25
    eta1 = .75
    delta = .75
    gamma = 1.25
    eps_tr = 0.01

    # Instantiate and scale initial design
    sim_count = 0
    x_0 = lhs_sampling(n_0, inpRanges, nInputs, rng)[0]
    # take first samples, check falsification
    curSample = x_0[0, :]
    curVal = calculate_robustness(curSample, prob)
    sim_count += 1
    history['cost'] = curVal
    history['rob'] = curVal
    history['samples'] = curSample
    ####### store global samples separately
    history['global_samples'] = curSample
    minmax_val = minmax(curVal)
    history['f_star'] = minmax_val
    bestCost = minmax_val
    run['bestCost'] = minmax_val
    run['bestSample'] = curSample
    run['bestRob'] = minmax_val
    run['falsified'] = minmax_val <= 0
    run['nTests'] = sim_count
    # check if best value is falsifying, if so, exit as necessary
    if fcn_cmp(minmax_val, 0):  # and StopCond:
        print('SOAR_Taliro: FALSIFIED by initializing samples!')
    for i in range(1, n_0):
        curSample = x_0[i, :]
        curVal = calculate_robustness(curSample, prob)
        sim_count += 1
        # store as necessary
        history['cost'] = np.vstack([history['cost'], curVal])
        history['rob'] = np.vstack([history['rob'], curVal])
        history['samples'] = np.vstack([history['samples'], curSample])
        ####### store global samples separately
        history['global_samples'] = np.vstack([history['global_samples'], curSample])
        # find and store the best value seen so far
        minmax_val = minmax(history['rob'])
        history['f_star'] = np.vstack([history['f_star'], minmax_val])
        minmax_idx = np.where(history['rob'] == minmax_val)[0][0]
        bestCost = minmax_val
        run['bestCost'] = minmax_val
        run['bestSample'] = history['samples'][minmax_idx, :]
        run['bestRob'] = minmax_val
        run['falsified'] = minmax_val <= 0
        run['nTests'] = sim_count
        # check if best value is falsifying, if so, exit as necessary
        if fcn_cmp(minmax_val, 0):  # and StopCond:
            print('SOAR_Taliro: FALSIFIED by initializing samples!')

    # set up for surrogate modeling
    curSample = history['samples']
    curVal = history['rob']
    xTrain = np.array(curSample)
    yTrain = np.array(curVal)
    all_x = xTrain
    all_y = yTrain

    # initialize local samples
    history['local_samples'] = np.empty((0, nInputs), dtype=float)

    # store restart locations and local samples into a trajectory dict
    traj = dict()
    traj['restart'] = np.empty((0, nInputs), dtype=float)
    traj['xk'] = np.empty((0, nInputs), dtype=float)

    local_budget_used = []
    # starting simulation loop
    while sim_count < nSamples:
        # Fit Gaussian Process Meta Model
        GPmod = OK_Rmodel_kd_nugget(xTrain, yTrain, 0, 2, 10)
        # optimize EI using pso
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        max_bound = inpRanges[0][:, 1] * np.ones(nInputs)
        min_bound = inpRanges[0][:, 0] * np.ones(nInputs)
        bounds = (min_bound, max_bound)
        optimizer = GlobalBestPSO(n_particles=n_0, dimensions=nInputs, options=options, bounds=bounds)
        EI_obj = lambda x: neg_EIcalc_kd(x, all_x, GPmod, all_y)
        EI_star, x_EI = optimizer.optimize(EI_obj, iters=200)
        # optimize CD using constrained pso
        const = lambda x: EIcalc_kd(x.reshape(-1, 1), all_x, GPmod, all_y) - alpha_lvl_set * (-EI_star)
        CD_obj = lambda x: neg_CrowdingDist_kd(np.transpose(x.reshape(-1, 1)), all_x)
        lb = inpRanges[0][:, 0]
        ub = inpRanges[0][:, 1]
        x0, _ = pso(CD_obj, lb, ub, f_ieqcons=const, maxiter=200)
        # store acquisition function sample appropriately and sample it
        curSample = np.vstack((curSample, x0.reshape(-1, 1).T))
        f0 = calculate_robustness(x0, prob)
        curVal = np.vstack((curVal, f0.T))
        sim_count += 1
        # store as necessary
        history['cost'] = np.vstack((history['cost'], curVal[-1]))
        history['rob'] = np.vstack((history['rob'], curVal[-1]))
        history['samples'] = np.vstack((history['samples'], curSample[-1, :]))
        # add restart location x0 to LOCAL samples
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
            print('Best ==>' + str(bestCost))
        # check if best value is falsifying, if so, exit as necessary
        if fcn_cmp(bestCost, 0):  # and StopCond:
            run['falsified'] = 1
            run['nTests'] = sim_count
            print('SOAR_Taliro: FALSIFIED!')
            print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
        # check if budget has been exhausted
        if sim_count >= nSamples:
            run['nTests'] = sim_count
            print('SOAR_Taliro: Samples Exhausted!')
            print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
            break
        all_x = np.vstack([all_x, x0])
        all_y = np.vstack([all_y, f0])
        xTrain = np.vstack([xTrain, x0])
        yTrain = np.vstack([yTrain, f0])

        ############## store x0 for each global iteration for trajectory demo purposes
        restart = x0
        ##############

        ######### LOCAL SEARCH PHASE ###########
        # Initialize TR Bounds
        TR_Bounds = np.vstack([x0 - inpRanges[0][:, 0], inpRanges[0][:, 1] - x0, (inpRanges[0][:, 1] - inpRanges[0][:, 0]) / 10])
        TR_size = min(TR_Bounds.flatten())
        TR = np.vstack([x0 - TR_size, x0 + TR_size])
        n_0 = 5 * nInputs

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

        if local_search == 'gradient':
            while local_counter <= max_loc_iter and TR_size > eps_tr * min((inpRanges[0][:, 1] - inpRanges[0][:, 0]).flatten()) and dimOK4budget:
                start = sim_count
                print('starting gradient local search...')
                xk, fk, rho = gradient_based_tr(x0, f0, TR_size, nInputs, prob)
                # store acquisition function sample appropriately and sample it
                curSample = np.vstack((curSample, xk.reshape(-1, 1).T))
                fk = calculate_robustness(xk, prob)
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
                    history['bestRob'] = np.vstack([history['bestRob'], bestCost])
                    print('Best ==>' + str(bestCost))
                # check if best value is falsifying, if so, exit as necessary
                if fcn_cmp(bestCost, 0):  # and StopCond:
                    run['falsified'] = 1
                    run['nTests'] = sim_count
                    print('SOAR_Taliro: FALSIFIED!')
                    print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')

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
                    print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
                    break

        else:
            while local_counter <= max_loc_iter and TR_size > eps_tr * min((inpRanges[0][:, 1] - inpRanges[0][:, 0]).flatten()) and dimOK4budget and sim_count + n_0 - m <= nSamples:
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
                            print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
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
                # ###### add to global samples separately
                # history['global_samples'] = np.vstack([history['global_samples'], curSample[-1, :]])
                # find and store the best value seen so far
                minmax_val = minmax(curVal)
                # print(sim_count)
                # history['budget_used'] = np.vstack([history['budget_used'], np.array([sim_count])])
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
                    print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')

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
                    print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
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
    print(traj)
    return run, history

















