import numpy as np
import math

from EIcalc_kd import EIcalc_kd, neg_EIcalc_kd
from CrowdingDist_kd import CrowdingDist_kd, neg_CrowdingDist_kd
from local_search import local_gp_tr, gradient_based_tr
from test_functions import Himmelblau_2d
from pysoar.coreAlgorithm.testFunction import callCounter
from kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from pysoar.coreAlgorithm.sampling import lhs_sampling
from kriging_gpr.examples.calculate_robustness import calculate_robustness
from kriging_gpr.utils.normalize_data import normalize_data

from pyswarms.single import LocalBestPSO, GlobalBestPSO
from pyswarm import pso


##### v3 ####### fixed sim_count for budget and added local search info storing
def PySOAR(inpRanges, prob, repNum, local_search):
    rng = np.random.default_rng(1000 + repNum)
    nSamples = 200
    # StopCond = False
    run = dict()
    history = dict()
    # get polarity and set the fcn_cmp
    fcn_cmp = lambda x, y: True if x <= y else False
    minmax = lambda x: min(x)
    # start SOAR here
    n_0 = 10 * nInputs
    dimOK4budget = True
    if n_0 >= nSamples:
        n_0 = max(nSamples - 10, math.ceil(nSamples/2))
        dimOK4budget = False

    curSample = np.zeros((n_0, nInputs))
    curVal = np.zeros((n_0, 1))
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
    for i in range(n_0):
        curSample[i, :] = x_0[i, :]
        curVal[i] = calculate_robustness(curSample[i, :], prob)
        sim_count += 1
        # instantiate storage/history if first sample
        if i == 0:
            history['cost'] = np.zeros((n_0, 1))
            history['rob'] = np.zeros((n_0, 1))
            history['samples'] = np.zeros((n_0, nInputs))
        # store as necessary
        history['cost'][i] = curVal[i]
        history['rob'][i] = curVal[i]
        history['samples'][i, :] = curSample[i, :]
        # find and store the best value seen so far
    minmax_val = minmax(curVal)
    history['bestRob'] = minmax_val
    history['budget_used'] = np.array([sim_count])
    minmax_idx = np.where(curVal == minmax_val)[0][0]
    bestCost = minmax_val
    run['bestCost'] = minmax_val
    run['bestSample'] = curSample[minmax_idx, :]
    run['bestRob'] = minmax_val
    run['falsified'] = minmax_val <= 0
    run['nTests'] = sim_count
    # check if best value is falsifying, if so, exit as necessary
    if fcn_cmp(minmax_val, 0):# and StopCond:
        print('SOAR_Taliro: FALSIFIED by initializing samples!')

    # set up for surrogate modeling
    xTrain = np.array(curSample)
    yTrain = np.array(curVal)
    all_x = xTrain
    all_y = yTrain

    local_counter = 0
    local_budget_used = []
    while sim_count < nSamples: # sim_count = evaluations that were already exhausted
        # Fit Gaussian Process Meta Model
        GPmod = OK_Rmodel_kd_nugget(xTrain, yTrain, 0, 2, 10)
        # optimize EI using pso
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
        optimizer = GlobalBestPSO(n_particles=n_0, dimensions=nInputs, options=options)
        EI_obj = lambda x: neg_EIcalc_kd(x, all_x, GPmod, all_y)
        EI_star, x_EI = optimizer.optimize(EI_obj, iters=200)
        x_EI = x_EI.reshape(-1, 1).T
        # optimize CD using constrained pso
        const = lambda x: EIcalc_kd(x.reshape(-1, 1), all_x, GPmod, all_y) - alpha_lvl_set * (-EI_star)
        CD_obj = lambda x: neg_CrowdingDist_kd(np.transpose(x.reshape(-1, 1)), all_x)
        lb = inpRanges[0][:, 0]
        ub = inpRanges[0][:, 1]
        x0, _ = pso(CD_obj, lb, ub, f_ieqcons=const, maxiter=200)
        # store acquisition function sample appropriately and sample it
        curSample = np.vstack((curSample, x0.reshape(-1, 1).T))
        f0 = np.transpose(calculate_robustness(x0, prob))
        curVal = np.vstack((curVal, f0.T))
        sim_count += 1
        # store as necessary
        history['cost'] = np.vstack((history['cost'], curVal[-1]))
        history['rob'] = np.vstack((history['rob'], curVal[-1]))
        history['samples'] = np.vstack((history['samples'], curSample[-1, :]))
        # find and store the best value seen so far
        minmax_val = minmax(curVal)
        print(sim_count)
        history['budget_used'] = np.vstack([history['budget_used'], np.array([sim_count])])
        minmax_idx = np.where(curVal == minmax_val)[0][0]
        if fcn_cmp(minmax_val, bestCost):
            bestCost = minmax_val
            history['bestRob'] = np.vstack([history['bestRob'], minmax_val])
            run['bestCost'] = minmax_val
            run['bestRob'] = minmax_val
            run['bestSample'] = curSample[minmax_idx, :]
            print('Best ==>' + str(run['bestSample']) + str(minmax_val))
        else:
            history['bestRob'] = np.vstack([history['bestRob'], bestCost])
            print('Best ==>' + str(bestCost))
        # check if best value is falsifying, if so, exit as necessary
        if fcn_cmp(bestCost, 0): # and StopCond:
            run['falsified'] = 1
            run['nTests'] = sim_count
            print('SOAR_Taliro: FALSIFIED!')
            print('FinGlobalMod_: ', repNum, 'GPmod','yTrain')
        # check if budget has been exhausted
        if sim_count >= nSamples :
            run['nTests'] = sim_count
            print('SOAR_Taliro: Samples Exhausted!')
            print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
        all_x = np.vstack([all_x, x0])
        all_y = np.vstack([all_y, f0])
        xTrain = np.vstack([xTrain, x0])
        yTrain = np.vstack([yTrain, f0])

        ######### LOCAL SEARCH PHASE ###########
        # Initialize TR Bounds
        TR_Bounds = np.vstack([x0 - inpRanges[0][:, 0], inpRanges[0][:, 1] - x0, (inpRanges[0][:, 1] - inpRanges[0][:, 0]) / 10])
        TR_size = min(TR_Bounds.flatten())
        TR = np.vstack([x0 - TR_size, x0 + TR_size])
        n_0 = 5 * nInputs
        m = 1
        all_local_x = x0
        all_local_y = f0
        xTrain_local = np.zeros((n_0 + 1, nInputs))
        yTrain_local = np.zeros((n_0 + 1, 1))
        xTrain_local[0, :] = x0
        yTrain_local[0, 0] = f0
        ####### Enter TR Meta Model Loop ######
        while TR_size > eps_tr * min((inpRanges[0][:, 1] - inpRanges[0][:,0]).flatten()) and sim_count+n_0-m < nSamples and dimOK4budget:
            start = sim_count
            if local_search == 'gradient':
                print('starting gradient local search...')
                xk, fk, rho = gradient_based_tr(x0, TR_size, nInputs, prob)
                sim_count += 4
                # store acquisition function sample appropriately and sample it
                curSample = np.vstack((curSample, xk.reshape(-1, 1).T))
                fk = np.transpose(calculate_robustness(xk, prob))
                curVal = np.vstack((curVal, f0.T))
                sim_count += 1
                # store as necessary
                history['cost'] = np.vstack((history['cost'], curVal[-1]))
                history['rob'] = np.vstack((history['rob'], curVal[-1]))
                history['samples'] = np.vstack((history['samples'], curSample[-1, :]))
                # find and store the best value seen so far
                minmax_val = minmax(curVal)
                print(sim_count)
                history['budget_used'] = np.vstack([history['budget_used'], np.array([sim_count])])
                minmax_idx = np.where(curVal == minmax_val)[0][0]
                if fcn_cmp(minmax_val, bestCost):
                    bestCost = minmax_val
                    history['bestRob'] = np.vstack([history['bestRob'], minmax_val])
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
                # check if budget has been exhausted
                if sim_count >= nSamples:
                    run['nTests'] = sim_count
                    print('SOAR_Taliro: Samples Exhausted!')
                    print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
            else:
                # initialize local samples and values
                print('starting localGPs TR search...')
                curSample_local = np.zeros((n_0 - m, nInputs))
                curVal_local = np.zeros((n_0 - m, 1))
                if n_0 - m > 0:
                    # draw a new lhs over the current TR
                    x0_local = lhs_sampling(n_0 - m, inpRanges, nInputs, rng)[0]
                    for i in range(n_0 - m):
                        curSample_local[i, :] = x0_local[i, :]
                        curVal_local[i] = calculate_robustness(curSample_local[i, :], prob)
                        sim_count += 1
                    # store as necessary
                    history['cost'] = np.vstack((history['cost'], curVal_local))
                    history['rob'] = np.vstack((history['rob'], curVal_local))
                    history['samples'] = np.vstack((history['samples'], curSample_local))
                    # find and store the best value seen so far
                    minmax_val = minmax(curVal_local)
                    print(sim_count)
                    history['budget_used'] = np.vstack([history['budget_used'], np.array([sim_count])])
                    minmax_idx = np.where(curVal_local == minmax_val)[0][0]
                    if fcn_cmp(minmax_val, bestCost):
                        bestCost = minmax_val
                        history['bestRob'] = np.vstack([history['bestRob'], minmax_val])
                        run['bestCost'] = minmax_val
                        run['bestSample'] = curSample_local[minmax_idx, :]
                        run['bestRob'] = minmax_val
                        print('Best ==>' + str(run['bestSample']) + str(minmax_val))
                    else:
                        history['bestRob'] = np.vstack([history['bestRob'], bestCost])
                        print('Best ==>' + str(bestCost))
                    if fcn_cmp(bestCost, 0): # and StopCond:
                        run['falsified'] = 1
                        run['nTests'] = sim_count
                        print('SOAR_Taliro: FALSIFIED!')
                        print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
                    # add newly drawn points to list of all local points
                    all_local_x = np.vstack([all_local_x, x0_local])
                    all_local_y = np.vstack([all_local_y, curVal_local])
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
                # find and store the best value seen so far
                minmax_val = minmax(curVal)
                print(sim_count)
                history['budget_used'] = np.vstack([history['budget_used'], np.array([sim_count])])
                minmax_idx = np.where(curVal == minmax_val)[0][0]
                if fcn_cmp(minmax_val, bestCost):
                    bestCost = minmax_val
                    history['bestRob'] = np.vstack([history['bestRob'], minmax_val])
                    run['bestCost'] = minmax_val
                    run['bestRob'] = minmax_val
                    run['bestSample'] = curSample[minmax_idx, :]
                    print('Best ==>' + str(run['bestSample']) + str(minmax_val))
                else:
                    history['bestRob'] = np.vstack([history['bestRob'], bestCost])
                    best_idx = np.where(curVal == bestCost)
                    print('Best ==>' + str(bestCost))
                # check if best value is falsifying, if so, exit as necessary
                if fcn_cmp(bestCost, 0): #and StopCond:
                    run['falsified'] = 1
                    run['nTests'] = sim_count
                    print('SOAR_Taliro: FALSIFIED!')
                    print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')
            all_x = np.vstack([all_x, xk])
            all_y = np.vstack([all_y, fk])
            # add EI point to the global set and local set
            xTrain = np.vstack([xTrain, xk])
            yTrain = np.vstack([yTrain, fk])
            all_local_x = np.vstack([all_local_x, xk])
            all_local_y = np.vstack([all_local_y, fk])

            max_indicator = max(np.abs(xk - x0))/TR_size
            test = np.random.rand()
            if max_indicator < test:
                local_counter += 1
                local_budget = sim_count - start
                print('***********', local_counter, local_budget)
                local_budget_used.append(local_budget)
                print('????????', local_budget_used)
                history['TR_budget_used'] = np.array(local_budget_used)
                break
            local_counter += 1
            local_budget = sim_count - start
            print('***********', local_counter, local_budget)
            local_budget_used.append(local_budget)
            print('????????', local_budget_used)
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
                    valid_bound = np.hstack([x0 - inpRanges[0][:, 0], inpRanges[0][:, 1] - x0, TR_size])
                else:
                    # high pass of RC test
                    x0 = xk
                    valid_bound = np.hstack([x0 - inpRanges[0][:, 0], inpRanges[0][:, 1] - x0, TR_size * gamma])
                TR_size = min(valid_bound.flatten())
                TR = np.vstack([x0 - TR_size, x0 + TR_size])
            # check old local points in new TR, build local training set
            local_in_TR_idx = np.all(np.vstack([np.all(all_local_x >= TR.T[:,0], axis=1), np.all(all_local_x <= TR.T[:,1] , axis=1)]), axis=0)
            m = sum(local_in_TR_idx)
            xTrain_local[:m, :] = all_local_x[local_in_TR_idx, :]
            yTrain_local[:m, :] = all_local_y[local_in_TR_idx, 0].reshape(-1, 1)
        print(sim_count)

    print('SOAR_Taliro: Samples Exhausted!')
    run['nTests'] = nSamples
    return run, history
