import numpy as np
import math

from EIcalc_kd import EIcalc_kd, neg_EIcalc_kd
from CrowdingDist_kd import CrowdingDist_kd, neg_CrowdingDist_kd
from local_search import local_gp_tr, gradient_based_tr
from test_functions import Himmelblau_2d
from .testFunction import callCounter
from kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from pysoar.coreAlgorithm.sampling import lhs_sampling
from kriging_gpr.examples.calculate_robustness import calculate_robustness
from kriging_gpr.utils.normalize_data import normalize_data

from pyswarms.single import LocalBestPSO, GlobalBestPSO
from pyswarm import pso



def PySOAR(inpRanges, opt, repNum):
    params = opt.optim_params
    nSamples = params.n_tests
    StopCond = opt.falsification

    nInputs = inpRanges.shape[0]
    run = dict()
    history = dict()

    curSample = np.tile({0}, (1, opt.n_workers))
    curVal = np.tile({0}, 1, opt.n_workers)

    # get polarity and set the fcn_cmp
    if opt.parameterEstimation == 1:
        if opt.optimization == 'min':
            fcn_cmp = lambda x, y: True if x <= y else False
            minmax = lambda x: np.min(x)
        elif opt.optimization == 'max':
            fcn_cmp = lambda x, y: True if x >= y else False
            minmax = lambda x: np.max(x)
    else:
        fcn_cmp = lambda x, y: True if x <= y else False
        minmax = lambda x: np.min(x)

    # start SOAR here
    crowded_EI_flag = params.crowded_EI_flag
    n_0 = 10 * nInputs
    dimOK4budget = True
    if n_0 >= nSamples:
        n_0 = np.max(nSamples - 10, math.ceil(nSamples/2))
        dimOK4budget = False
    B = 1
    B_n0_setting = 1

    # Crowded EI level set threshold
    alpha_lvl_set = params.crowding_threshold

    # parameters for the TR algorithm, user defined
    # for RC test and TR control
    eta0 = params.TR_lowpass_thresh
    eta1 = params.TR_highpass_thresh
    delta = params.TR_delta
    gamma = params.TR_gamma

    # Instantiate and scale initial design
    sim_count = 0
    x_0 = lhs_sampling(n_0, inpRanges, nInputs, rng)
    x_0 = x_0 * (inpRanges[:, 1] - inpRanges[:, 0]) + inpRanges[:, 0]

    # take first samples, check falsification
    for i in range(n_0):
        curSample[i] = x_0[i, :]
        curVal[i] = calculate_robustness(curSample[i], problem.fun)
        sim_count = sim_count + 1
        # instantiate storage/history if first sample
        if len(curVal[i]) > 1 and i == 0:
            if type(curVal[0]) == 'hydis':
                history['cost'] = hydis(np.zeros((nSamples, 1)))
                history['rob'] = hydis(np.zeros((nSamples, 1)))
            else:
                history['cost'] = np.zeros((nSamples, 1))
                history['rob'] = np.zeros((nSamples, 1))
            history['samples'] = np.zeros((nSamples, nInputs))
        # store as necessary
        if len(curVal[i]) > 1:
            if type(curVal[i]) == 'hydis':
                history['cost'][i] = hydisc2m(curVal[i])
                history['rob'][i] = hydisc2m(curVal[i])
            else:
                history['cost'][i] = curVal[i]
                history['rob'][i] = curVal[i]
            history['samples'][i, :] = curSample[i]
        # find and store the best value seen so far
        if type(curVal[0]) == 'hydis':
            minmax_val, minmax_idx = minmax(hydisc2m(curVal))
        else:
            minmax_val, minmax_idx = minmax(np.array(curVal))
        bestCost = minmax_val
        run['bestCost'] = minmax_val
        run['bestSample'] = curSample[minmax_idx]
        run['bestRob'] = minmax_val
        run['falsified'] = minmax_val <= 0
        run['nTests'] = sim_count

        # check if best value is falsifying, if so, exit as necessary
        if fcn_cmp(minmax_val, 0) and StopCond:
            if len(fcn_cmp(minmax_val, 0)) > 1:
                if type(minmax_val) == 'hydis':
                    history['cost'][i+1: ] = hydis([], [])
                    history['rob'][i+1:] = hydis([], [])
                else:
                    history['cost'][i+1:] = []
                    history['rob'][i+1:] = []
                history['samples'][i+1:, :] = []
            print('SOAR_Taliro: FALSIFIED by initializing samples!')

    # set up for surrogate modeling
    xTrain = np.array(curSample)
    yTrain = np.array(curVal)
    all_x = xTrain
    all_y = yTrain

    # # Initialize nspso Parameters
    # gen = 500 # Maximum number of generations
    # minW = 0.4
    # maxW = 1.0
    # C1 = 2.0
    # C2 = 2.0
    # CHI = 1.0
    # leader_selection = 5  # Maxmium vel in percentage
    # v_coeff = 0.5 # Uniform mutation percentage
    # diversity_mechanism = 'crowding_distance'

    while sim_count < nSamples:
        # Fit Gaussian Process Meta Model
        GPmod = OK_Rmodel_kd_nugget(xTrain, yTrain, 0, 2, 10)
        # # optimize EI and CD with PyGMO.algorithm.nspso
        # MultiObj_fun = lambda x: [-EIcalc_kd(x, xTrain, GPmod, yTrain), -CrowdingDist_kd(x, all_x)]
        # alg = algorithm.nspso(gen, minW, maxW, C1, C2, CHI, v_coeff, leader_selection, diversity_mechanism)
        # pop = population(MultiObj_fun, 100)
        # pop = alg.evolve(pop)
        # F = np.array([ind.cur_f for ind in pop]).T
        # minNegEI, index = np.min(F[0][:, 0])
        EI_values = EIcalc_kd(x_0, xTrain, GPmod, yTrain)
        EI_range = max(EI_values) - min(EI_values)
        relative_top_percent_threshold = EI_range * alpha_lvl_set

        # Selection of next centroid based upon maxEI
        maxEI =  max(EI_values)
        all_maxEI_pointer = 0
        all_maxEI_locations = np.zeros((EI_values.shape[0], nInputs))
        maxEI_value = np.zeros(EI_values.shape[0])
        crowding_distance = np.zeros(all_maxEI_locations.shape[0])
        DimWise_Crowd = np.zeros(nInputs)
        # Find all locations in top EI alpha level seT
        for i in range(EI_values.shape[0]):
            if EI_values[i] >= maxEI - relative_top_percent_threshold:
                all_maxEI_locations[all_maxEI_pointer, :] = xTrain[i, :]
                maxEI_value[all_maxEI_pointer, 0] = EI_values[i]
                all_maxEI_pointer += 1
        # Determine crowding distance at all locations in alpha level set
        for i in range(all_maxEI_locations.shape[0]):
            for j in range(nInputs):
                DimWise_Crowd[j] = min(np.abs((np.ones(all_x.shape[0], 1) * all_maxEI_locations[i, j]) - all_x[:, j]))
            crowding_distance[i] = sum(DimWise_Crowd)
        max_crowd = max(crowding_distance)
        Index = np.argmax(crowding_distance)
        x0 = all_maxEI_locations[Index, :]

        # store acquisition function sample appropriately and sample it
        curSample[0] = x0.T
        curVal = calculate_robustness(curSample, problem.fun)
        f0 = curVal[0]
        sim_count += 1

        # store as necessary
        if len() > 1:
            if type(curVal[0]) == 'hydis':
                history['cost'][sim_count] = hydisc2m(curVal)
                history['rob'][sim_count] = hydisc2m(curVal)
            else:
                history['cost'][sim_count] = curVal[0]
                history['rob'][sim_count] = curVal[0]
            history['samples'][sim_count, :] = curSample[0]
        # find and store the best value seen so far
        if type(curVal[0]) == 'hydis':
            minmax_val, minmax_idx = minmax(hydisc2m(curVal))
        else:
            minmax_val, minmax_idx = minmax(np.array(curVal))
        if fcn_cmp(minmax_val, bestCost):
            bestCost = minmax_val
            run['bestCost'] = minmax_val
            run['bestRob'] = minmax_val
            run['bestSample'] = curSample[minmax_idx]
            if opt.dispinfo > 0:
                if type(minmax_val) == 'hydis':
                    print('Best ==> <' + str(minmax_val[0]) + ',' + str(minmax_val[1]) + '>')  # ??????
                else:
                    print('Best ==>' + str(minmax_val))
        # check if best value is falsifying, if so, exit as necessary
        if fcn_cmp(bestCost, 0) and StopCond:
            run['falsified'] = 1
            run['nTests'] = sim_count
            if len() > 1:
                if type(minmax_val) == 'hydis':
                    history['cost'][sim_count+1:] = hydis([], [])
                    history['rob'][sim_count+1:] = hydis([], [])
                else:
                    history['cost'][sim_count + 1:] = []
                    history['rob'][sim_count + 1:] = []
                history['samples'][sim_count+1:, :] = []
            print('SOAR_Taliro: FALSIFIED!')
            print('FinGlobalMod_: ', repNum, 'GPmod','yTrain')

        # check if budget has been exhausted
        if sim_count >= nSamples:
            run['nTests'] = sim_count
            print('SOAR_Taliro: Samples Exhausted!')
            print('FinGlobalMod_: ', repNum, 'GPmod', 'yTrain')

        all_x = np.vstack([all_x, x0])
        all_y = np.vstack([all_y, curVal[0]])
        xTrain = np.vstack([xTrain, x0])
        yTrain = np.vstack([yTrain, curVal[0]])

        ######### LOCAL SEARCH PHASE ###########
        # Initialize TR Bounds
        TR_Bounds = np.vstack([x0 - inpRanges[:, 0], inpRanges[:, 1] - x0, (inpRanges[:, 1] - inpRanges[:, 0]) / 10])
        TR_size = min(TR_Bounds)
        TR = np.hstack([x0 - TR_size, x0 + TR_size])
        n_0 = 5 * nInputs
        m = 1
        all_local_x = x0
        all_local_y = f0
        xTrain_local = np.zeros((n_0 + 1, nInputs))
        yTrain_local = np.zeros((n_0 + 1, 1))
        xTrain_local[0, :] = x0
        yTrain_local[0, 0] = f0

        ####### Enter TR Meta Model Loop ######
        while TR_size > 0.01*min((inpRanges[:, 2]-inpRanges[:,1])) and sim_count+n_0-m < nSamples and dimOK4budget:
            xk, fk, rho = local_gp_tr(x0, xTrain_local, yTrain_local) # or polynomial model
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
                break
            # execute RC testing and TR control
            if rho < eta0:
                x0 = x0
                TR_size *= delta
                TR = np.hstack[x0.T - TR_size, x0.T + TR_size]
            else:
                if eta0 < rho and rho < eta1:
                    # low pass of RC test
                    x0 = xk
                    valid_bound = np.hstack([x0.T - inpRanges[:,1], inpRanges[:, 2] - x0.T, TR_size])
                else:
                    # high pass of RC test
                    x0 = xk
                    valid_bound = np.hstack([x0.T - inpRanges[:, 1], inpRanges[:, 2] - x0.T, TR_size * gamma])
                TR_size = np.min(valid_bound)
                f0 = fk
                TR = np.hstack[x0.T - TR_size, x0.T + TR_size]

            # check old local points in new TR, build local training set
            local_in_TR_idx = np.all(np.hstack([np.all(all_local_x >= TR[:,1].T, axis=0), np.all(all_local_x <= TR[:,2].T , axis=0)]), axis=0)
            m = np.sum(local_in_TR_idx, axis=1)

            xTrain_local[:m, :] = all_local_x[local_in_TR_idx, :]
            yTrain_local[:m, :] = all_local_y[local_in_TR_idx, 0]

        print(sim_count)

    print('SOAR_Taliro: Samples Exhausted!')
    run['nTests'] = nSamples
    return run, history
