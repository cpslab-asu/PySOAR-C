from ..kriging_gpr.interface import OK_Rpredict, OK_Rmodel_kd_nugget
from ..utils import compute_robustness, quadratic_model, EIcalc_kd

import numpy as np
import math
from scipy.optimize import minimize, Bounds, LinearConstraint
from numdifftools import Jacobian, Hessian
from pyswarms.single import LocalBestPSO

def gradient_based_tr(x0, f0, TR_size, nInputs, prob):
    a = np.zeros((2*nInputs, nInputs))
    for i in range(2*nInputs):
        if i <= nInputs:
            for j in range(nInputs):
                if i == j:
                    a[i, j] = 1
                else:
                    a[i, j] = 0
        else:
            for j in range(nInputs):
                if i - nInputs == j:
                    a[i, j] = -1
                else:
                    a[i, j] = 0
    # ub = (TR_size/2) * np.ones((2*nInputs, ))
    # lb = 0 * np.ones((2*nInputs, ))
    jac = Jacobian(prob)
    jac_x0 = jac(x0.flatten())
    hes = Hessian(prob)
    hes_x0 = hes(x0.flatten())
    fun1 = lambda s: quadratic_model(s, f0, jac_x0.T, hes_x0)
    b = (TR_size / 2) * np.ones((2 * nInputs,))
    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: np.array([b[0] - x[0],
                                            b[1] - x[1],
                                            b[2],
                                            b[3] + x[1]]),
                 'jac': lambda x: np.array([[-1.0, 0],
                                            [0, -1.0],
                                            [0, 0],
                                            [0, 1.0]])}
    min_bds = (-TR_size / 2) * np.ones((nInputs,))
    max_bds = (TR_size / 2) * np.ones((nInputs,))
    bounds = Bounds(min_bds, max_bds)
    res = minimize(fun1, x0.flatten(), method='SLSQP', constraints=ineq_cons, bounds=bounds)
    sk = res['x']
    xk = x0 + sk
    fk = calculate_robustness(xk, prob)
    rho = (f0 - fk) / (quadratic_model(np.zeros((1, nInputs)), f0, jac_x0.T, hes_x0) -
                       quadratic_model(sk, f0, jac_x0.T, hes_x0))
    return xk, fk, rho


def local_gp_tr(x0, f0, n_0, nInputs, prob, TR, xTrain_local, yTrain_local):
    # Fit Gaussian Process Meta Model Locally
    GPmod_local = OK_Rmodel_kd_nugget(xTrain_local, yTrain_local, 0, 2, 10)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    min_bound = np.ones(nInputs) * TR[:, 0]
    max_bound = np.ones(nInputs) * TR[:, 1]
    bounds = (min_bound, max_bound)
    optimizer = LocalBestPSO(n_particles=n_0, dimensions=nInputs, options=options, bounds=bounds)
    fun = lambda x: EIcalc_kd(x, xTrain_local, GPmod_local, yTrain_local)
    _, xk = optimizer.optimize(fun, iters=200)
    fk = np.transpose(calculate_robustness(xk, prob)) # budget + 1
    rho = (f0 - fk) / (OK_Rpredict(GPmod_local, x0.reshape(-1, 1), 0)[0] - OK_Rpredict(GPmod_local, xk.reshape(-1, 1), 0)[0])
    return xk, fk, rho[0][0]


### v2
# def local_gp_tr(x0, xTrain_local, yTrain_local):
#     # print(x0.shape, xTrain_local.shape, yTrain_local.shape)
#     f0 = calculate_robustness(x0, prob)
#     curSample_local = np.tile(0., (n_0 - m, nInputs))
#     curVal_local = np.tile(0., (n_0 - m, 1))
#     if n_0 - m > 0:
#         # draw a new lhs over the current TR
#         x0_local = lhs_sampling(n_0 - m, inpRanges, nInputs, rng)
#         x0_local = x0_local[0] * (TR[:, 1] - TR[:, 0]) + TR[:, 0]
#         for i in range(n_0 - m):
#             curSample_local[i, :] = x0_local[i, :]
#             curVal_local[i] = calculate_robustness(curSample_local[i, :], prob)
#         # add newly drawn points to list of all local points
#         # all_local_x = np.vstack([all_local_x, x0_local])
#         # all_local_y = np.vstack([all_local_y, curVal_local])
#         # all_x = np.vstack([all_x, x0_local])
#         # all_y = np.vstack([all_y, curVal_local])
#         xTrain_local = np.vstack([xTrain_local, x0_local])
#         yTrain_local = np.vstack([yTrain_local, curVal_local])
#     # Fit Gaussian Process Meta Model Locally
#     GPmod_local = OK_Rmodel_kd_nugget(xTrain_local, yTrain_local, 0, 2, 10)
#     options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
#     min_bound = np.ones(nInputs) * TR[:, 0]
#     max_bound = np.ones(nInputs) * TR[:, 1]
#     bounds = (min_bound, max_bound)
#     optimizer = LocalBestPSO(n_particles=n_0, dimensions=nInputs, options=options, bounds=bounds)
#     fun = lambda x: EIcalc_kd(x, xTrain_local, GPmod_local, yTrain_local)
#     bestEI, xk = optimizer.optimize(fun, iters=100)
#     # store acquisition function sample appropriately and sample it
#     curSample_local[0, :] = xk
#     xk = xk.reshape((1, 1, xk.shape[0]))
#     curVal_local = calculate_robustness(xk, prob)
#     fk = curVal_local
#     rho = (f0 - fk) / (OK_Rpredict(GPmod_local, x0[0], 0)[0] - OK_Rpredict(GPmod_local, xk[0], 0)[0])
#     return xk, fk, rho[0][0]


### v1 ###
# def gradient_based_tr(x0):
#     b = Bounds(TR_size / 2 * np.ones((2 * nInputs, 1)))
#     jac = Jacobian(problem.fHandle)
#     jac_x0 = jac(x0)
#     hes = Hessian(problem.fHandle)
#     hes_x0 = hes(x0)
#     f0 = calculate_robustness(x0, problem.fun)
#     fun = lambda s: quadratic_model(s, f0, jac_x0.T, hes_x0)
#     res = minimize(fun, x0, method='trust-constr', options={'maxiter': 1}, bounds=b)
#     sk = res['x'] # optimal step size
#     xk = x0 + sk
#     fk = problem.fun(xk)
#     rho = (f0 - fk) / (quadratic_model(np.zeros((1, nInputs)), f0, jac_x0.T, hes_x0) -
#                        quadratic_model(sk, f0, jac_x0.T, hes_x0))
#     return xk, fk, rho

# def local_gp_tr(x0):
#     TR_Bounds = [x0 - problem.lb, problem.ub - x0, (problem.ub - problem.lb)/10]
#     TR_size = np.min(TR_Bounds)
#     TR = [x0 - TR_size, x0 + TR_size]
#     n_0 = 5 * problem.dim
#     m = 1
#     x0_local = lhs_sampling(n_0-m, [0.0, 1.0], problem.dim, rng)
#     x0_local = x0_local * (TR[:, 2] - TR[:, 1]) + TR[:, 1]
#     xTrain_local = [xTrain_local, x0_local]
#     curVal_local = problem.fun(x0_local)
#     yTrain_local = [yTrain_local, curVal_local]
#     GPmod_local = OK_Rmodel_kd_nugget(xTrain_local, yTrain_local, 0, 2, 10)
#     options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
#     min_bound = np.ones(problem.dim) *  TR[0]
#     max_bound = np.ones(problem.dim) *  TR[1]
#     bounds = (min_bound, max_bound)
#     optimizer = LocalBestPSO(n_particles=n_0, dimensions=problem.dim, options=options, bounds=bounds)
#     fun = lambda x: EIcalc_kd(x, xTrain_local, GPmod_local, yTrain_local)
#     bestEI, xk = optimizer.optimize(fun, iters=100)
#     return xk, bestEI



