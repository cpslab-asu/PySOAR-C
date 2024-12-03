from copy import deepcopy

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from numdifftools import Jacobian, Hessian
from pyswarms.single import LocalBestPSO
from scipy.optimize import minimize

from .kriging import OK_Rpredict, OK_Rmodel_kd_nugget
from .utils import compute_robustness, quadratic_model, EIcalc_kd
from .gpr import GPR
from .sampling import uniform_sampling


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


def local_best_ei(pred_sample_x, pred_sample_y, tf_wrapper, tf_dim, trust_region, xTrain_local, yTrain_local, behavior, gpr_model, rng):
    # Fit Gaussian Process Meta Model Locally
    gpr = GPR(deepcopy(gpr_model))
    gpr.fit(xTrain_local, yTrain_local)

    EI_obj = lambda x: -1*EIcalc_kd(yTrain_local, x, gpr)
    lower_bound_theta = np.ndarray.flatten(trust_region[:, 0])
    upper_bound_theta = np.ndarray.flatten(trust_region[:, 1])
    
    random_samples = uniform_sampling(1000, trust_region, tf_dim, rng)
    min_bo_val = EI_obj(random_samples)

    min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])[0,:]
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
    xk = np.array([np.array(new_params.x)])
    
    fk, falsified = compute_robustness(xk, 3, behavior, trust_region, tf_wrapper)
    
    rho = (pred_sample_y - fk) / (gpr.predict(pred_sample_x)[0] - gpr.predict(xk)[0])

    return xk, fk, rho[0], falsified
