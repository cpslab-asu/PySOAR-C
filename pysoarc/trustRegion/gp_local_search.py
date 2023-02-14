from ..kriging_gpr.interface import OK_Rpredict, OK_Rmodel_kd_nugget
from ..utils import compute_robustness, quadratic_model, EIcalc_kd
from ..gprInterface import GPR
from ..sampling import uniform_sampling

from copy import deepcopy
import numpy as np
import math
from scipy.optimize import minimize

def local_best_ei(pred_sample_x, pred_sample_y, tf_wrapper, test_fn, tf_dim, trust_region, xTrain_local, yTrain_local, behavior, gpr_model, rng):
    # Fit Gaussian Process Meta Model Locally
    gpr = GPR(deepcopy(gpr_model))
    gpr.fit(xTrain_local, yTrain_local)

    EI_obj = lambda x: -1*EIcalc_kd(yTrain_local, x, gpr)
    lower_bound_theta = np.ndarray.flatten(trust_region[:, 0])
    upper_bound_theta = np.ndarray.flatten(trust_region[:, 1])
    
    random_samples = uniform_sampling(2000, trust_region, tf_dim, rng)
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
    xk = np.array([np.array(new_params.x)])
    
    rob = tf_wrapper(xk, test_fn, behavior)
    # print(pred_sample_y[0][0])
    # print(rob[0][0])
    # print(gpr.predict(pred_sample_x)[0])
    # print(gpr.predict(xk)[0])
    rho = (pred_sample_y[0][0] - rob[0][0]) / (gpr.predict(pred_sample_x)[0] - gpr.predict(xk)[0])

    return xk, rob, rho[0]
    