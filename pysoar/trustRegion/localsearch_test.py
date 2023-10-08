import numpy as np
from local_search import *
from test_functions import *
from CrowdingDist_kd import *

nInputs = 10
n_0 = 10 * nInputs
rng = np.random.default_rng(1000)
# curSample = np.zeros((n_0, nInputs))
# curVal = np.zeros((n_0, 1))
curSample = np.tile(0, (n_0, nInputs))
curVal = np.tile(0, (n_0, 1))
inpRanges = np.full([1, nInputs, 2], [0., 1.]) # (1, 10, 2)
x_0 = lhs_sampling(n_0, inpRanges, nInputs, rng)
x_0 = x_0[0] * (inpRanges[0][:, 1] - inpRanges[0][:, 0]) + inpRanges[0][:, 0]
m = 1
for i in range(n_0):
    curSample[i, :] = x_0[i, :]
    curVal[i] = calculate_robustness(curSample[i], callCounter(griewank))[0].T
xTrain = np.array(curSample[0])
yTrain = np.array(curVal[0])
all_x = xTrain
all_y = yTrain
x0 = lhs_sampling(1, inpRanges, nInputs, rng)
f0 = calculate_robustness(x0, callCounter(griewank)).T
all_local_x = x0[0]
all_local_y = f0[0]
xTrain_local = np.zeros((n_0 + 1, nInputs))
yTrain_local = np.zeros((n_0 + 1, 1))
xTrain_local[0, :] = x0
yTrain_local[0, 0] = f0
TR_Bounds = np.vstack([x0[0] - 0., 1. - x0[0], 1./10*np.ones((1, nInputs))])
TR_size = np.min(TR_Bounds)
TR = np.hstack([x0[0].T - TR_size, x0[0].T + TR_size]) # (2, nInputs)
prob = callCounter(griewank)

def gradient_based_tr(x0, TR_size, nInputs, prob):
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
    ub = (TR_size/2) * np.ones((2*nInputs, ))
    lb = 0 * np.ones((2*nInputs, ))
    jac = Jacobian(prob)
    jac_x0 = jac(x0.flatten())
    hes = Hessian(prob)
    hes_x0 = hes(x0.flatten())
    f0 = calculate_robustness(x0, prob)
    fun = lambda s: quadratic_model(s, f0, jac_x0.T, hes_x0)
    linear_constraint = LinearConstraint(a, lb, ub)
    res = minimize(fun, x0.flatten(), method='trust-constr', constraints=[linear_constraint, ])
    sk = res['x'] # optimal step size
    xk = x0 + sk
    fk = calculate_robustness(xk, prob)
    rho = (f0 - fk) / (quadratic_model(np.zeros((1, nInputs)), f0, jac_x0.T, hes_x0) -
                       quadratic_model(sk, f0, jac_x0.T, hes_x0))
    return xk, fk, rho

xk, fk, rho = gradient_based_tr(x0, TR_size, nInputs, prob)
print(xk, fk, rho)


def local_gp_tr(x0, nInputs, prob, xTrain_local, yTrain_local):
    # Fit Gaussian Process Meta Model Locally
    GPmod_local = OK_Rmodel_kd_nugget(xTrain_local, yTrain_local, 0, 2, 10)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    min_bound = np.ones(nInputs) * TR[:, 0]
    max_bound = np.ones(nInputs) * TR[:, 1]
    bounds = (min_bound, max_bound)
    optimizer = LocalBestPSO(n_particles=n_0, dimensions=nInputs, options=options, bounds=bounds)
    fun = lambda x: EIcalc_kd(x, xTrain_local, GPmod_local, yTrain_local)
    bestEI, xk = optimizer.optimize(fun, iters=100)
    # store acquisition function sample appropriately and sample it
    curSample_local[0, :] = xk
    xk = xk.reshape((1, 1, xk.shape[0]))
    curVal_local = calculate_robustness(xk, prob)
    fk = curVal_local
    # store TODO
    rho = (f0 - fk) / (OK_Rpredict(GPmod_local, x0[0], 0)[0] - OK_Rpredict(GPmod_local, xk[0], 0)[0])
    return xk, fk, rho[0][0]

xk, fk, rho = local_gp_tr(x0, nInputs, prob, xTrain_local, yTrain_local)
print(xk, fk, rho)


