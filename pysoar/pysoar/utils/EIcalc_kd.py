from ..kriging_gpr.interface import OK_Rpredict

import numpy as np
from scipy.stats import norm



def EIcalc_kd(x_0, x, M_model, y):
    curr_best = min(y)
    curr_best_ind = np.argmin(y)
    b_0 = np.ones((x_0.shape[0], 1))
    y_0, s_0 = OK_Rpredict(M_model, x_0, 0)
    i = 0
    found = 0
    while i < x_0.shape[0] and found == 0:
        if x_0[i, :].all() == x[curr_best_ind, :].all():
            curr_best = y_0[i]
            found = 1
        else:
            i += 1
    counts = x_0.shape[0]
    ei_0 = np.zeros((x_0.shape[0], 1))
    s_0 = s_0.flatten().astype(float)
    s_0 = np.sqrt(s_0).reshape(-1, 1)
    for i in range(counts):
        if s_0[i] > 0:
            ei_0[i] = (curr_best - y_0[i]) * norm.cdf((curr_best-y_0[i])/s_0[i]) \
                      + s_0[i] * norm.pdf((curr_best-y_0[i])/s_0[i])
        else:
            ei_0[i] = 0
    return ei_0.flatten() #, s_0, y_0

def neg_EIcalc_kd(x_0, x, M_model, y):
    curr_best = min(y)
    curr_best_ind = np.argmin(y)
    b_0 = np.ones((x_0.shape[0], 1))
    y_0, s_0 = OK_Rpredict(M_model, x_0, 0)
    i = 0
    found = 0
    while i < x_0.shape[0] and found == 0:
        if x_0[i, :].all() == x[curr_best_ind, :].all():
            curr_best = y_0[i]
            found = 1
        else:
            i += 1
    counts = x_0.shape[0]
    ei_0 = np.zeros((x_0.shape[0], 1))
    s_0 = s_0.flatten().astype(float)
    s_0 = np.sqrt(s_0).reshape(-1, 1)
    for i in range(counts):
        if s_0[i] > 0:
            ei_0[i] = (curr_best - y_0[i]) * norm.cdf((curr_best - y_0[i]) / s_0[i]) \
                      + s_0[i] * norm.pdf((curr_best - y_0[i]) / s_0[i])
        else:
            ei_0[i] = 0
    return -ei_0.flatten()