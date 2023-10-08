from .CrowdingDist_kd import CrowdingDist_kd
from .EIcalc_kd import EIcalc_kd
import numpy as np

def ei_cd(samples, x_train, y_train, gpr, alpha_lvl_set, EI_star):

    if len(samples.shape) == 1:
        c = EIcalc_kd(y_train, samples, gpr) - (alpha_lvl_set * (EI_star))
        if c >= 0:
            ret = -1 * CrowdingDist_kd(samples, x_train)
        else:
            ret = 0
    elif len(samples.shape) > 1:
        ret = []
        # print(x)
        for sample in samples:
            c = EIcalc_kd(y_train, sample, gpr) - (alpha_lvl_set * (EI_star))
            # print(c[i])
            if c >= 0:
                ret.append(-1 * CrowdingDist_kd(sample, x_train))
            else:
                ret.append(0)
        ret = np.array(ret)
    return ret