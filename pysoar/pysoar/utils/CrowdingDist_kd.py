import numpy as np


def CrowdingDist_kd(x_0, x):
    # np.sum(np.min(np.abs(x_0[0,:]-x), 0))
    # if len(x_0.shape) == 1:
    cd = np.sum(np.min(np.abs(x_0-x), 0))
    # elif len(x_0.shape) > 1:
    #     cd = []
    #     for sample in x_0:
    #         cd.append(np.sum(np.min(np.abs(x_0-x), 0)))
    cd = np.array(cd)
    return cd


