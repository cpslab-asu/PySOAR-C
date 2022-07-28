import numpy as np


def CrowdingDist_kd(x_0, x):
    DimWise_Crowd = np.zeros((1, x_0.shape[1]))
    cd_0 = np.zeros((x_0.shape[0], 1))
    for i in range(x_0.shape[0]):
        for j in range(x_0.shape[1]):
            DimWise_Crowd[0, j] = np.min(np.abs((np.ones((x.shape[0], 1)) * x_0[i, j]) - x[:, j]))
        cd_0[i, 0] = np.sum(DimWise_Crowd)
    return cd_0.flatten()


def neg_CrowdingDist_kd(x_0, x):
    DimWise_Crowd = np.zeros((1, x_0.shape[1]))
    cd_0 = np.zeros((x_0.shape[0], 1))
    for i in range(x_0.shape[0]):
        for j in range(x_0.shape[1]):
            DimWise_Crowd[0, j] = np.min(np.abs((np.ones((x.shape[0], 1)) * x_0[i, j]) - x[:, j]))
        cd_0[i, 0] = np.sum(DimWise_Crowd)
    return -cd_0.flatten()


