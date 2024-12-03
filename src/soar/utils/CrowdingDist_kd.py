import numpy as np


def CrowdingDist_kd(x_0, x):
   
    # cd = np.sum(np.min(np.abs(x_0-x), 0))
    
    # cd = np.array(cd)
    for sample in x_0:
        cd = np.sum(np.sqrt(np.sum((sample - x)**2, 1)))
    return cd


