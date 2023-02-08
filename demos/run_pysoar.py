from pysoar import PySOAR
# from test_functions import Himmelblau_2d

import numpy as np

# nInputs = 2
# n_0 = 10 * nInputs

def Himmelblau_2d(X):
    return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.


inpRanges = np.array([[[-5.,5.],[-5.,5.]], [[-5.,5.],[-5.,5.]]])

# DEMO
repNum = 50
_, history = PySOAR(inpRanges, Himmelblau_2d, repNum, local_search='gradient')
print(history)
np.save('pysoar_rep{}.npy'.format(repNum), history)


# # run macroreplications
# macrorep = 2
# falsified = np.zeros((macrorep, 1))
# best_rob = np.zeros((macrorep, 1))
# for repNum in range(macrorep):
#     run, history = PySOAR(inpRanges, prob, repNum, local_search='gradient')
#     best_rob[repNum, 0] = run['bestRob']
#     if best_rob[repNum, 0] <= 0:
#         falsified[repNum, 0] = 1
# print(falsified, best_rob)