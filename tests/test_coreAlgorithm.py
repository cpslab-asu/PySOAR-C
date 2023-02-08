import math
# from pysoar.pysoar.coreAlgorithm.PySOAR import PySOAR
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.options import Options, SignalOptions
from staliro.core.interval import Interval
import numpy as np
from numpy.typing import NDArray
from staliro.staliro import staliro
from staliro.options import Options
from staliro.specifications import RTAMTDense

from pysoar.coreAlgorithm import PySOAR_Timed
from pysoar import PySOAR
# from test_functions import Himmelblau_2d
from pysoar.gprInterface import InternalGPR

import numpy as np


results_folder = "NonLinearFunction"
benchmark = "Himmelblaus"

# print(specification)

def test_function(X):
    return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.
    # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 
    

MAX_BUDGET = 500
NUMBER_OF_MACRO_REPLICATIONS = 1
trs_max_budget = 20

point_history, modes, simultation_time = PySOAR(
    n_0= 50,
    nSamples = MAX_BUDGET,
    trs_max_budget = trs_max_budget,
    max_loc_iter = 10,
    inpRanges = np.array([[-6.,0.],[-6.,6.]]),
    alpha_lvl_set = .95,
    eta0 = .25,
    eta1 = .75,
    delta = .75,
    gamma = 1.25,
    eps_tr = 0.01,
    prob = test_function,
    gpr_model = InternalGPR(),
    seed = 1234567,
    local_search= "gp_local_search",
    folder_name= results_folder,
    benchmark_name= f"{benchmark}_budget_{MAX_BUDGET}_{trs_max_budget}_maxtrs",
    behavior = "Minimization"
    )

# import pickle

# with open("ARCHCOMP2022_PYSOAR/AT1_budget_2000_1_reps/AT1_budget_2000_1_reps_seed_572502984.pkl", "rb") as f:
# # with open("NonLinearFunction/Himmelblaus_budget_500_20_maxtrs/Himmelblaus_budget_500_20_maxtrs_seed_1234567.pkl", "rb") as f:
#     point_history, modes, simultation_time = pickle.load(f)

# point_history = np.array(point_history, dtype = object)
# distinction = np.array(point_history[:,2])
# modes = np.array(point_history[:,3])
# regions = np.array(point_history[:,4])

# print(point_history.shape)
# subset_1 = point_history[modes == 0, :]
# print(np.array(subset_1[:,1]))


# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle
# fig = plt.figure()
        
# ax = fig.add_subplot(111)

# ax2 = 
# Look at 7, 11, 21, 23
# for i in list([38]):
#     # print(distinction == i)

#     subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
#     # subset_1 = point_history[distinction == i, :]

#     if subset_1.shape[0] != 0:
#         all_points = np.stack(subset_1[:,1])
        
#         ax1.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
#         for itera,p in enumerate(subset_1):
#             points = p[1]
#             region = p[4]
#             x = region[0,0]
#             y = region[1,0]
#             w = region[0,1] - region[0,0]
#             h = region[1,1] - region[1,0]

#             if itera == 0:
#                 ax1.plot(points[0], points[1], 'b*', markersize = 7)
#                 ax1.annotate(str(itera),xy=(points[0]+0.001, points[1]+0.001))
#                 ax1.add_patch( Rectangle((x,y),
#                                     w, h,
#                                     ec = 'black',
#                                     fc ='red',
#                                     alpha = 0.1))
#             else:
#                 ax1.plot(points[0], points[1], 'b.', markersize = 4)
#                 ax1.annotate(str(itera),xy=(points[0]+0.001, points[1]+0.001))
#                 ax1.add_patch( Rectangle((x,y),
#                                     w, h,
#                                     ec = 'black',
#                                     fc ='green',
#                                     alpha = 0.1))


# for i in np.unique(distinction):
#     # print(distinction == i)

#     subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
#     # subset_1 = point_history[distinction == i, :]

#     if subset_1.shape[0] != 0:
#         all_points = np.stack(subset_1[:,1])
#         ax2.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
#         for itera,p in enumerate(subset_1):
#             points = p[1]
#             region = p[4]
#             # print(points)
#             # print(region)
#             x = region[0,0]
#             y = region[1,0]
#             w = region[0,1] - region[0,0]
#             h = region[1,1] - region[1,0]

#             if itera == 0:
#                 ax2.plot(points[0], points[1], 'b*', markersize = 7)
#                 ax2.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
#             else:
#                 ax2.plot(points[0], points[1], 'b.', markersize = 4)
#                 ax2.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
#             # ax.add_patch( Rectangle((x,y),
#             #                     w, h,
#             #                     ec = 'black',
#             #                     fc ='green',
#             #                     alpha = 0.1))    


# for i in np.unique(distinction):
#     # print(distinction == i)

#     # subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
#     subset_1 = point_history[distinction == i, :]

#     if subset_1.shape[0] != 0:
#         all_points = np.stack(subset_1[:,1])
#         ax.plot(all_points[:,0], all_points[:,1], "b.", markersize = 4)
#         ax.plot(all_points[0,0], all_points[0,1], "b*", markersize = 7)
# ax1.set_aspect('equal', adjustable = 'box')
# ax2.set_aspect('equal', adjustable = 'box')
# plt.show()