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

from pysoarc.coreAlgorithm import Behavior

# from pysoar.coreAlgorithm import PySOAR_Timed
from pysoarc import PySOARC
# from test_functions import Himmelblau_2d
from pysoarc.gprInterface import InternalGPR

import numpy as np


results_folder = "NonLinearFunction_mo"
benchmark = "Himmelblaus"

# print(specification)

# def test_function(X):
#     return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.
#     # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 
    
from ha_tf import HA

ha = HA()

MAX_BUDGET = 500
NUMBER_OF_MACRO_REPLICATIONS = 100
trs_max_budget = 10
import pickle

for i in range(NUMBER_OF_MACRO_REPLICATIONS):
    starting_seed = 1234565+i
    point_history  = PySOARC(
        n_0= 20,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=10,
        inpRanges = np.array([[-1.,1.],[-1.,1.]]),
        alpha_lvl_set = .05,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.01,
        min_tr_size=10.0,
        TR_threshold=0.05,
        test_fn = ha.get_cost,
        gpr_model = InternalGPR(),
        seed = starting_seed,
        local_search= "gp_local_search",
        behavior = Behavior.FALSIFICATION
        )
    # with open(f"/home/daittan/RA_Work/pysoar-c/demos/res/pysoarc_max500_rep_50_seed{starting_seed}.pickle", "wb") as f:
    #     pickle.dump(point_history, f)



# # with open("ARCHCOMP2022_PYSOAR/AT1_budget_2000_1_reps/AT1_budget_2000_1_reps_seed_572502984.pkl", "rb") as f:
# # with open("NonLinearFunction_mo/Himmelblaus_budget_250_10_maxtrs/Himmelblaus_budget_250_10_maxtrs_seed_1234567.pkl", "rb") as f:




# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle
# fig = plt.figure()
        
# ax = fig.add_subplot(111)

# # ax2 = 
# # Look at 7, 11, 21, 23
# # for i in list([38]):
# #     # print(distinction == i)

# #     subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
# #     # subset_1 = point_history[distinction == i, :]

# #     if subset_1.shape[0] != 0:
# #         all_points = np.stack(subset_1[:,1])
        
# #         ax1.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
# #         for itera,p in enumerate(subset_1):
# #             points = p[1]
# #             region = p[4]
# #             x = region[0,0]
# #             y = region[1,0]
# #             w = region[0,1] - region[0,0]
# #             h = region[1,1] - region[1,0]

# #             if itera == 0:
# #                 ax1.plot(points[0], points[1], 'b*', markersize = 7)
# #                 ax1.annotate(str(itera),xy=(points[0]+0.001, points[1]+0.001))
# #                 ax1.add_patch( Rectangle((x,y),
# #                                     w, h,
# #                                     ec = 'black',
# #                                     fc ='red',
# #                                     alpha = 0.1))
# #             else:
# #                 ax1.plot(points[0], points[1], 'b.', markersize = 4)
# #                 ax1.annotate(str(itera),xy=(points[0]+0.001, points[1]+0.001))
# #                 ax1.add_patch( Rectangle((x,y),
# #                                     w, h,
# #                                     ec = 'black',
# #                                     fc ='green',
# #                                     alpha = 0.1))


# from pysoarc.coreAlgorithm.PySOARC import GlobalPhase, LocalPhase, LocalBest, InitializationPhase
# point_streak = []
# p1_handles = []
# p2_handles = []
# p3_handles = []
# p4_handles = []

# for point_types in point_history:
    
#     if type(point_types) is InitializationPhase:
#         p1, = ax.plot(
#             point_types.initial_samples_x[:,0], 
#             point_types.initial_samples_x[:,1],
#             'b*', 
#             markersize = 7, 
#             label = "Initial Points"
#         )
#         p1_handles.append(p1)
#     elif type(point_types) is GlobalPhase:
#         p2, = ax.plot(
#             point_types.restart_point_x[:,0], 
#             point_types.restart_point_x[:,1],
#             'k*', 
#             markersize = 7, 
#             label = "Global Points"
#         )
#         p2_handles.append(p2)
#     elif type(point_types) == LocalPhase:
#         p3, = ax.plot(
#             point_types.local_phase_x[:,0], 
#             point_types.local_phase_x[:,1],
#             'b.', 
#             markersize = 7, 
#             label = "Local Points"
#         )
#         p3_handles.append(p3)
#     elif type(point_types) == LocalBest:
#         p4, = ax.plot(
#             point_types.local_best_x[:,0], 
#             point_types.local_best_x[:,1],
#             'k.', 
#             markersize = 7, 
#             label = "Local Points"
#         )
#         p4_handles.append(p4)
    
# ax.legend([p1_handles[0],p2_handles[0], p3_handles[0], p4_handles[0]], 
#             ["Initial Points", "Global Points", "Local Points", "Local Best"])
# plt.savefig('pysoarc_samples.pdf') 


# # global_handles = []
# # local_handles = []
# # # for i in np.unique(distinction):
# # #     # print(distinction == i)

# # #     subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
# # #     # subset_1 = point_history[distinction == i, :]

# # #     if subset_1.shape[0] != 0:
# # #         all_points = np.stack(subset_1[:,1])
# # #         ax.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
# # #         for itera,p in enumerate(subset_1):
# # #             points = p[1]
# # #             region = p[4]
# # #             # print(points)
# # #             # print(region)
# # #             x = region[0,0]
# # #             y = region[1,0]
# # #             w = region[0,1] - region[0,0]
# # #             h = region[1,1] - region[1,0]

# # #             if itera == 0:
# # #                 p1, = ax.plot(points[0], points[1], 'k*', markersize = 7, label = "Global Points")
# # #                 # ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
# # #                 global_handles.append(p1)
# # #             else:
# # #                 p2, = ax.plot(points[0], points[1], 'b.', markersize = 4, label = "Local Points")
# # #                 local_handles.append(p2)
            
            
# # #                 # ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
# # #             # ax.add_patch( Rectangle((x,y),
# # #             #                     w, h,
# # #             #                     ec = 'black',
# # #             #                     fc ='green',
# # #             #                     alpha = 0.1))    


# # # # for i in np.unique(distinction):
# # # #     # print(distinction == i)

# # # #     # subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
# # # #     subset_1 = point_history[distinction == i, :]

# # # #     if subset_1.shape[0] != 0:
# # # #         all_points = np.stack(subset_1[:,1])
# # # #         p1 = ax.plot(all_points[:,0], all_points[:,1], "b.", markersize = 4)
# # # #         p2 = ax.plot(all_points[0,0], all_points[0,1], "k*", markersize = 7)
# # # # ax.set_aspect('equal', adjustable = 'box')
# # # # ax2.set_aspect('equal', adjustable = 'box')
# # ax.legend([global_handles[0],local_handles[0]], ["Global Points", "Local Points"])
