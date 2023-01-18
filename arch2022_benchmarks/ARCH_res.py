

import numpy as np
import pickle
import os
dirs = os.listdir("ARCHCOMP2022_PYSOAR/AT1_budget_2000_10_reps_modeicd/")
print(dirs)

ff = []
for files in dirs:
    with open(f"ARCHCOMP2022_PYSOAR/AT1_budget_2000_10_reps_modeicd/{files}", "rb") as f:
        point_history, modes, simultation_time = pickle.load(f)

    point_history = np.array(point_history, dtype = object)
    ff.append(point_history.shape[0])

print(ff)
print(np.min(ff))
print(np.mean(ff))
print(np.median(ff))
print(np.max(ff))

# point_history = np.array(point_history, dtype = object)
# distinction = np.array(point_history[:,2])
# modes = np.array(point_history[:,3])
# regions = np.array(point_history[:,4])

# print(point_history.shape)
# print( np.unique(distinction))

# all_local_search = []
# for i in np.unique(distinction):
#     # print(distinction == i)

#     subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
#     # subset_1 = point_history[distinction == i, :]

#     if subset_1.shape[0] != 0:
#         local_search = []
#         for itera,p in enumerate(subset_1):
#             local_search.append(p[5])

#         all_local_search.append(local_search)

# from matplotlib import pyplot as plt

# fig = plt.figure()
        
# ax = fig.add_subplot(111)
# for itera, i in enumerate(all_local_search):
#     x_pos = [x for x in range(len(i))]
#     ax.plot(x_pos, i, markersize = 0.1)
#     for x,y in zip(x_pos, i):
#         ax.plot(x,y,".")
#         ax.annotate(f"{itera}",xy=(x+0.001, y+0.001))

# plt.show()