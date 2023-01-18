import numpy as np
import pickle
from shapely.geometry import Polygon
from ha_tf import HA

seed = 352427
fr = []
sim = []
for i in range(50):
# i = 6
    with open(f"Thesis/2state_ha_pysoar_budget_300_15_maxtrs_1/2state_ha_pysoar_budget_300_15_maxtrs_1_seed_{seed + i}.pkl","rb") as f:
        data = pickle.load(f)
    point_history = np.array(data[0], dtype = object)
    if (point_history[-1,5]) <= 0:
        fr.append(1)
    else:
        fr.append(0)
    sim.append(point_history[-1,0])
print(sum(fr), sim)   
print(np.mean(sim))
print(np.median(sim))
run = 9
with open(f"Thesis/2state_ha_pysoar_budget_300_15_maxtrs_1/2state_ha_pysoar_budget_300_15_maxtrs_1_seed_{seed + run}.pkl","rb") as f:
    point_history, modes, simultation_time = pickle.load(f)

point_history = np.array(point_history, dtype = object)
distinction = np.array(point_history[:,2])
modes = np.array(point_history[:,3])
regions = np.array(point_history[:,4])

print(point_history.shape)
# subset_1 = point_history[modes == 0, :]
# print(np.array(subset_1[:,1]))


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
fig = plt.figure()
        
ax = fig.add_subplot(111)


global_handles = []
local_handles = []
for i in np.unique(distinction):
    # print(distinction == i)

    subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
    # subset_1 = point_history[distinction == i, :]

    if subset_1.shape[0] != 0:
        all_points = np.stack(subset_1[:,1])
        ax.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
        for itera,p in enumerate(subset_1):
            points = p[1]
            region = p[4]
            # print(points)
            # print(region)
            x = region[0,0]
            y = region[1,0]
            w = region[0,1] - region[0,0]
            h = region[1,1] - region[1,0]

            if itera == 0:
                p1, = ax.plot(points[0], points[1], 'k*', markersize = 7, label = "Global Points")
                # ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
                global_handles.append(p1)
            else:
                p2, = ax.plot(points[0], points[1], 'b.', markersize = 4, label = "Local Points")
                local_handles.append(p2)
            
            
                # ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
            # ax.add_patch( Rectangle((x,y),
            #                     w, h,
            #                     ec = 'black',
            #                     fc ='green',
            #                     alpha = 0.1))    


for i in np.unique(distinction):
    # print(distinction == i)

    # subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
    subset_1 = point_history[distinction == i, :]

    if subset_1.shape[0] != 0:
        all_points = np.stack(subset_1[:,1])
        p1 = ax.plot(all_points[:,0], all_points[:,1], "b.", markersize = 4)
        p2 = ax.plot(all_points[0,0], all_points[0,1], "k*", markersize = 7)
ax.set_aspect('equal', adjustable = 'box')
# ax2.set_aspect('equal', adjustable = 'box')
ax.legend([global_handles[0],local_handles[0]], ["Global Points", "Local Points"], loc = 6)
plt.savefig(f'pysoar_samples_f_{sim[run]}.png', bbox_inches='tight', dpi = 500, transparent = True)

fig, ax = plt.subplots()
all_points = np.stack(point_history[:,1])
ha = HA()
for x,y in all_points:
    time, traj = ha._generate_traj([x,y])
    plt.plot(traj[:,0], traj[:,1], "-b", linewidth = 0.5)

green_plot = Polygon([(-1,-1), (-1,1),(1,1),(1,-1)])
yellow_plot = Polygon([(0.85,0.85), (0.85,0.95),(0.95,0.95),(0.95,0.85)])
unsafe_1 = Polygon([(-1.8,-1.6), (-1.8,-1.4), (-1.4, -1.4), (-1.4,-1.6)])
unsafe_2 = Polygon([(3.7, -1.6), (3.7,-1.4), (4.1,-1.4), (4.1,-1.6)])

x_green, y_green = green_plot.exterior.xy
x_yellow, y_yellow = yellow_plot.exterior.xy
x_unsafe_1, y_unsafe_1 = unsafe_1.exterior.xy
x_unsafe_2, y_unsafe_2 = unsafe_2.exterior.xy

ax.plot(x_green, y_green, color='green', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax.plot(x_yellow, y_yellow, color='yellow', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax.plot(x_unsafe_1, y_unsafe_1, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax.plot(x_unsafe_2, y_unsafe_2, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
ax.set_xlim(-2,4)
ax.set_ylim(-2,2)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
# plt.grid()
plt.savefig(f"pysoar_samples_f_dynamics_{sim[run]}.png", bbox_inches='tight', dpi = 500, transparent = True)
#######################################################################################
