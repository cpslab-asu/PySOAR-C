import numpy as np
import csv
import matplotlib.pyplot as plt
from test_functions import *
import pandas as pd


# # load from npy the dictionaries into lists
# samples = []
# f_star = []
# TR_budget_used = []
# rob = []
# for repNum in range(5):
#     history = np.load('run_results/pysoar_rep{}.npy'.format(repNum), allow_pickle=True).item()
#     samples.append(history['samples'])
#     rob.append(history['rob'])
#     f_star.append(history['f_star'])
#     TR_budget_used.append(history['TR_budget_used'])
#
# # sample plots with function contour overlaid
# local_minima = np.array([[3., 2.], [-2.805118, 3.131312], [-3.77931, -3.283186], [3.584428, -1.848126]])
# for repNum in range(5):
#     fig = plt.figure(figsize=(8, 8))
#     npts = 201
#     x, y = np.mgrid[-6:6:npts * 1j, -6:6:npts * 1j]
#     z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 - 5.
#     levels = np.logspace(-10, 0, 10)
#     plt.contour(x, y, z, levels, cmap="viridis")
#     plt.xlabel('$x_1$')
#     plt.ylabel('$x_2$')
#     plt.xticks([-6, -3, 0, 3, 6])
#     plt.yticks([-6, -3, 0, 3, 6])
#     plt.xlim([-6, 0])
#     plt.ylim([-6, 6])
#     plt.scatter(samples[repNum][:, 0], samples[repNum][:, 1], s=10)
#     plt.scatter(local_minima[:, 0], local_minima[:, 1], c='y')
#     plt.title('Samples for rep {}'.format(repNum))
#     plt.savefig('run_results/samples_rep{}.pdf'.format(repNum))
#     plt.show()
#
# # f* with 95% CI over macro replications
# B = 300
# ci = 1.96 * np.std(np.array(f_star), axis=0)/ np.sqrt(B)
# avg = np.mean(np.array(f_star), axis=0)
# fig, ax = plt.subplots()
# ax.plot(np.arange(B), avg, lw=0.8)
# ax.fill_between(np.arange(B), (avg - ci).flatten(), (avg + ci).flatten(), color='orange', alpha=0.25, ls='--', lw=0.5)
# ax.set_xlabel('simulation')
# ax.set_ylabel('$f^*$')
# plt.savefig('run_results/bestrob_95CI.pdf')
#
# # store result into tables for each sim and TR budget used for each local loop
# for repNum in range(5):
#     TRsim = np.arange(len(TR_budget_used[repNum])) + 1
#     TRdata = np.array([TRsim, TR_budget_used[repNum]])
#     TRtable = pd.DataFrame(TRdata, index=['TR sim', 'budget used']).transpose()
#     pd.DataFrame.to_csv(TRtable, 'run_results/TR_sim_rep{}.csv'.format(repNum))
#
#     sim = np.arange(1, B+1)
#     result = np.hstack([samples[repNum], rob[repNum]])
#     res_data = np.hstack([sim.reshape(-1, 1), result])
#     res_table = pd.DataFrame(res_data.T, index=['sim', '$x1$', '$x2$', '$f$']).transpose()
#     pd.DataFrame.to_csv(res_table, 'run_results/result_rep{}.csv'.format(repNum))


# DEMO: state map with restart-local samples trajectories
restart = np.array([[-4.35530356, -5.02388361],
       [-4.35530356, -5.02388361],
       [-4.35530356, -5.02388361],
       [-4.35530356, -5.02388361],
       [-4.35530356, -5.02388361],
       [-4.35530356, -5.02388361],
       [-4.35530356, -5.02388361],
       [-2.01519658, -3.81907827], #7
       [-2.01519658, -3.81907827],
       [-2.01519658, -3.81907827],
       [-2.01519658, -3.81907827],
       [-2.01519658, -3.81907827],
       [-2.01519658, -3.81907827],
       [-2.01519658, -3.81907827],
       [-2.01519658, -3.81907827],
       [-4.28648909,  4.05557573], #15
       [-4.28648909,  4.05557573],
       [-4.28648909,  4.05557573],
       [-4.28648909,  4.05557573],
       [-4.28648909,  4.05557573],
       [-4.28648909,  4.05557573],
       [-4.28648909,  4.05557573],
       [-4.28648909,  4.05557573],
       [-5.39770083, -0.58911282], #23
       [-5.39770083, -0.58911282],
       [-5.39770083, -0.58911282],
       [-5.39770083, -0.58911282],
       [-5.39770083, -0.58911282],
       [-5.39770083, -0.58911282],
       [-5.39770083, -0.58911282],
       [-5.39770083, -0.58911282],
       [-4.30057412, -1.93619448],# 31
       [-4.30057412, -1.93619448],
       [-4.30057412, -1.93619448],
       [-4.30057412, -1.93619448],
       [-4.30057412, -1.93619448],
       [-4.30057412, -1.93619448],
       [-4.30057412, -1.93619448],
       [-4.30057412, -1.93619448],
       [-1.03915119, -0.13138279],#39
       [-1.03915119, -0.13138279],
       [-1.03915119, -0.13138279],
       [-1.03915119, -0.13138279]])

xk = np.array([[-4.14513699, -4.72388361],
       [-4.04091342, -4.34888361],
       [-3.92499384, -3.88013361],
       [-3.80736178, -3.3999537 ],
       [-3.78071167, -3.28908935],
       [-3.77931411, -3.28320232],
       [-3.77931166, -3.28318393],
       [-2.31519658, -3.51907827],
       [-2.69019658, -3.14407827],
       [-3.15894658, -3.07900098],
       [-3.74488408, -3.27571995],
       [-3.77980137, -3.2833245 ],
       [-3.77927054, -3.28324446],
       [-3.77932134, -3.28316972],
       [-3.77930717, -3.28319052],
       [-3.98648909,  3.75557573],
       [-3.61148909,  3.38057573],
       [-3.14273909,  3.15864825],
       [-2.85200731,  3.13405157],
       [-2.8062155 ,  3.13137094],
       [-2.80511872,  3.13131256],
       [-2.80511841,  3.13131225],
       [-2.80511841,  3.13131225],
       [-5.09770083, -0.88911282],
       [-4.72270083, -1.26411282],
       [-4.25395083, -1.73286282],
       [-3.69676695, -2.31880032],
       [-3.7131666 , -3.0512222 ],
       [-3.7869564 , -3.31362087],
       [-3.77948394, -3.28368356],
       [-3.77934689, -3.28313243],
       [-4.00057412, -2.23619448],
       [-3.66788693, -2.61119448],
       [-3.72650696, -3.07994448],
       [-3.78483225, -3.3059774 ],
       [-3.77944373, -3.28350336],
       [-3.77933104, -3.28315558],
       [-3.77930448, -3.28319447],
       [-3.77931186, -3.28318363],
       [-1.33915119,  0.16861721],
       [-1.71415119,  0.54361721],
       [-2.18290119,  1.01236721],
       [-2.76883869,  1.59830471]])

traj1 = np.vstack([restart[0, :], xk[:7, :]])
traj2 = np.vstack([restart[7, :], xk[7:15, :]])
traj3 = np.vstack([restart[15, :], xk[15:23, :]])
traj4 = np.vstack([restart[23, :], xk[23:31, :]])
traj5 = np.vstack([restart[31, :], xk[31:39, :]])

fig = plt.figure(figsize=(8, 8))
# npts = 201
# x, y = np.mgrid[-6:6:npts * 1j, -6:6:npts * 1j]
# z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 - 5.
# levels = np.logspace(-10, 0, 10)
# plt.contour(x, y, z, levels, cmap="viridis",linewidths=0.5)
plt.plot([-4.13, -4.13], [-6, 6], ls='--', color='black', lw=0.5)
plt.plot([-4.13, -3.4], [-3.7, -3.7], ls='--', color='black', lw=0.5)
plt.plot([-4.13, -3.4], [-2.82, -2.82], ls='--', color='black', lw=0.5)
plt.plot([-3.4, -3.4], [-6, 6], ls='--', color='black', lw=0.5)
plt.plot([-3.4, -2.3], [2.65, 2.65], ls='--', color='black', lw=0.5)
plt.plot([-3.4, -2.3], [3.55, 3.55], ls='--', color='black', lw=0.5)
plt.plot([-2.3, -2.3], [-6, 6], ls='--', color='black', lw=0.5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xticks([-6, -4, -2, 0, 2, 4, 6])
plt.yticks([-6, -4, -2, 0, 2, 4, 6])
plt.xlim([-6, 0])
plt.ylim([-6, 6])
plt.plot(traj1[:, 0], traj1[:, 1], marker='o', markersize=1.5, label='local search 1', lw=0.8)
plt.plot(traj2[:, 0], traj2[:, 1], marker='o', markersize=1.5, label='local search 2', lw=0.8)
plt.plot(traj3[:, 0], traj3[:, 1], marker='o', markersize=1.5, label='local search 3', lw=0.8)
plt.plot(traj4[:, 0], traj4[:, 1], marker='o', markersize=1.5, label='local search 4', lw=0.8)
plt.plot(traj5[:, 0], traj5[:, 1], marker='o', markersize=1.5, label='local search 5', lw=0.8)
# plt.plot(traj6[:, 0], traj6[:, 1], marker='o', markersize=2, label='6')
# plt.plot(traj7[:, 0], traj7[:, 1], marker='o', markersize=2, label='7')
plt.scatter(traj1[0, 0], traj1[0, 1], s=30, marker='*')
plt.scatter(traj2[0, 0], traj2[0, 1], s=30, marker='*')
plt.scatter(traj3[0, 0], traj3[0, 1], s=30, marker='*')
plt.scatter(traj4[0, 0], traj4[0, 1], s=30, marker='*')
plt.scatter(traj5[0, 0], traj5[0, 1], s=30, marker='*')
# plt.scatter(traj6[0, 0], traj6[0, 1], color='r', s=50)
# plt.scatter(traj7[0, 0], traj7[0, 1], color='r', s=50)
plt.legend()
plt.savefig('run_results/pysoar_traj_demo.pdf')


# density plot
fig = plt.figure(figsize=(8, 8))
npts = 201
x, y = np.mgrid[-6:6:npts * 1j, -6:6:npts * 1j]
z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 - 5.
levels = np.logspace(-10, 0, 10)
plt.contour(x, y, z, levels, cmap="viridis", linewidths=0.5)
plt.plot([-4.13, -4.13], [-6, 6], ls='--', color='black', lw=0.5)
plt.plot([-4.13, -3.4], [-3.7, -3.7], ls='--', color='black', lw=0.5)
plt.plot([-4.13, -3.4], [-2.82, -2.82], ls='--', color='black', lw=0.5)
plt.plot([-3.4, -3.4], [-6, 6], ls='--', color='black', lw=0.5)
plt.plot([-3.4, -2.3], [2.65, 2.65], ls='--', color='black', lw=0.5)
plt.plot([-3.4, -2.3], [3.55, 3.55], ls='--', color='black', lw=0.5)
plt.plot([-2.3, -2.3], [-6, 6], ls='--', color='black', lw=0.5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xticks([-6, -4, -2, 0, 2, 4, 6])
plt.yticks([-6, -4, -2, 0, 2, 4, 6])
plt.xlim([-6, 0])
plt.ylim([-6, 6])
repNum = 100
history = np.load('run_results/pysoar_rep{}.npy'.format(repNum), allow_pickle=True).item()
global_samples = history['global_samples']
local_samples = history['local_samples']
plt.scatter(global_samples[:, 0], global_samples[:, 1], label='global samples', s=10)
plt.scatter(local_samples[:, 0], local_samples[:, 1], label='local samples', s=3)
plt.legend()
plt.savefig('run_results/pysoar_density.pdf')