

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pysoarc.coreAlgorithm.PySOARC import InitializationPhase, GlobalPhase, LocalPhase, LocalBest
import pickle
import numpy as np

def _generate_dataset(output_type: int, *args):
    if output_type not in [0,1]:
        raise ValueError
    
    print(type(args[0][0]))
    if type(args[0][0]) == InitializationPhase:
        x_train = args[0][0].initial_samples_x
        y_train = args[0][0].initial_samples_y[:,output_type]
    else:
        raise ValueError
    
    return x_train, y_train

def _generate_pysoarcdataset(output_type: int, *args):
    if output_type not in [0,1]:
        raise ValueError
    
    
    if type(args[0][0]) == InitializationPhase:
        x_train = args[0][0].initial_samples_x
        y_train = args[0][0].initial_samples_y[:,output_type]
    else:
        raise ValueError

    for arg in args[0][1:]:
        if type(arg) == GlobalPhase:
            x_train = np.vstack((x_train, arg.restart_point_x))
            y_train = np.hstack((y_train, arg.restart_point_y[:,output_type]))
        elif type(arg) == LocalPhase:
            x_train = np.vstack((x_train, arg.local_phase_x))
            y_train = np.hstack((y_train, arg.local_phase_y[:,output_type]))
        elif type(arg) == LocalBest:
            x_train = np.vstack((x_train, arg.local_best_x))
            y_train = np.hstack((y_train, arg.local_best_y[:,output_type]))
    
    return x_train, y_train

starting_seed = 1234565
evals = []
fals = 0
for rep in range(1):
    seed = starting_seed+rep
    print(seed)
    # with open(f"/home/daittan/RA_Work/pysoar-c/demos/pysoarc_res/pysoarc_max500_rep_50_seed{seed}.pickle", "rb") as f:
    with open(f"/home/daittan/RA_Work/pysoar-c/demos/pysoarc_max500_rep_50_seed{seed}.pickle", "rb") as f:
        point_history = pickle.load(f)
    # print(point_history)
    x,y = _generate_pysoarcdataset(0, point_history)

    
    evals.append(x.shape[0])
    if x.shape[0] < 500:
        fals += 1
    
    fig = plt.figure()
            
    ax = fig.add_subplot(111)


    point_streak = []
    p1_handles = []
    p2_handles = []
    p3_handles = []
    p4_handles = []
    
    for point_types in point_history:
        
        if type(point_types) is InitializationPhase:
            p1, = ax.plot(
                point_types.initial_samples_x[:,0], 
                point_types.initial_samples_x[:,1],
                'b*', 
                markersize = 7, 
                label = "Initial Points"
            )
            
            p1_handles.append(p1)
        elif type(point_types) is GlobalPhase:
            p2, = ax.plot(
                point_types.restart_point_x[:,0], 
                point_types.restart_point_x[:,1],
                'k*', 
                markersize = 7, 
                label = "Global Points"
            )
            p2_handles.append(p2)
        elif type(point_types) == LocalPhase:
            p3, = ax.plot(
                point_types.local_phase_x[:,0], 
                point_types.local_phase_x[:,1],
                'b.', 
                markersize = 7, 
                label = "Local Points"
            )
            p3_handles.append(p3)
        elif type(point_types) == LocalBest:
            p4, = ax.plot(
                point_types.local_best_x[:,0], 
                point_types.local_best_x[:,1],
                'k.', 
                markersize = 7, 
                label = "Local Points"
            )
            p4_handles.append(p4)
    handles = []
    handle_names = []
    if len(p1_handles) > 0:
        handles.append(p1_handles[0])
        handle_names.append("Initial Points")
    if len(p2_handles) > 0:
        handles.append(p2_handles[0])
        handle_names.append("Global Points")
    if len(p3_handles) > 0:
        handles.append(p3_handles[0])
        handle_names.append("Local Points")
    if len(p4_handles) > 0:
        handles.append(p4_handles[0])
        handle_names.append("Local Best")
    ax.legend(handles, 
                handle_names)
    plt.savefig(f'/home/daittan/RA_Work/pysoar-c/demos/pysoarc_samples_seed{seed}.pdf') 

print(evals)
print(f"{fals}, {np.mean(evals)}, {np.median(evals)}")