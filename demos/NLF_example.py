import pickle
import numpy as np


from pysoar import PySOAR
from pysoar.gprInterface import InternalGPR

def test_function(X):
    a = (100 * (X[1] - X[0]**2)**2 + ((X[0]-1)**2))
    b = (100 * (X[2] - X[1]**2)**2 + ((X[1]-1)**2))
    c = (100 * (X[3] - X[2]**2)**2 + ((X[2]-1)**2))
    d = (100 * (X[4] - X[3]**2)**2 + ((X[3]-1)**2))
    return a + b + c + d
    
    
# Max number of simulations
MAX_BUDGET = 500

# Total Number of replication
NUMBER_OF_MACRO_REPLICATIONS = 10

# Trust Region Local Search Budget
trs_max_budget = 50

for replication in range(NUMBER_OF_MACRO_REPLICATIONS):
    point_history, modes, sim_times = PySOAR(
        n_0= 100, # Initial Sampling budget 
        nSamples = MAX_BUDGET, # Max number of function evaluations
        trs_max_budget = trs_max_budget, #Trust Region Local Search Budget
        max_loc_iter=30, # Maximum nuyumber of local iterations for a restart point
        inpRanges = np.array([[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.]]), # Input Bounds
        alpha_lvl_set = .05, # Used in EI Calculation - Refer to EI^c in the paper
        eta0 = .25, # Eta_0 parameter defined in the paper - used for deciding wether to expand or contract trust region
        eta1 = .75, # Eta_1 parameter defined in the paper - used for deciding wether to expand or contract trust region
        delta = .75, # Used for contraction of the trust region
        gamma = 1.25, # Helps decide bounds on the trust region
        eps_tr = 0.001, # epsilon value - quits local loop when trust region (TR) less than epsilon*area(TR)
        prob = test_function, # Test Function
        gpr_model = InternalGPR(), # Gaussian Process Regressor
        seed = 1234567, # Seed
        local_search= "gp_local_search", # Currently we only support "gp_local_search"
        behavior = "Minimization" # "Minimization" minimizes the test fucntion. "Falsification" quits when a falsifying input is found.
        )

    with open(f"Pysor_res_{replication}", "wb") as f:
        pickle.dump((point_history, modes, sim_times), f)


point_history = np.array(point_history, dtype = object)
distinction = np.array(point_history[:,2])
modes = np.array(point_history[:,3])
regions = np.array(point_history[:,4])


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
fig = plt.figure()
        
ax = fig.add_subplot(111)



for i in np.unique(distinction):

    subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]

    if subset_1.shape[0] != 0:
        all_points = np.stack(subset_1[:,1])
        ax.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
        for itera,p in enumerate(subset_1):
            points = p[1]
            region = p[4]
            x = region[0,0]
            y = region[1,0]
            w = region[0,1] - region[0,0]
            h = region[1,1] - region[1,0]

            if itera == 0:
                ax.plot(points[0], points[1], 'b*', markersize = 7)
                ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
            else:
                ax.plot(points[0], points[1], 'b.', markersize = 4)
                ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))  


for i in np.unique(distinction):
    subset_1 = point_history[distinction == i, :]

    if subset_1.shape[0] != 0:
        all_points = np.stack(subset_1[:,1])
        ax.plot(all_points[:,0], all_points[:,1], "b.", markersize = 4)
        ax.plot(all_points[0,0], all_points[0,1], "b*", markersize = 7)

plt.show()
plt.savefig(f"test.png", bbox_inches='tight', dpi = 500, transparent = False)