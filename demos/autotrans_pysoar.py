import pickle
import numpy as np

from pysoar import run_pysoar
from pysoar.gprInterface import InternalGPR

from autotrans import AutotransModel
from staliro.options import SignalOptions, Options
from staliro.specifications import RTAMTDense
from staliro.signals import piecewise_constant    
from staliro.staliro import staliro
    
# Max number of simulations
MAX_BUDGET = 200

# Total Number of replication
NUMBER_OF_MACRO_REPLICATIONS = 1

# Trust Region Local Search Budget
trs_max_budget = 50


optimizer = run_pysoar(
    n_0= 10, # Initial Sampling budget 
    trs_max_budget = trs_max_budget, #Trust Region Local Search Budget
    max_loc_iter=5, # Maximum nuyumber of local iterations for a restart point
    alpha_lvl_set = .05, # Used in EI Calculation - Refer to EI^c in the paper
    eta0 = .25, # Eta_0 parameter defined in the paper - used for deciding wether to expand or contract trust region
    eta1 = .75, # Eta_1 parameter defined in the paper - used for deciding wether to expand or contract trust region
    delta = .75, # Used for contraction of the trust region
    gamma = 1.25, # Helps decide bounds on the trust region
    eps_tr = 0.001, # epsilon value - quits local loop when trust region (TR) less than epsilon*area(TR)
    gpr_model = InternalGPR(), # Gaussian Process Regressor
    local_search= "gp_local_search", # Currently we only support "gp_local_search"
    behavior = "Minimization" # "Minimization" minimizes the test fucntion. "Falsification" quits when a falsifying input is found.
)
    
AT1_phi = "G[0, 20] (speed <= 120)"
specification = RTAMTDense(AT1_phi, {"speed": 0})

signals = [
    SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
    SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
]

options = Options(runs=1, iterations=MAX_BUDGET, interval=(0, 50),  signals=signals)

model = AutotransModel()
result = staliro(model, specification, optimizer, options)