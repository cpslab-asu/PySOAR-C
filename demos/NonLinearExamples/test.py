import pickle
from soar.optimizer import _generate_dataset
import numpy as np
import matplotlib.pyplot as plt

starting_seed = 1234565

with open(f"/home/local/ASURITE/tkhandai/Research_Work/pysoarc-cleanup/PySOAR-C/demos/NonLinearExamples/LS-exp/ackley/dim_2/seed_1923.pkl", "rb") as f:
    algo_journey = pickle.load(f)

local_sample_x, local_samples_y = _generate_dataset(0, algo_journey)


plt.plot(np.minimum.accumulate(local_samples_y), "-")
plt.plot(local_samples_y, ".")
plt.show()