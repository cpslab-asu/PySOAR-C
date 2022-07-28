from math import pi

from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from numpy import array, deg2rad, float32, float64
from staliro.models import Blackbox

INITIAL_ALT = 2330


@Blackbox
def f16_blackbox(X, T, _):
    power = 9
    alpha = deg2rad(2.1215)
    beta = 0
    alt = INITIAL_ALT
    vel = 540
    phi, theta, psi = X

    init_cond = [vel, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    step = 1 / 30
    autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

    result = run_f16_sim(init_cond, max(T), autopilot, step, extended_states=True)
    trajectories = result["states"][:, 11:12].T.astype(float64)
    timestamps = array(result["times"], dtype=(float32))

    return trajectories, timestamps
