from __future__ import annotations

import enum
import logging
import math
import numbers
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import gpytorch
import numpy as np
import torch
from attrs import frozen
from gpytorch.settings import fast_pred_var
from numpy.typing import NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.result import Result as PyMmooResult
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from scipy.optimize import minimize as minimize_scipy
from scipy.stats import norm
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .gpr import GPR, GaussianProcessRegressor

# from .regions import local_best_ei
from .sampling import lhs_sampling, uniform_sampling

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class SoarOptions:
    n_0: int
    trs_max_budget: int
    max_loc_iter: int
    alpha_lvl_set: float
    eta0: float
    eta1: float
    delta: float
    gamma: float
    eps_tr: float
    min_tr_size: float
    tr_threshold: float
    gpr_model: GaussianProcessRegressor
    behavior: Behavior
    local_search: str

# def local_best_ei(
#     pred_sample_x,
#     pred_sample_y,
#     tf_wrapper,
#     test_fn,
#     tf_dim,
#     trust_region,
#     xTrain_local,
#     yTrain_local,
#     behavior: Behavior,
#     gpr_model,
#     rng,
#     mu, sigma
# ) -> tuple[NDArray, NDArray, np.float64]:
#     # Fit Gaussian Process Meta Model Locally
#     gpr = GPR(deepcopy(gpr_model))
#     gpr.fit(xTrain_local, yTrain_local)

#     EI_obj = lambda x: -1 * EIcalc_kd(yTrain_local, x, gpr)
#     lower_bound_theta = np.ndarray.flatten(trust_region[:, 0])
#     upper_bound_theta = np.ndarray.flatten(trust_region[:, 1])

#     random_samples = uniform_sampling(5000, trust_region, tf_dim, rng)
#     min_bo_val = EI_obj(random_samples)

#     min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])[0]
#     min_bo_val = np.min(min_bo_val)

#     for _ in range(5):
#         new_params = minimize_scipy(
#             EI_obj,
#             bounds=list(zip(lower_bound_theta, upper_bound_theta)),
#             x0=min_bo,
#         )

#         if not new_params.success:
#             continue

#         if min_bo is None or EI_obj(new_params.x) < min_bo_val:
#             min_bo = new_params.x
#             min_bo_val = EI_obj(min_bo)
#     new_params = minimize_scipy(
#         EI_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
#     )
#     xk = np.array([np.array(new_params.x)])

#     rob = tf_wrapper(xk, test_fn, behavior)

#     if rob is None and behavior is Behavior.COVERAGE:
#         rho = [None]
#     else:
#         rho = (pred_sample_y[0][0] - rob[0][0]) / (
#             gpr.predict(pred_sample_x)[0] - gpr.predict(xk)[0] + 1e-6
#         )

#     rho_ret = rho[0]
#     return xk, rob, rho_ret


# def local_best_ei(
#     pred_sample_x,
#     pred_sample_y,
#     tf_wrapper,
#     test_fn,
#     tf_dim,
#     trust_region,
#     xTrain_local,
#     yTrain_local,
#     behavior: Behavior,
#     gpr_model,
#     rng,
#     mu,
#     sigma,
#     batch_size = 1,
#     max_cholesky_size = 2000,
#     hypers = {}
# ) -> tuple[NDArray, NDArray, np.float64]:


#     # Convert to PyTorch tensors
    
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     dtype = torch.float64
#     # with gpytorch.settings.max_cholesky_size(max_cholesky_size):
#     X_torch = torch.tensor(xTrain_local, dtype=dtype, device=device)
#     y_torch = torch.tensor(yTrain_local.reshape(-1,1), dtype=dtype, device=device)

#     likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
#     covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
#         MaternKernel(
#             nu=2.5, ard_num_dims=tf_dim, lengthscale_constraint=Interval(0.005, 4.0)
#         )
#     )
#     model = SingleTaskGP(
#         X_torch, y_torch, covar_module=covar_module, likelihood=likelihood
#     )
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)

#     with gpytorch.settings.max_cholesky_size(max_cholesky_size):
#         # Fit the model
#         fit_gpytorch_mll(mll)
#     trust_region_torch = torch.from_numpy(trust_region).to(device)
#     n_candidates = 5000
#     # dim = X_torch.shape[-1]
#     sobol = SobolEngine(tf_dim, scramble=True)
#     pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
#     pert = trust_region_torch[:,0] + (trust_region_torch[:,1] - trust_region_torch[:,0]) * pert

#     # Create a perturbation mask
#     prob_perturb = min(20.0 / tf_dim, 1.0)
#     mask = torch.rand(n_candidates, tf_dim, dtype=dtype, device=device) <= prob_perturb
#     ind = torch.where(mask.sum(dim=1) == 0)[0]
#     mask[ind, torch.randint(0, tf_dim - 1, size=(len(ind),), device=device)] = 1

#     # Create candidate points from the perturbations and the mask
#     X_cand = torch.from_numpy(pred_sample_x).to(device).expand(n_candidates, tf_dim).clone()
#     X_cand[mask] = pert[mask]

#     # Sample on the candidate points
#     thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
#     with torch.no_grad():  # We don't need gradients when using TS
#         X_next = thompson_sampling(X_cand, num_samples=batch_size)

    

#     # Evaluate candidate
#     rob = tf_wrapper(X_next.cpu().numpy(), test_fn, behavior)
    
#     model.eval()
#     likelihood.eval()

#     # Predict at X_next
    
    
#     # Compute improvement ratio
#     if rob is None and behavior is Behavior.COVERAGE:
#         rho = None
#     else:
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             rob_cand = likelihood(model(X_next))
#             mu_cand =  rob_cand.mean.cpu().numpy() * sigma + mu
#             # Predict at candidate point
            
#             pred = likelihood(model(torch.tensor(pred_sample_x).to(device)))
#             mu_pred =  pred.mean.cpu().numpy() * sigma + mu
#             # Predict at candidate point


#         # Calculate actual and predicted improvements
#         actual_improve = pred_sample_y[0][0] - rob[:,0]

#         predicted_improve = mu_pred - mu_cand
#         rho = np.mean(actual_improve / (predicted_improve + 1e-10))
#     # del X_torch, y_torch, gp
#     return X_next.cpu().numpy(), rob, rho

def local_best_ei(
    pred_sample_x,
    pred_sample_y,
    tf_wrapper,
    test_fn,
    tf_dim,
    trust_region,
    xTrain_local,
    yTrain_local,
    behavior: Behavior,
    gp,
    rng,
    mu,
    sigma,
    hypers,
    batch_size = 1,
    max_cholesky_size = 2000,
) -> tuple[NDArray, NDArray, np.float64]:


    # Convert to PyTorch tensors
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float64
    with gpytorch.settings.max_cholesky_size(2000):
        X_torch = torch.tensor(xTrain_local, dtype=dtype, device=device)
        y_torch = torch.tensor(yTrain_local, dtype=dtype, device=device)

        # Train GP using TuRBO's method
        gp = train_gp(
            train_x=X_torch,
            train_y=y_torch,
            use_ard=True,
            num_steps=50,
            hypers=hypers
        )
        
    
    # Generate candidate points using Sobol sequence
    n_cand = 5000
    sobol = SobolEngine(tf_dim, scramble=True, seed=int(rng.integers(1e6)))
    pert = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
    pert = trust_region[:,0] + (trust_region[:,1] - trust_region[:,0]) * pert
    # X_cand_norm = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
    prob_perturb = min(20.0 / tf_dim, 1.0)
    mask = np.random.rand(n_cand, tf_dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    mask[ind, np.random.randint(0, tf_dim - 1, size=len(ind))] = 1

    # Create candidate points
    X_cand = pred_sample_x.copy() * np.ones((n_cand, tf_dim))
    X_cand[mask] = pert[mask]

    if len(X_cand) < 1024:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = device, dtype

    # We may have to move the GP to a new device
    gp = gp.to(dtype=dtype, device=device)

    # We use Lanczos for sampling if we have enough data
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(max_cholesky_size):
        X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
        y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([batch_size])).t().cpu().detach().numpy()
# # Thompson sampling
    # with torch.no_grad(), fast_pred_var():
    #     X_cand_torch = torch.tensor(X_cand_norm, dtype=dtype, device=device)
    #     posterior = gp.likelihood(gp(X_cand_torch))
    #     y_cand = posterior.sample(torch.Size([batch_size])).cpu().numpy()

    X_next = np.ones((batch_size, tf_dim))
    for i in range(batch_size):
        indbest = np.argmin(y_cand[:,i])
        X_next[i,:] = deepcopy(X_cand[indbest, :])
        y_cand[indbest, :] = np.inf
    # Select best candidate
    # best_cand_idx = np.argmin(y_cand)
    # x_cand_norm = X_cand_norm[best_cand_idx][None, :]
    # x_cand_orig = from_unit_cube(X_next, lb_tr, ub_tr)

    # Evaluate candidate
    rob = tf_wrapper(X_next, test_fn, behavior)
    
    # Compute improvement ratio
    if rob is None and behavior is Behavior.COVERAGE:
        rho = None
    else:
        with torch.no_grad(), fast_pred_var():
            # Predict at candidate point
            cand_torch = torch.tensor(X_next, dtype=dtype, device=device)
            mu_cand = gp(cand_torch).mean.cpu().numpy() * sigma + mu
            
            # Predict at reference point
            # pred_x_norm = to_unit_cube(pred_sample_x, lb_tr, ub_tr)
            pred_x_torch = torch.tensor(pred_sample_x, dtype=dtype, device=device)
            mu_pred = gp(pred_x_torch).mean.cpu().numpy()[0] * sigma + mu

        # Calculate actual and predicted improvements
        actual_improve = pred_sample_y[0][0] - rob[:,0]

        predicted_improve = mu_pred - mu_cand
        rho = np.mean(actual_improve / (predicted_improve + 1e-10))
    del X_torch, y_torch, gp
    return X_next, rob, rho, hypers


class Behavior(enum.IntEnum):
    """Behavior when falsifying case for system is encountered.

    Attributes:
     ----------
         FALSIFICATION: Stop searching when the first falsifying case is encountered
         MINIMIZATION: Continue searching after encountering a falsifying case until iteration
                       budget is exhausted
    """

    FALSIFICATION = enum.auto()
    MINIMIZATION = enum.auto()
    COVERAGE = enum.auto()


def _surrogate(gpr_model: GPR, x_train: NDArray):
    """_surrogate Model function

    Attributes:
    ----------
        model: Gaussian process model
        X: Input points

    Returns:
    ---------
        Predicted values of points using gaussian process model
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float64
    with torch.no_grad(), fast_pred_var():
            # Predict at candidate point
        cand_torch = torch.tensor(x_train, dtype=dtype, device=device)
        posterior = gpr_model(cand_torch)
        mu_cand = posterior.mean.cpu().numpy()
        mu_std = posterior.stddev.cpu().numpy()
    return mu_cand, mu_std
    # return gpr_model.predict(x_train)


def EIcalc_kd(y_train: NDArray, sample: NDArray, gpr_model) -> NDArray:
    """Acquisition Model: Expected Improvement

    Attributes:
    ----------
        y_train: corresponding robustness values. Expected NDArray of shape (m,) where m is the number of output samples
        sample: Sample(s) whose EI is to be calculated
        gpr_model: GPR model
        sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

    Returns:
    ---------
        EI of samples
    """
    curr_best = np.min(y_train)
    if len(sample.shape) == 2:
        mu, std = _surrogate(gpr_model, sample)
        ei_list = []
        for mu_iter, std_iter in zip(mu, std):
            pred_var = std_iter
            if pred_var > 0:
                var_1 = curr_best - mu_iter
                var_2 = var_1 / pred_var

                ei = (var_1 * norm.cdf(var_2)) + (pred_var * norm.pdf(var_2))
            else:
                ei = 0.0

            ei_list.append(ei)
        return_ei = np.array(ei_list)
    elif len(sample.shape) == 1:

        mu, std = _surrogate(gpr_model, sample.reshape(1, -1))
        pred_var = std[0]
        if pred_var > 0:
            var_1 = curr_best - mu[0]
            var_2 = var_1 / pred_var

            ei = (var_1 * norm.cdf(var_2)) + (pred_var * norm.pdf(var_2))
        else:
            ei = 0.0
        return_ei = ei
    return return_ei


def CrowdingDist_kd(x_0: NDArray, x: NDArray) -> float:
    """COmputes the crowding distance between a x and set of points x_0

    Attributes:
    ----------
        x_0 : Single point of Shape (1,d), where d is the dimensionality
        x: Exisiting points to caluclate crowding distance from. Expected shape (n,d), where n is the dimensionality and d is the dimensionality

    Returns:
    --------
        float: Crowding distance of x_0 from x
    """
    # return np.sum(np.sqrt(np.sum((x_0 - x) ** 2, 1)))
    # return np.sum(np.min(np.abs(x_0 - x),0))
    return np.min(np.sqrt(np.sum((x_0 - x) ** 2, axis=1)))
    


def ei_cd(
    samples,
    x_train: NDArray,
    y_train: NDArray,
    gpr: GPR,
    alpha_lvl_set: float,
    EI_star: float,
):

    if len(samples.shape) == 1:
        c = EIcalc_kd(y_train, samples, gpr) - (alpha_lvl_set * (EI_star))
        if c >= 0:
            ret = -1 * CrowdingDist_kd(samples, x_train)
        else:
            ret = 0
    elif len(samples.shape) > 1:
        ret = []
        # print(x)
        for sample in samples:
            c = EIcalc_kd(y_train, sample, gpr) - (alpha_lvl_set * (EI_star))
            # print(c[i])
            if c >= 0:
                ret.append(-1 * CrowdingDist_kd(sample, x_train))
            else:
                ret.append(0)
        ret = np.array(ret)
    return ret


def pointsInTR(
    samples_in: NDArray, samples_out: NDArray, subregion: NDArray
) -> tuple[NDArray, NDArray]:
    """

    Attributes:
    ----------
        samples_in: Samples from Training set. Expected shape is (n,d) where n is the number of samples and d is the dimensionality
        samples_out: Evaluated values of samples from Training set. Expected shape is (n,) where n is the number of samples
        region_support: Min and Max of all dimensions

    Returns:
    --------
        list: Divided samples and coreesponding robustness values
    """
    region_samples = []
    corresponding_robustenss = []
    if samples_in.shape[0] == samples_out.shape[0] and samples_out.shape[0] != 0:

        boolArray = []
        for dimension in range(len(subregion)):
            subArray = samples_in[:, dimension]
            logical_subArray = np.logical_and(
                subArray >= subregion[dimension, 0], subArray <= subregion[dimension, 1]
            )
            boolArray.append(np.squeeze(logical_subArray))
        corresponding_robustenss = samples_out[(np.all(boolArray, axis=0))]
        region_samples = samples_in[(np.all(boolArray, axis=0)), :]
    else:

        corresponding_robustenss = np.array([])
        region_samples = np.array([[]])

    return region_samples, corresponding_robustenss


@dataclass(frozen=True, slots=True)
class LocalBest:
    """
    Represents a phase in a local optimization phase.

    Attributes:
    ----------
    local_best_x : A restart point in the search space, typically represented as a
    1-dimensional vector of size `d`, where `d` is the dimensionality
    of the problem.

    local_best_y : A 1x2 vector that contains evaluations of the global and local
    test functions at the restart point.
        - The first column corresponds to the evaluation of the global test function.
        - The second column corresponds to the evaluation of the local test function.
    """

    local_best_x: NDArray
    local_best_y: NDArray

    def __post_init__(self):
        if type(self.local_best_x) != np.ndarray:
            raise TypeError("local_best_x must be an NDArray")
        if type(self.local_best_y) != np.ndarray:
            raise TypeError("local_best_y must be an NDArray")
        if len(self.local_best_x.shape) != 2:
            raise ValueError("local_best_x must be a 2-dimensional vector.")
        if self.local_best_y.shape[1] !=  2:
            raise ValueError("local_best_y must be a nx2 matrix.")


@dataclass(frozen=True, slots=True)
class LocalPhase:
    """
    Represents the data collected during the local sampling phase in an optimization or analysis process.

    Attributes:
    ----------
        region_support (NDArray): A NumPy array of shape (d, 2), where `d` is the dimensionality of the problem space.
            - Each row of `region_support` defines the lower and upper bounds for a dimension in the local sampling region.

        local_phase_x (NDArray): A NumPy array of shape (n, d), where `n` is the number of samples and `d` is the dimensionality.
            - Represents the input points sampled during the local phase.

        local_phase_y (NDArray): A NumPy array of shape (n, 2), where `n` is the number of samples.
            - The first column contains the evaluation of the global test function for the corresponding sample in `local_phase_x`.
            - The second column contains the evaluation of the local test function for the same sample.
    """

    region_support: NDArray
    local_phase_x: NDArray
    local_phase_y: NDArray

    def __post_init__(self):
        if type(self.local_phase_x) != np.ndarray:
            raise TypeError("local_phase_x must be an NDArray")
        if type(self.local_phase_y) != np.ndarray:
            raise TypeError("local_phase_y must be an NDArray")
        if type(self.region_support) != np.ndarray:
            raise TypeError("region_support must be an NDArray")
        if len(self.region_support.shape) != 2 or self.region_support.shape[1] != 2:
            raise ValueError("region_support must be a (d, 2) matrix.")
        if len(self.local_phase_x.shape) != 2:
            raise ValueError("local_phase_x must be an (n, d) matrix.")
        if len(self.local_phase_y.shape) != 2 or self.local_phase_y.shape[1] != 2:
            raise ValueError("local_phase_y must be an (n, 2) matrix.")


@dataclass(frozen=True, slots=True)
class GlobalPhase:
    """
    Represents a phase in a global optimization or testing process.

    Attributes:
    ----------
    restart_point_x : Any
        A restart point in the search space, typically represented as a
        1-dimensional vector of size `d`, where `d` is the dimensionality
        of the problem.

    restart_point_y : Any
        A 1x2 vector that contains evaluations of the global and local
        test functions at the restart point.
        - The first column corresponds to the evaluation of the global test function.
        - The second column corresponds to the evaluation of the local test function.
    """

    restart_point_x: NDArray[np.double]
    restart_point_y: NDArray[np.double]

    def __post_init__(self):
        if type(self.restart_point_x) != np.ndarray:
            raise TypeError("restart_point_x must be an NDArray")
        if type(self.restart_point_y) != np.ndarray:
            raise TypeError("restart_point_y must be an NDArray")
        if len(self.restart_point_x.shape) != 2:
            raise ValueError(
                f"restart_point_x must be a 2-dimensional vector. Received {self.restart_point_x}"
            )
        if self.restart_point_y.shape != (1, 2):
            raise ValueError("restart_point_y must be a 1x2 matrix.")


@dataclass(frozen=True, slots=True)
class InitializationPhase:
    """Optimizer startup phase.

    This class represents the initial phase of an optimization process, where samples are generated
    and evaluated. The attributes of this class hold the initial set of input samples and their associated
    cost values that were evaluated during the optimizer's startup phase.

    Attributes:
    ----------
        initial_samples_x: A DxN matrix of floating-point numbers, where D is the
                                                dimensionality of each sample and N is the number of
                                                samples. Each column represents a sample in the input
                                                space.
        initial_samples_y: A 2-dimensional array (vector) of floating-point numbers
                                                 representing the cost or objective function values of
                                                 each corresponding sample in `initial_samples_x`.
    """

    initial_samples_x: NDArray[np.double]
    initial_samples_y: NDArray[np.double]

    def __post_init__(self):
        if type(self.initial_samples_x) != np.ndarray:
            raise TypeError("initial_samples_x must be an NDArray")
        if type(self.initial_samples_y) != np.ndarray:
            raise TypeError("initial_samples_y must be an NDArray")
        if len(self.initial_samples_x.shape) != 2:
            raise ValueError("initial_samples_x must be a 2D array.")
        if len(self.initial_samples_y.shape) != 2:
            raise ValueError("initial_samples_y must be a 2D array.")
        if self.initial_samples_x.shape[0] != self.initial_samples_y.shape[0]:
            raise ValueError(
                "The number of samples in initial_samples_x must match the length of initial_samples_y."
            )


def _generate_dataset(output_type: int, *args):
    """
    Generate a dataset for training based on input phases and the desired output type.

    This function combines data from different phases (InitializationPhase, GlobalPhase,
    LocalPhase, and LocalBest) into a single training dataset. The dataset consists of input
    features `x_train` and corresponding output labels `y_train`.

    Attributes:
    ----------
    output_type : int
        Specifies the column of output labels to be used for training. Must be 0 or 1.
        If it is 0, we take the 0th column cost function in evaluations.
        If it is 0, we take the 1st column cost function in evaluations.
    *args : tuple
        A variable-length argument list containing phase objects. The first argument
        must be an `InitializationPhase` object. Subsequent arguments can be any
        combination of `GlobalPhase`, `LocalPhase`, or `LocalBest` objects.

    Returns:
    -------
    tuple
        A tuple `(x_train, y_train)` where:
        - `x_train` (ndarray): Combined input features from all phases.
        - `y_train` (ndarray): Combined output labels corresponding to the `output_type`
          column from all phases.

    Raises:
    ------
    ValueError
        - If `output_type` is not 0 or 1.
        - If the first argument is not an `InitializationPhase` object.
        - If an unrecognized phase type is provided in `args`.

    Notes:
    -----
    - The function assumes that the objects provided in `args` contain the following attributes:
      - `InitializationPhase`: `initial_samples_x`, `initial_samples_y`
      - `GlobalPhase`: `restart_point_x`, `restart_point_y`
      - `LocalPhase`: `local_phase_x`, `local_phase_y`
      - `LocalBest`: `local_best_x`, `local_best_y`
    - The `output_type` determines which column of the `y` arrays is included in `y_train`.

    """

    if output_type not in [0, 1,2]:
        raise ValueError
    if output_type in [0,1]:
        if type(args[0][0]) == InitializationPhase:
            x_train = args[0][0].initial_samples_x
            y_train = args[0][0].initial_samples_y[:, output_type]
        else:
            raise ValueError

        for arg in args[0][1:]:
            if type(arg) == GlobalPhase:
                x_train = np.vstack((x_train, arg.restart_point_x))
                y_train = np.hstack((y_train, arg.restart_point_y[:, output_type]))
            elif type(arg) == LocalPhase:
                x_train = np.vstack((x_train, arg.local_phase_x))
                y_train = np.hstack((y_train, arg.local_phase_y[:, output_type]))
            elif type(arg) == LocalBest:
                x_train = np.vstack((x_train, arg.local_best_x))
                y_train = np.hstack((y_train, arg.local_best_y[:, output_type]))
    else:
        if type(args[0][0]) == InitializationPhase:
            x_train = args[0][0].initial_samples_x
            y_train = args[0][0].initial_samples_y[:, :]
        else:
            raise ValueError

        for arg in args[0][1:]:
            if type(arg) == GlobalPhase:
                x_train = np.vstack((x_train, arg.restart_point_x))
                y_train = np.vstack((y_train, arg.restart_point_y[:, :]))
            elif type(arg) == LocalPhase:
                x_train = np.vstack((x_train, arg.local_phase_x))
                y_train = np.vstack((y_train, arg.local_phase_y[:, :]))
            elif type(arg) == LocalBest:
                x_train = np.vstack((x_train, arg.local_best_x))
                y_train = np.vstack((y_train, arg.local_best_y[:, :]))
    return x_train, y_train


def _is_falsification(evaluation: Optional[Union[NDArray, None]]) -> bool:
    """
    Determines whether a given evaluation result indicates a falsification condition.

    Attributes:
    ----------
    evaluation : Optional[Union[NDArray, None]]
        An array-like object (typically a NumPy array) containing evaluation results
        or `None`. The evaluation is expected to have at least two elements if not `None`.

    Returns:
    -------
    bool
        - `True` if the evaluation satisfies any of the following falsification conditions:
            1. The evaluation is `None`.
            2. Any value in the evaluation is `NaN`.
            3. The first element of the evaluation is `<= 0` and the second element is `< 0`.
        - `False` otherwise.
    """
    # Check if the evaluation is None or contains any NaN values.
    if evaluation is None or any(np.isnan(evaluation)):
        return True

    # Check if the first element is <= 0 and the second element is < 0.
    return evaluation[0] <= 0 and evaluation[1] < 0


class Fn:
    """
    A class wrapper for a function to track its call count and log input arguments
    alongside the computed output.

    Attributes:
    ----------
    func : callable
        The function to be wrapped and tracked.
    count : int
        The number of times the wrapped function has been called.

    Methods:
    -------
    __call__(*arg):
        Executes the wrapped function with the provided arguments, increments the call count,
        and prints the call count, the first argument, and the function's output.
    """

    def __init__(self, func, search_space = None):
        """
        Initializes the Fn object with a function to be tracked.

        Parameters:
        ----------
        func : callable
            The function to be wrapped and tracked.
        """
        self.func = func
        self.count = 0
        self.search_space = search_space

    def __call__(self, *arg):
        """
        Executes the wrapped function, increments the call count, and logs the details.

        Parameters:
        ----------
        *arg : tuple
            The arguments to pass to the wrapped function.

        Returns:
        -------
        Any
            The result of executing the wrapped function with the given arguments.
        """
        self.count = self.count + 1  # Increment the call count.

        
        # Call the wrapped function with the given arguments.
        dist = self.func(*arg)
        # if not isinstance(dist, tuple) or len(dist) != 2:
        #     # print(self.count, arg[0], dist)
        #     return dist, dist
        # # # Print the call count, the first argument, and the output.
        # # print(self.count, arg[0], dist)

        # return dist
        if isinstance(dist, numbers.Number):
            logger.debug(f"{self.count}, {arg[0]}, {dist}")
            return dist, dist
        elif isinstance(dist, tuple) and len(dist) == 2 and all(isinstance(d, numbers.Number) for d in dist):
            logger.debug(f"{self.count}, {arg[0]}, {dist}")
            return dist
        else:
            raise ValueError("Function must return either a numeric value or a tuple of two numeric values.")


def _evaluate_samples(
    samples: NDArray[np.double],
    fn: Fn,
    behavior: Behavior,
) -> NDArray:
    """Evaluate samples into their corresponding cost values.

    If the behavior is to COVERAGE or FALSIFICATION then the function will terminate when the first
    negative cost value is found.

    Attributes:
    ----------
        samples: The set of samples to evaluate as a MxN matrix where N is the number of samples
        fn: The cost function that evaluates a M-dim vector into a O-dim vector of cost values
        behavior: The behavior to use if a negative cost value is encountered

    Returns:
    ----------
        A OxN matrix of cost values where each row is the cost value of the corresponding sample.
    """

    evaluations = []
    
    orig_samples = from_unit_cube(samples, fn.search_space[:,0], fn.search_space[:,1])
    for sample in orig_samples:
        evaluation = np.array(fn(sample), dtype=np.double)
        evaluations.append(evaluation)

        if _is_falsification(evaluation) and behavior in (
            Behavior.FALSIFICATION,
            Behavior.COVERAGE,
        ):
            break

    return np.array(evaluations)


##### v9 ####### add user defined parameters to input, break once falsified
def soarc(
    n_0: int,
    n_samples: int,
    trs_max_budget: int,
    max_loc_iter: int,
    inp_ranges: NDArray,
    alpha_lvl_set: float,
    eta0: float,
    eta1: float,
    delta: float,
    gamma: float,
    eps_tr: float,
    min_tr_size: float,
    tr_threshold: float,
    test_fn: Callable[[NDArray],float|tuple[float, float]],
    gpr_model: GaussianProcessRegressor | None,
    seed: int,
    local_search: str,
    behavior: Behavior = Behavior.FALSIFICATION,
) -> list[InitializationPhase | GlobalPhase | LocalPhase | LocalBest]:
    # Notes:
    # Add none check for InternalGPR
    # Rename gprs for something meaningful
    # Stick to one documnetation
    # Check function/class names
    
    best_till_now = np.inf
    search_inp_ranges = np.array(inp_ranges)
    tf_dim = inp_ranges.shape[0]
    
    test_fn: Fn = Fn(test_fn, search_inp_ranges)
    inp_ranges = np.array([[0,1] for _ in range(tf_dim)])
    
    if search_inp_ranges.ndim != 2:
        raise ValueError("input ranges should be 2-dimensional")  # noqa: TRY003

    if search_inp_ranges.shape[1] != 2:
        raise ValueError("input range 2nd dimension should be equal to 2")  # noqa: TRY003

    rng = np.random.default_rng(seed)
    np.random.seed(seed + 1000)

    if n_0 > n_samples:
        raise ValueError(  # noqa: TRY003
            f"Received n_0({n_0}) > n_samples ({n_samples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (n_samples)"
        )
    
    # initial_samples = lhs_sampling(n_0, inp_ranges, tf_dim, rng)
    initial_samples = lhs_sampling(n_0, inp_ranges, tf_dim, rng)
    initial_sample_distances = _evaluate_samples(initial_samples, test_fn, behavior)
    
    best_till_now = min(np.min(initial_sample_distances), best_till_now)
    
    num_pts = min(initial_sample_distances.shape[0], initial_samples.shape[0])
    
    initial_points = InitializationPhase(
        initial_samples_x=initial_samples[:num_pts, :], initial_samples_y=initial_sample_distances[:num_pts, :]
    )

    algo_journey: list[InitializationPhase | GlobalPhase | LocalPhase | LocalBest] = [
        initial_points
    ]

    if any(_is_falsification(sd) for sd in initial_sample_distances) and (
        behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE
    ):
        return algo_journey
    hypers = None
    while test_fn.count < n_samples:
        logger.debug(f"{test_fn.count} Evaluations Completed")
        x_train, y_train = _generate_dataset(1, algo_journey)
        mu_y = np.median(y_train)
        sigma_y = np.std(y_train)
        sigma_y = 1.0 if sigma_y < 1e-6 else sigma_y
        y_train = (y_train - mu_y) / sigma_y
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float64
        with gpytorch.settings.max_cholesky_size(2000):
            X_torch = torch.tensor(x_train, dtype=dtype, device=device)
            y_torch = torch.tensor(y_train, dtype=dtype, device=device)

            # Train GP using TuRBO's method
            gp = train_gp(
                train_x=X_torch,
                train_y=y_torch,
                use_ard=True,
                num_steps=50,
                hypers={} if hypers is None else hypers
            )
        gp = gp.to(dtype=dtype, device=device)
        hypers = gp.state_dict()
        
        # print(f"{test_fn.count} Evaluations completed -> {x_train.shape}, {y_train.shape}")
        # gpr = GPR(deepcopy(gpr_model))
        # gpr.fit(x_train, y_train)

        lower_bound_theta = np.ndarray.flatten(inp_ranges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inp_ranges[:, 1])

        EI_obj= lambda x: -1 * EIcalc_kd(y_train, x, gp)
        

        random_samples = uniform_sampling(5000, inp_ranges, tf_dim, rng)
        min_bo_val = EI_obj(random_samples)

        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])[0]
        min_bo_val = np.min(min_bo_val)

        for _ in range(5):
            new_params = minimize_scipy(
                EI_obj,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            if not new_params.success:
                continue

            if min_bo is None or EI_obj(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = EI_obj(min_bo)
        new_params = minimize_scipy(
            EI_obj, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        ei_star = -1*EI_obj(new_params.x)
        
        problem = FunctionalProblem(
            n_var=inp_ranges.shape[0],
            objs=[
                # lambda x: -1 * EIcalc_kd(y_train, x, gpr),
                lambda x: -1*CrowdingDist_kd(x, x_train)
            ],
            constr_ieq=[lambda x: alpha_lvl_set*ei_star - EIcalc_kd(y_train, x, gp)],
            xl=lower_bound_theta,
            xu=upper_bound_theta,
        )
        
        # algorithm = NSGA2(
        #     pop_size=500,
        #     sampling=FloatRandomSampling(),
        #     crossover=SBX(prob=0.9, eta=15),
        #     mutation=PM(eta=20),
        #     eliminate_duplicates=True,
        # )

        ga_seed = rng.integers(low=1, high=100000, size=1)
        # ga_result: PyMmooResult = minimize(
        #     problem=problem,
        #     algorithm=algorithm,
        #     termination=("n_gen", 50),
        #     seed=ga_seed[0],
        #     verbose=False,
        # )
        algorithm = DE(
            pop_size=50,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=True
        )
        ga_result = minimize(problem,
               algorithm,
               ("n_gen", 200),
               seed=1,
               verbose=True)

        # minNegEIindex = np.argmin(ga_result.F[:, 0])
        # minNegEI = ga_result.F[minNegEIindex, 0]
        global_rp_x = ga_result.X if ga_result.X is not None else new_params.x
        del X_torch, y_torch, gp
        # best_crowd = math.inf

        # for k in range(ga_result.F.shape[0]):
        #     if ga_result.F[k, 0] <= minNegEI * (1 - alpha_lvl_set):
        #         if ga_result.F[k, 1] < best_crowd:
        #             # _logger.debug(
        #             #     f"{ga_result.F[k, 0]} <= {minNegEI * (1-alpha_lvl_set)} -> {ga_result.F[k,0] <= minNegEI * (1-alpha_lvl_set)} \\ {ga_result.F[k,1]} < {best_crowd} -> {ga_result.F[k,1] < best_crowd} \n{global_rp_x}\n*************************************************"
        #             # )
        #             best_crowd = ga_result.F[k, 1]
        #             global_rp_x = ga_result.X[k, :]
        global_rp_x = np.array([global_rp_x])
        global_rp_y = _evaluate_samples(global_rp_x, test_fn, behavior)
        
        # print(global_rp_x, global_rp_y)
        best_till_now = min(np.min(global_rp_y), best_till_now)
        logger.debug(f"Restart Phase: Best till Now -> {best_till_now}")
        algo_journey.append(
            GlobalPhase(restart_point_x=global_rp_x, restart_point_y=global_rp_y)
        )

        if _is_falsification(global_rp_y[0]) and (
            behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE
        ):
            # TODO
            return algo_journey

        local_sample_x, local_samples_y = _generate_dataset(0, algo_journey)

        TR_Bounds = np.vstack(
            [
                global_rp_x[0, :] - inp_ranges[:, 0],
                inp_ranges[:, 1] - global_rp_x[0, :],
                (inp_ranges[:, 1] - inp_ranges[:, 0]) / min_tr_size,
            ]
        ).flatten()

        TR_size = np.min(np.abs(TR_Bounds))

        trust_region = np.empty((inp_ranges.shape))
        for d in range(tf_dim):
            trust_region[d, 0] = max(global_rp_x[0, d] - TR_size, inp_ranges[d, 0])
            trust_region[d, 1] = min(global_rp_x[0, d] + TR_size, inp_ranges[d, 1])

        # print(trust_region)
        local_sample_x_subset, local_sample_y_subset = pointsInTR(
            local_sample_x, local_samples_y, trust_region
        )
        num_points_present = local_sample_x_subset.shape[0]

        local_counter = 0
        restart_point_x, restart_point_y = deepcopy(global_rp_x), deepcopy(global_rp_y)
        # print(trust_region)
        if local_search == "gp_local_search":

            while (
                local_counter < max_loc_iter
                and TR_size > eps_tr * np.min(inp_ranges[:, 1] - inp_ranges[:, 0])
                and test_fn.count + (max(trs_max_budget - num_points_present, 0) + 20) < n_samples
            ):
                print(trust_region)
                # print(f"Needed: {trs_max_budget}, present: {num_points_present}, More {num_samples_needed} points needed")
                if trs_max_budget - num_points_present > 0:
                    num_samples_needed = trs_max_budget - num_points_present

                    local_additional_x = lhs_sampling(
                        num_samples_needed, trust_region, tf_dim, rng
                    )
                    local_additional_y = _evaluate_samples(
                        local_additional_x, test_fn, behavior
                    )
                    best_till_now = min(np.min(local_additional_y), best_till_now)
                    logger.debug(f"In Local Phase: Best till Now -> {best_till_now}")
                    algo_journey.append(
                        LocalPhase(trust_region, local_additional_x, local_additional_y)
                    )

                    if any(_is_falsification(sd) for sd in local_additional_y) and (
                        behavior is Behavior.FALSIFICATION
                        or behavior is Behavior.COVERAGE
                    ):
                        return algo_journey

                x_train_hd, y_train_hd = _generate_dataset(0, algo_journey)
                # mu_y = np.median(y_train_hd)
                # sigma_y = np.std(y_train_hd)
                # sigma_y = 1.0 if sigma_y < 1e-6 else sigma_y
                # y_train_hd = (y_train_hd - mu_y) / sigma_y
                # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                # dtype = torch.float64
                
        
                local_sample_x_subset, local_sample_y_subset = pointsInTR(
                    x_train_hd, y_train_hd, trust_region
                )
                mu_y = np.median(local_sample_y_subset)
                sigma_y = np.std(local_sample_y_subset)
                sigma_y = 1.0 if sigma_y < 1e-6 else sigma_y
                local_sample_y_subset = (local_sample_y_subset - mu_y) / sigma_y
                # Fit Gaussian Process Meta Model Locally
                local_best_x, local_best_y, rho, hypers = local_best_ei(
                        restart_point_x,
                        restart_point_y,
                        _evaluate_samples,
                        test_fn,
                        tf_dim,
                        trust_region,
                        local_sample_x_subset,
                        local_sample_y_subset,
                        behavior,
                        None,
                        rng,
                        mu = mu_y,
                        sigma = sigma_y,
                        hypers = hypers,
                        batch_size=20
                    )
                # del X_torch, y_torch, gp
                # print(local_best_y)
                best_till_now = min(np.min(local_best_y), best_till_now)
                logger.debug(f"In Local BO phase Best till Now -> {best_till_now}")
                algo_journey.append(LocalBest(local_best_x, local_best_y))

                # if _is_falsification(local_best_y[0]) and (
                #     behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE
                # ):
                #     return algo_journey
                if any(_is_falsification(sd) for sd in local_best_y) and (
                    behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE
                ):
                    return algo_journey

                x_train_hd, y_train_hd = _generate_dataset(2, algo_journey)
                local_sample_x_subset, local_sample_y_subset = pointsInTR(
                    x_train_hd, y_train_hd, trust_region
                )
                # max_indicator = np.max(np.abs(local_best_x - restart_point_x)) / TR_size
                max_indicator = np.max(np.abs(local_best_x[np.argmin(local_best_y[:,1])].reshape(1,-1) - restart_point_x)) / TR_size
                test = rng.random()
                if max_indicator < test:
                    break

                # execute RC testing and TR control
                if rho < eta0:
                    TR_size *= delta
                    trust_region = np.empty((inp_ranges.shape))
                    for d in range(tf_dim):
                        trust_region[d, 0] = max(
                            restart_point_x[0, d] - TR_size, inp_ranges[d, 0]
                        )
                        trust_region[d, 1] = min(
                            restart_point_x[0, d] + TR_size, inp_ranges[d, 1]
                        )
                else:
                    if rho < eta1:
                        
                        # if local_best_y[0,0] < restart_point_y[0,0]:
                        idx = np.argmin(local_best_y[:,0])
                        restart_point_x = local_best_x[idx].reshape(1,-1)
                        restart_point_y = local_best_y[idx].reshape(1,-1)
                        # low pass of RC test
                        # idx = np.argmin(local_sample_y_subset[:,1])
                        # restart_point_x = local_sample_x_subset[idx].reshape(1,-1)
                        # restart_point_y = local_sample_y_subset[idx].reshape(1,-1)

                        valid_bound = np.array(
                            [
                                np.min(np.abs(restart_point_x[0, :] - inp_ranges[:, 0])),
                                np.min(np.abs(inp_ranges[:, 1] - restart_point_x[0, :])),
                                TR_size,
                            ]
                        ).flatten()
                        
                        TR_size = np.min(valid_bound)
                        
                            
                        trust_region = np.empty((inp_ranges.shape))

                        for d in range(tf_dim):
                            trust_region[d, 0] = max(
                                restart_point_x[0, d] - TR_size, inp_ranges[d, 0]
                            )
                            trust_region[d, 1] = min(
                                restart_point_x[0, d] + TR_size, inp_ranges[d, 1]
                            )
                    else:
                        # high pass of RC test
                        # restart_point_x = local_best_x
                        # restart_point_y = local_best_y
                        # if local_best_y[0,0] < restart_point_y[0,0]:
                        # idx = np.argmin(local_sample_y_subset[:,1])
                        # restart_point_x = local_sample_x_subset[idx].reshape(1,-1)
                        # restart_point_y = local_sample_y_subset[idx].reshape(1,-1)
                        idx = np.argmin(local_best_y[:,0])
                        restart_point_x = local_best_x[idx].reshape(1,-1)
                        restart_point_y = local_best_y[idx].reshape(1,-1)
                        valid_bound = np.array(
                            [
                                np.min(np.abs(restart_point_x[0, :] - inp_ranges[:, 0])),
                                np.min(np.abs(inp_ranges[:, 1] - restart_point_x[0, :])),
                                # (inp_ranges[:, 1] - inp_ranges[:, 0]) / min_tr_size,
                                TR_size * gamma,
                            ]
                        ).flatten()
                        # TR_size *= gamma
                        # if len(valid_bound[valid_bound >= tr_threshold]) != 0:
                        #     TR_size = np.min(valid_bound[valid_bound >= tr_threshold])
                        # else:
                        #     TR_size = TR_size * gamma
                        TR_size = np.min(valid_bound)
                        
                        # TR_size = np.min(valid_bound[valid_bound >= tr_threshold])
                        trust_region = np.empty((inp_ranges.shape))

                        for d in range(tf_dim):
                            trust_region[d, 0] = max(
                                restart_point_x[0, d] - TR_size, inp_ranges[d, 0]
                            )
                            trust_region[d, 1] = min(
                                restart_point_x[0, d] + TR_size, inp_ranges[d, 1]
                            )
                # print("*****************")
                # print(best_till_now)
                # print(trust_region)
                # print("*****************")
                local_counter += 1
                x_train_hd, y_train_hd = _generate_dataset(0, algo_journey)

                local_sample_x_subset, local_sample_y_subset = pointsInTR(
                    x_train_hd, y_train_hd, trust_region
                )
                num_points_present = local_sample_x_subset.shape[0]

                # print(f"{TR_size} ---- {eps_tr * np.min(inp_ranges[:, 1] - inp_ranges[:,0])}")

                # check if budget has been exhausted
        logger.debug(f"Best Till Now = {best_till_now}")
    return algo_journey
