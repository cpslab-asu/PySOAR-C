from __future__ import annotations

import enum
import logging
import math
import numbers
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from attrs import define, field
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from scipy.optimize import minimize as minimize_scipy
from scipy.stats import norm

from .gpr import GPR, DefaultGPR
from .sampling import lhs_sampling, uniform_sampling

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymoo.core.result import Result as PyMmooResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    gpr_model,
    rng,
) -> tuple[NDArray, NDArray, np.float64]:
    # Fit Gaussian Process Meta Model Locally
    gpr = GPR(deepcopy(gpr_model))
    gpr.fit(xTrain_local, yTrain_local)

    EI_obj = lambda x: -1 * EIcalc_kd(yTrain_local, x, gpr)
    lower_bound_theta = np.ndarray.flatten(trust_region[:, 0])
    upper_bound_theta = np.ndarray.flatten(trust_region[:, 1])

    random_samples = uniform_sampling(2000, trust_region, tf_dim, rng)
    min_bo_val = EI_obj(random_samples)

    min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])[0]
    min_bo_val = np.min(min_bo_val)

    for _ in range(9):
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
    xk = np.array([np.array(new_params.x)])

    rob = tf_wrapper(xk, test_fn, behavior)

    if rob is None and behavior is Behavior.COVERAGE:
        rho = [None]
    else:
        rho = (pred_sample_y[0][0] - rob[0][0]) / (
            gpr.predict(pred_sample_x)[0] - gpr.predict(xk)[0] + 1e-6
        )

    rho_ret = rho[0]
    return xk, rob, rho_ret


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


class PredictionError(Exception):
    ...


class TestSetShapeError(PredictionError):
    def __init__(self, x_test: NDArray):
        super().__init__(f"Received samples set input: Expected (n, dim) array, received {x_test.shape} instead.")


class MeanShapeError(PredictionError):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__(f"Mean from GPR should be of shape (n, ). Received {shape} instead.")


class StdDevShapeError(PredictionError):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__(f"Standard Deviation from GPR should be of shape (n, ). Received {shape} instead.")


class MismatchedDimensionError(PredictionError):
    def __init__(self, mean: NDArray, std_dev: NDArray):
        super().__init__(f"Mean and std_dev mismatch. Mean has a shape of {mean.shape} and std_dev has a shape of {std_dev.shape}.")


def _surrogate(model: GPR, x_test: NDArray) -> tuple[NDArray, NDArray]:
    """_surrogate Model function

    Attributes:
    ----------
        model: Gaussian process model
        X: Input points

    Returns:
    ---------
        Predicted values of points using gaussian process model
    """

    x_test = np.array(x_test)

    if x_test.ndim != 2:  # noqa: PLR2004
        raise TestSetShapeError(x_test)

    mean, std_dev = model.predict(x_test)

    if mean.ndim != 1:
        raise MeanShapeError(mean.shape)

    if std_dev.ndim != 1:
        raise StdDevShapeError(std_dev.shape)

    if mean.shape != std_dev.shape:
        raise MismatchedDimensionError(mean, std_dev)

    return mean, std_dev


def EIcalc_kd(y_train: NDArray, sample: NDArray, model: GPR) -> NDArray:
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
        mu, std = _surrogate(model, sample)
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

        mu, std = _surrogate(model, sample.reshape(1, -1))
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
    return np.sum(np.sqrt(np.sum((x_0 - x) ** 2, 1)))


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
    regionSamples = []
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
        regionSamples = samples_in[(np.all(boolArray, axis=0)), :]
    else:

        corresponding_robustenss = np.array([])
        regionSamples = np.array([[]])

    return regionSamples, corresponding_robustenss


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
        if self.local_best_y.shape != (1, 2):
            raise ValueError("local_best_y must be a 1x2 matrix.")


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

    samples: NDArray[np.double]
    costs: NDArray[np.double]

    def __post_init__(self):
        if type(self.samples) != np.ndarray:
            raise TypeError("initial_samples_x must be an NDArray")
        if type(self.costs) != np.ndarray:
            raise TypeError("initial_samples_y must be an NDArray")
        if len(self.samples.shape) != 2:
            raise ValueError("initial_samples_x must be a 2D array.")
        if len(self.costs.shape) != 2:
            raise ValueError("initial_samples_y must be a 2D array.")
        if self.samples.shape[0] != self.costs.shape[0]:
            raise ValueError(
                "The number of samples in initial_samples_x must match the length of initial_samples_y."
            )


def _generate_dataset(
    index: Literal[0, 1],
    init: InitializationPhase,
    *phases: GlobalPhase | LocalPhase | LocalBest
) -> tuple[NDArray, NDArray]:
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

    x_train = init.samples
    y_train = init.costs

    for phase in phases:
        if isinstance(phase, GlobalPhase):
            x_train = np.vstack((x_train, phase.restart_point_x))
            y_train = np.hstack((y_train, phase.restart_point_y[:, index]))
        elif isinstance(phase, LocalPhase):
            x_train = np.vstack((x_train, phase.local_phase_x))
            y_train = np.hstack((y_train, phase.local_phase_y[:, index]))
        else:
            x_train = np.vstack((x_train, phase.local_best_x))
            y_train = np.hstack((y_train, phase.local_best_y[:, index]))

    return x_train, y_train


def _is_falsification(evaluation: NDArray | None) -> bool:
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

    def __init__(self, func):
        """
        Initializes the Fn object with a function to be tracked.

        Parameters:
        ----------
        func : callable
            The function to be wrapped and tracked.
        """
        self.func = func
        self.count = 0

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

    for sample in samples:
        evaluation = np.array(fn(sample), dtype=np.double)
        evaluations.append(evaluation)

        if _is_falsification(evaluation) and behavior in (
            Behavior.FALSIFICATION,
            Behavior.COVERAGE,
        ):
            break

    return np.array(evaluations)


def _initialize() -> InitializationPhase:
    initial_samples = lhs_sampling(n_0, input_ranges, tf_dim, rng)
    initial_costs = _evaluate_samples(initial_samples, fn, behavior)
    init = InitializationPhase(initial_samples, initial_costs)


def _global_search() -> GlobalPhase:
    pass


def _local_search() -> LocalPhase | LocalBest:
    pass


@define(slots=True)
class Result:
    init: InitializationPhase
    phases: list[GlobalPhase | LocalPhase | LocalBest] = field(factory=list)


##### v9 ####### add user defined parameters to input, break once falsified
def soarc(
    n_0: int,
    nSamples: int,
    trs_max_budget: int,
    max_loc_iter: int,
    inpRanges: NDArray,
    alpha_lvl_set: float,
    eta0: float,
    eta1: float,
    delta: float,
    gamma: float,
    eps_tr: float,
    min_tr_size: float,
    TR_threshold: float,
    test_fn: Callable[[NDArray], float] | Callable[[NDArray], tuple[float, float]],
    gpr_model: GaussianProcessRegressor | None,
    seed: int,
    local_search: str,
    behavior: Behavior = Behavior.FALSIFICATION,
) -> Result:
    # Notes:
    # Add none check for InternalGPR
    # Rename gprs for something meaningful
    # Stick to one documnetation
    # Check function/class names

    inpRanges = np.array(inpRanges)
    fn = Fn(test_fn)
    gpr = DefaultGPR() if gpr_model is None else gpr_model

    if inpRanges.ndim != 2:
        raise ValueError("input ranges should be 2-dimensional")

    if inpRanges.shape[1] != 2:
        raise ValueError("input range 2nd dimension should be equal to 2")

    rng = np.random.default_rng(seed)
    tf_dim = inpRanges.shape[0]

    if n_0 > nSamples:
        raise ValueError(
            f"Received n_0({n_0}) > nSamples ({nSamples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (nSamples)"
        )

    result = Result(init)
    n_evals = initial_costs.size

    if any(_is_falsification(sd) for sd in initial_costs) and behavior is not Behavior.MINIMIZATION:
        return result

    while fn.count < nSamples:
        x_train, y_train = _generate_dataset(1, init, *result.phases)
        gpr = deepcopy(gpr)
        gpr.fit(x_train, y_train)

        lower_bound_theta = np.ndarray.flatten(inpRanges[:, 0])
        upper_bound_theta = np.ndarray.flatten(inpRanges[:, 1])

        problem = FunctionalProblem(
            n_var=inpRanges.shape[0],
            objs=[
                lambda x: -1 * EIcalc_kd(y_train, x, gpr),
                lambda x: -1 * CrowdingDist_kd(x, x_train),
            ],
            xl=lower_bound_theta,
            xu=upper_bound_theta,
        )

        algorithm = NSGA2(
            pop_size=50,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        ga_seed = rng.integers(low=1, high=100000, size=1)
        ga_result: PyMmooResult = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", 100),
            seed=ga_seed[0],
            verbose=False,
        )

        minNegEIindex = np.argmin(ga_result.F[:, 0])
        minNegEI = ga_result.F[minNegEIindex, 0]
        global_rp_x = ga_result.X[minNegEIindex]
        best_crowd = math.inf

        for k in range(ga_result.F.shape[0]):
            if ga_result.F[k, 0] <= minNegEI * (1 - alpha_lvl_set) and ga_result.F[k, 1] < best_crowd:
                # _logger.debug(
                #     f"{ga_result.F[k, 0]} <= {minNegEI * (1-alpha_lvl_set)} -> {ga_result.F[k,0] <= minNegEI * (1-alpha_lvl_set)} \\ {ga_result.F[k,1]} < {best_crowd} -> {ga_result.F[k,1] < best_crowd} \n{global_rp_x}\n*************************************************"
                # )
                best_crowd = ga_result.F[k, 1]
                global_rp_x = ga_result.X[k, :]

        global_rp_x = np.array([global_rp_x])
        global_rp_y = _evaluate_samples(global_rp_x, fn, behavior)

        result.phases.append(
            GlobalPhase(restart_point_x=global_rp_x, restart_point_y=global_rp_y)
        )

        if _is_falsification(global_rp_y[0]) and behavior is not Behavior.MINIMIZATION:
            return result

        local_sample_x, local_samples_y = _generate_dataset(0, init, *result.phases)
        TR_Bounds = np.vstack(
            [
                global_rp_x[0, :] - inpRanges[:, 0],
                inpRanges[:, 1] - global_rp_x[0, :],
                (inpRanges[:, 1] - inpRanges[:, 0]) / min_tr_size,
            ]
        ).flatten()

        TR_size = np.min(np.abs(TR_Bounds[TR_Bounds >= TR_threshold]))

        trust_region = np.empty(inpRanges.shape)

        for d in range(tf_dim):
            trust_region[d, 0] = max(global_rp_x[0, d] - TR_size, inpRanges[d, 0])
            trust_region[d, 1] = min(global_rp_x[0, d] + TR_size, inpRanges[d, 1])

        local_sample_x_subset, local_sample_y_subset = pointsInTR(local_sample_x, local_samples_y, trust_region)
        num_points_present = local_sample_x_subset.shape[0]
        local_counter = 0
        restart_point_x, restart_point_y = deepcopy(global_rp_x), deepcopy(global_rp_y)

        if local_search == "gp_local_search":
            while (
                local_counter < max_loc_iter
                and TR_size > eps_tr * np.min(inpRanges[:, 1] - inpRanges[:, 0])
                and fn.count + (max(trs_max_budget - num_points_present, 0) + 1)
                < nSamples
            ):

                # print(f"Needed: {trs_max_budget}, present: {num_points_present}, More {num_samples_needed} points needed")
                if trs_max_budget - num_points_present > 0:
                    num_samples_needed = trs_max_budget - num_points_present
                    local_additional_x = lhs_sampling(num_samples_needed, trust_region, tf_dim, rng)
                    local_additional_y = _evaluate_samples(local_additional_x, fn, behavior)

                    result.phases.append(LocalPhase(trust_region, local_additional_x, local_additional_y))

                    if any(_is_falsification(sd) for sd in local_additional_y) and (
                        behavior is Behavior.FALSIFICATION
                        or behavior is Behavior.COVERAGE
                    ):
                        return result

                x_train_hd, y_train_hd = _generate_dataset(0, init, *result.phases)
                local_sample_x_subset, local_sample_y_subset = pointsInTR(x_train_hd, y_train_hd, trust_region)

                # Fit Gaussian Process Meta Model Locally

                local_best_x, local_best_y, rho = local_best_ei(
                    restart_point_x,
                    restart_point_y,
                    _evaluate_samples,
                    test_fn,
                    tf_dim,
                    trust_region,
                    local_sample_x_subset,
                    local_sample_y_subset,
                    behavior,
                    gpr_model,
                    rng,
                )

                result.phases.append(LocalBest(local_best_x, local_best_y))

                if _is_falsification(local_best_y[0]) and (
                    behavior is Behavior.FALSIFICATION or behavior is Behavior.COVERAGE
                ):
                    return result

                max_indicator = np.max(np.abs(local_best_x - restart_point_x)) / TR_size
                test = rng.random()
                if max_indicator < test:
                    break

                # execute RC testing and TR control
                if rho < eta0:
                    TR_size *= delta
                    trust_region = np.empty((inpRanges.shape))
                    for d in range(tf_dim):
                        trust_region[d, 0] = max(
                            restart_point_x[0, d] - TR_size, inpRanges[d, 0]
                        )
                        trust_region[d, 1] = min(
                            restart_point_x[0, d] + TR_size, inpRanges[d, 1]
                        )
                else:
                    if eta0 < rho < eta1:
                        # low pass of RC test
                        restart_point_x = local_best_x
                        restart_point_y = local_best_y

                        valid_bound = np.array(
                            [
                                np.min(np.abs(restart_point_x[0, :] - inpRanges[:, 0])),
                                np.min(np.abs(inpRanges[:, 1] - restart_point_x[0, :])),
                                TR_size,
                            ]
                        ).flatten()
                        TR_size = np.min(valid_bound[valid_bound >= TR_threshold])
                        trust_region = np.empty((inpRanges.shape))

                        for d in range(tf_dim):
                            trust_region[d, 0] = max(
                                restart_point_x[0, d] - TR_size, inpRanges[d, 0]
                            )
                            trust_region[d, 1] = min(
                                restart_point_x[0, d] + TR_size, inpRanges[d, 1]
                            )
                    else:
                        # high pass of RC test
                        restart_point_x = local_best_x
                        restart_point_y = local_best_y
                        valid_bound = np.array(
                            [
                                np.min(np.abs(restart_point_x[0, :] - inpRanges[:, 0])),
                                np.min(np.abs(inpRanges[:, 1] - restart_point_x[0, :])),
                                TR_size * gamma,
                            ]
                        ).flatten()
                        # TR_size *= gamma
                        TR_size = np.min(valid_bound[valid_bound >= TR_threshold])
                        trust_region = np.empty((inpRanges.shape))

                        for d in range(tf_dim):
                            trust_region[d, 0] = max(
                                restart_point_x[0, d] - TR_size, inpRanges[d, 0]
                            )
                            trust_region[d, 1] = min(
                                restart_point_x[0, d] + TR_size, inpRanges[d, 1]
                            )

                local_counter += 1
                x_train_hd, y_train_hd = _generate_dataset(0, init, *result.phases)

                local_sample_x_subset, local_sample_y_subset = pointsInTR(
                    x_train_hd, y_train_hd, trust_region
                )
                num_points_present = local_sample_x_subset.shape[0]

                # print(f"{TR_size} ---- {eps_tr * np.min(inpRanges[:, 1] - inpRanges[:,0])}")

                # check if budget has been exhausted

    return result
