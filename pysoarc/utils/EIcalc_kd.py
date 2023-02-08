from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm


def _surrogate(gpr_model: Callable, x_train: NDArray):
    """_surrogate Model function

    Args:
        model: Gaussian process model
        X: Input points

    Returns:
        Predicted values of points using gaussian process model
    """

    return gpr_model.predict(x_train)

def EIcalc_kd(y_train: NDArray, sample: NDArray, gpr_model: Callable) -> NDArray:
    """Acquisition Model: Expected Improvement

    Args:
        y_train: corresponding robustness values
        sample: Sample(s) whose EI is to be calculated
        gpr_model: GPR model
        sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

    Returns:
        EI of samples
    """
    curr_best = np.min(y_train)
    # print(sample.shape)
    if len(sample.shape) == 2:
        mu, std = _surrogate(gpr_model, sample)
        ei_list = []
        for mu_iter, std_iter in zip(mu, std):
            pred_var = std_iter
            if pred_var > 0:
                var_1 = curr_best - mu_iter
                var_2 = var_1 / pred_var

                ei = (var_1 * norm.cdf(var_2)) + (
                    pred_var * norm.pdf(var_2)
                )
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

            ei = (var_1 * norm.cdf(var_2)) + (
                pred_var * norm.pdf(var_2)
            )
        else:
            ei = 0.0
        return_ei = ei

    
        

    return return_ei