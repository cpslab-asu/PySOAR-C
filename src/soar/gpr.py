from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class GPR(ABC):
    @abstractmethod
    def fit(self, x_train: NDArray, y_train: NDArray):
        """Method to fit gpr Model

        Attributes:
        ----------
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        
        """

        raise NotImplementedError()

    @abstractmethod
    def predict(self, x_test: NDArray) -> tuple[NDArray, NDArray]:
        """Method to predict mean and std_dev from gpr model

        Attributes:
        ----------
            x_train: Samples from Training set.
            

        Returns:
        ---------
            mean
            std_dev
        """

        raise NotImplementedError()

    def do_fit(self, x_train: ArrayLike, y_train: ArrayLike):
        """ Wrapper to fit user defined gpr model

        Attributes:
        ----------
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        Raises:
        ----------
            TypeError: If x_train is not 2 dimensional numpy array
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train
        """

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if len(x_train.shape) != 2:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {x_train.shape} instead.")

        if len(y_train.shape) != 1:
            raise TypeError(f"Received evaluations set input: Expected (n,) array, received {y_train.shape} instead.")

        if x_train.shape[0] != y_train.shape[0]:
            raise TypeError(f"x_train, y_train set mismatch. x_train has shape {x_train.shape} and y_train has shape {y_train.shape}")

        self.fit(x_train, y_train)

    def do_predict(self, x_test: ArrayLike):
        """Wrapper to predict from user defined gpr model

        Attributes:
        ----------
            X: Samples for predicting

        Raises:
        ----------
            TypeError: If x_train is not 2 dimensional numpy array

        Returns:
        ----------
            mean
            std
        """
        
        x_test = np.array(x_test)

        if len(x_test.shape) != 2:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {x_test.shape} instead.")

        mean, std = self.predict(x_test)

        assert len(mean.shape) == 1, f"Mean from GPR should be of shape (n, ). Received {mean.shape} instead."
        assert len(std.shape) == 1, f"std_dev from GPR should be of shape (n, ). Received {std.shape} instead."
        assert mean.shape == std.shape, f"Mean and std_dev mismatch. Mean has a shape of {mean.shape} and std_dev has a shape of {std.shape}."

        return mean, std


def optimizer_lbfgs_b(obj_func, initial_theta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = fmin_l_bfgs_b(
            obj_func, initial_theta, maxiter=30000, maxfun=int(1e10)
        )
    return params[0], params[1]


class DefaultGPR(GPR):
    def __init__(self, random_state = 12345):
        self.scale = StandardScaler()
        self.model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state = random_state,
        )

    def fit(self, x_train: NDArray, y_train: NDArray):
        """Method to fit gpr Model

        Attributes:
        ----------
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
        """

        x_scaled = self.scale.fit_transform(x_train)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(x_scaled, y_train)

    def predict(self, x_test: NDArray) -> tuple[NDArray, NDArray]:
        """Method to predict mean and std_dev from gpr model

        Attributes:
        ----------
            x_train: Samples from Training set.
            

        Returns:
        ---------
            mean
            std_dev
        """

        x_scaled = self.scale.transform(x_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yPred, predSigma = self.model.predict(x_scaled, return_std=True)

        return yPred, predSigma

