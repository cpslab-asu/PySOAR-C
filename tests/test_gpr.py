import pathlib
import pickle

import numpy as np
import numpy.random as rand
import pytest

from soar.gpr import GPR, InternalGPR
from soar.sampling import uniform_sampling


@pytest.fixture()
def rng() -> rand.Generator:
    return rand.default_rng(10001)


@pytest.fixture()
def data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


def _internal_function(x):
    return x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2


def test_gpr_incorrect_input_shape_fitting(rng: rand.Generator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
    in_samples_1 = uniform_sampling(20, region_support, 3, rng)
    out_samples_1 = _internal_function(in_samples_1)

    with pytest.raises(TypeError):
        gpr.fit(np.array([in_samples_1]), out_samples_1)


def test_gpr_incorrect_output_shape_fitting(rng: rand.Generator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
    in_samples_1 = uniform_sampling(20, region_support, 3, rng)
    out_samples_1 = _internal_function(in_samples_1)

    with pytest.raises(TypeError):
        gpr.fit(in_samples_1, np.array([out_samples_1]).T)


def test_gpr_inconsistent_iodat_fitting(rng: rand.Generator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
    in_samples_1 = uniform_sampling(20, region_support, 3, rng)
    out_samples_1 = _internal_function(in_samples_1)
    in_samples_2 = uniform_sampling(10, region_support, 3, rng)
    out_samples_2 = _internal_function(in_samples_2)

    with pytest.raises(TypeError):
        gpr.fit(in_samples_1, out_samples_2)


def test_gpr_inconsistent_input_prediction(rng: rand.Generator):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
    in_samples_1 = uniform_sampling(20, region_support, 3, rng)
    out_samples_1 = _internal_function(in_samples_1)
    in_samples_2 = uniform_sampling(10, region_support, 3, rng)
    out_samples_2 = _internal_function(in_samples_2)

    gpr.fit(in_samples_1, out_samples_1)

    with pytest.raises(TypeError):
        gpr.predict(np.array([in_samples_1]))


def test_gpr_output_prediction(rng: rand.Generator, data_dir: pathlib.Path):
    gpr_model = InternalGPR()
    gpr = GPR(gpr_model)
    region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
    in_samples_1 = uniform_sampling(20, region_support, 3, rng)
    out_samples_1 = _internal_function(in_samples_1)

    gpr.fit(in_samples_1, out_samples_1)

    y_pred, y_std = gpr.predict(in_samples_1)
    data_file = data_dir / "test_1_gpr.pickle"

    with data_file.open("rb") as f:
        # pickle.dump([y_pred_1, y_std_1], f)
        gr_pred, gr_std = pickle.load(f)

    np.testing.assert_array_almost_equal(y_pred, gr_pred, decimal=2)
    np.testing.assert_array_almost_equal(y_std, gr_std, decimal=2)

    in_samples_2 = uniform_sampling(10, region_support, 3, rng)
    out_samples_2 = _internal_function(in_samples_2)
    y_pred, y_std = gpr.predict(in_samples_2)
    data_file = data_dir / "test_2_gpr.pickle"

    with data_file.open("rb") as f:
        # pickle.dump([y_pred_2, y_std_2], f)
        gr_pred, gr_std = pickle.load(f)

    np.testing.assert_array_almost_equal(y_pred, gr_pred, decimal=2)
    np.testing.assert_array_almost_equal(y_std, gr_std, decimal=2)
