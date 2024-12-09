import pathlib
import pickle

import pytest
import numpy as np
import numpy.random as random

from soar.sampling import lhs_sampling, uniform_sampling


@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)


@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
    

def test_uniform_sampling_2d_region_3d_tf(rng: random.Generator):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 3
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, rng)


def test_uniform_sampling_3d_region_2d_tf(rng: random.Generator):
    region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, rng)


def test_uniform_sampling_region_desc(rng: random.Generator):
    region_support = np.array([[1, -1], [1, -1]])
    tf_dim = 2
    num_samples = 10
    
    with pytest.raises(ValueError):
        uniform_sampling(num_samples, region_support, tf_dim, rng)

def test_uniform_sampling_array_shape_check_2dim(rng: random.Generator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 2
    num_samples = 10

    samples_in_unif = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )
    
    with open(data_path / "unif_samp_t1.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test_uniform_sampling_array_shape_check_4dim(rng: random.Generator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
    tf_dim = 4
    num_samples = 10

    samples_in_unif = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )
    
    with open(data_path / "unif_samp_t2.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test_lhs_sampling_2d_region_3d_tf(rng: random.Generator):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 3
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, rng)


def test_lhs_sampling_3d_region_2d_tf(rng: random.Generator):
    region_support = np.array([[-1, 1, 2], [-1, 1, 2]])
    tf_dim = 2
    num_samples = 10

    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, rng)


def test_lhs_sampling_region_desc(rng: random.Generator):
    region_support = np.array([[1, -1], [1, -1]])
    tf_dim = 2
    num_samples = 10
    
    with pytest.raises(ValueError):
        lhs_sampling(num_samples, region_support, tf_dim, rng)

def test_lhs_sampling_array_shape_check_2dim(rng: random.Generator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1], [-1, 1]])
    tf_dim = 2
    num_samples = 10

    samples_in_unif = lhs_sampling(
        num_samples, region_support, tf_dim, rng
    )
    
    with open(data_path / "lhs_samp_t1.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)

def test_lhs_sampling_array_shape_check_4dim(rng: random.Generator, data_path: pathlib.Path):
    region_support = np.array([[-1, 1],[-3.8, 1.5], [-2, 1], [-1, 1]])
    tf_dim = 4
    num_samples = 10

    samples_in_unif = lhs_sampling(
        num_samples, region_support, tf_dim, rng
    )
    
    with open(data_path / "lhs_samp_t2.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(samples_in_unif, gr)
