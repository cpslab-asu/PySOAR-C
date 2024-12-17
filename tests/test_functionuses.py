import pathlib
import pickle

import pytest
import numpy as np
import numpy.random as random
from dataclasses import FrozenInstanceError

from soar.optimizer import Behavior, InitializationPhase, GlobalPhase, LocalBest, LocalPhase, CrowdingDist_kd, Fn
from soar.optimizer import _evaluate_samples
from soar.sampling import lhs_sampling, uniform_sampling

from .ha_tf import HA

@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)


@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"

# def test_FN_uniformRandom(rng: random.Generator, data_path: pathlib.Path):

#     ha = HA()

#     region_support = np.array([[-1.,1.],[-1.,1.]])
#     test_fn = Fn(ha.get_cost)
#     tf_dim = 2
#     num_samples = 500

#     samples_in_unif = uniform_sampling(
#         num_samples, region_support, tf_dim, rng
#     )

#     tf_wrapper = Fn(test_fn)
    
#     evaluations = []
#     for sample in samples_in_unif:
#         evaluation = np.array(tf_wrapper(sample), dtype=np.double)
#         evaluations.append(evaluation)
    
#     with open(data_path / "unif_samp_FN_GR.pickle", "rb") as f:
#         # pickle.dump(np.array(evaluations), f)
#         gr = pickle.load(f)
    
#     np.testing.assert_array_equal(np.array(evaluations), gr)

    
    # initial_sample_distances = _evaluate_samples(
    #     samples_in_unif, test_fn, Behavior.FALSIFICATION
    # )

def test_FN_uniformRandom(rng: random.Generator, data_path: pathlib.Path):

    ha = HA()

    region_support = np.array([[-1.,1.],[-1.,1.]])
    test_fn = Fn(ha.get_cost)
    tf_dim = 2
    num_samples = 500

    samples_in_unif = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )

    with open(data_path / "unif_samp_FN_GR_samples.pickle", "wb") as f:
        pickle.dump(samples_in_unif, f)
        # samples_in_unif = pickle.load(f)

    tf_wrapper = Fn(test_fn)
    
    evaluations = []
    for sample in samples_in_unif:
        evaluation = np.array(tf_wrapper(sample), dtype=np.double)
        evaluations.append(evaluation)
    
    with open(data_path / "unif_samp_FN_GR_evaluations.pickle", "rb") as f:
        # pickle.dump(np.array(evaluations), f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(np.array(evaluations), gr)
    
def test_FN_uniformRandom_count_noreset(rng: random.Generator):

    ha = HA()

    region_support = np.array([[-1.,1.],[-1.,1.]])
    tf_dim = 2
    num_samples = 500

    samples_in_unif_1 = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )

    tf_wrapper_1 = Fn(ha.get_cost)
    
    for sample in samples_in_unif_1:
        np.array(tf_wrapper_1(sample), dtype=np.double)
    
    assert tf_wrapper_1.count == 500

    samples_in_unif_2 = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )
    
    for sample in samples_in_unif_2:
        np.array(tf_wrapper_1(sample), dtype=np.double)
    assert tf_wrapper_1.count == 1000

def test_FN_uniformRandom_count_reset(rng: random.Generator, data_path: pathlib.Path):

    ha = HA()

    region_support = np.array([[-1.,1.],[-1.,1.]])
    tf_dim = 2
    num_samples = 500

    samples_in_unif_1 = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )

    tf_wrapper_1 = Fn(ha.get_cost)
    
    for sample in samples_in_unif_1:
        np.array(tf_wrapper_1(sample), dtype=np.double)
    
    assert tf_wrapper_1.count == 500


    tf_wrapper_2 = Fn(ha.get_cost)
    samples_in_unif_2 = uniform_sampling(
        num_samples, region_support, tf_dim, rng
    )
    
    for sample in samples_in_unif_2:
        np.array(tf_wrapper_2(sample), dtype=np.double)
    assert tf_wrapper_2.count == 500

def test_evaluateSamples_uniformRandom_MINIMIZATION(rng: random.Generator, data_path: pathlib.Path):

    ha = HA()
    test_fn = Fn(ha.get_cost)

    with open(data_path / "unif_samp_FN_GR_samples.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        samples_in_unif = pickle.load(f)

    tf_wrapper = Fn(test_fn)
    
    initial_sample_distances = _evaluate_samples(
        samples_in_unif, tf_wrapper, Behavior.MINIMIZATION
    )
    print(initial_sample_distances)

    with open(data_path / "unif_samp_FN_GR_evaluations_evaluateSamples_MINI.pickle", "rb") as f:
        # pickle.dump(initial_sample_distances, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(initial_sample_distances, gr)


def test_evaluateSamples_uniformRandom_FALSIFICATION(rng: random.Generator, data_path: pathlib.Path):

    ha = HA()
    test_fn = Fn(ha.get_cost)

    with open(data_path / "unif_samp_FN_GR_samples.pickle", "rb") as f:
        # pickle.dump(samples_in_unif, f)
        samples_in_unif = pickle.load(f)

    tf_wrapper = Fn(test_fn)
    
    initial_sample_distances = _evaluate_samples(
        samples_in_unif, tf_wrapper, Behavior.FALSIFICATION
    )
    print(initial_sample_distances)

    with open(data_path / "unif_samp_FN_GR_evaluations_evaluateSamples_FALSI.pickle", "rb") as f:
        # pickle.dump(initial_sample_distances, f)
        gr = pickle.load(f)
    
    np.testing.assert_array_equal(initial_sample_distances, gr)


    
    
