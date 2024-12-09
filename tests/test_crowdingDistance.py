import numpy as np
from numpy import random
import pathlib
import pytest
from soar.optimizer import CrowdingDist_kd

@pytest.fixture()
def rng() -> random.Generator:
    return np.random.default_rng(10001)

@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"

# Test cases
def test_single_point_single_existing_point():
    x_0 = np.array([[1, 1]])
    x = np.array([[2, 2]])
    expected_output = np.sqrt((1 - 2)**2 + (1 - 2)**2)
    assert np.isclose(CrowdingDist_kd(x_0, x), expected_output)

def test_single_point_multiple_existing_points():
    x_0 = np.array([[0, 0]])
    x = np.array([[1, 1], [2, 2], [3, 3]])
    expected_output = sum([np.sqrt((0 - xi[0])**2 + (0 - xi[1])**2) for xi in x])
    assert np.isclose(CrowdingDist_kd(x_0, x), expected_output)


def test_multiple_dimensions_3d():
    x_0 = np.array([[1, 2, 3]])
    x = np.array([[4, 5, 6], [7, 8, 9]])
    expected_output = sum([np.sqrt(sum((x_0[0] - xi)**2)) for xi in x])
    assert np.isclose(CrowdingDist_kd(x_0, x), expected_output)

def test_identical_points():
    x_0 = np.array([[1, 2]])
    x = np.array([[1, 2]])
    expected_output = 0.0
    assert np.isclose(CrowdingDist_kd(x_0, x), expected_output)

def test_large_dataset(rng: random.Generator):
    x_0 = rng.random((1, 10))
    x = rng.random((100, 10))
    result = CrowdingDist_kd(x_0, x)
    assert result > 0  # General check to ensure computation

def test_empty_x():
    x_0 = np.array([[1, 2]])
    x = np.empty((0, 2))
    expected_output = 0.0
    assert np.isclose(CrowdingDist_kd(x_0, x), expected_output)


def test_high_dimensional_input(rng: random.Generator):
    x_0 = rng.random((1, 1000))
    x = rng.random((10, 1000))
    result = CrowdingDist_kd(x_0, x)
    assert result > 0