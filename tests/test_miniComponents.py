import pathlib
import pickle

import pytest
import numpy as np
import numpy.random as random
from dataclasses import FrozenInstanceError

from soar.optimizer import Behavior, InitializationPhase, GlobalPhase, LocalBest, LocalPhase, CrowdingDist_kd
from soar.sampling import lhs_sampling, uniform_sampling


@pytest.fixture()
def rng() -> random.Generator:
    return random.default_rng(10001)


@pytest.fixture()
def data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"

def test_behavior_enum_values():
    """Test that the enum values are assigned sequentially starting from 1."""
    assert Behavior.FALSIFICATION == 1
    assert Behavior.MINIMIZATION == 2
    assert Behavior.COVERAGE == 3

def test_behavior_enum_members():
    """Test that the Behavior enum has the correct members."""
    expected_members = ["FALSIFICATION", "MINIMIZATION", "COVERAGE"]
    actual_members = list(Behavior.__members__.keys())
    assert actual_members == expected_members

def test_behavior_enum_uniqueness():
    """Test that all enum values are unique."""
    values = [member.value for member in Behavior]
    assert len(values) == len(set(values)), "Enum values are not unique"

def test_behavior_enum_docs():
    """Test that the class docstring and attribute docstrings are not empty."""
    assert Behavior.__doc__, "Class docstring is empty"
    assert Behavior.FALSIFICATION.name == "FALSIFICATION"
    assert Behavior.MINIMIZATION.name == "MINIMIZATION"
    assert Behavior.COVERAGE.name == "COVERAGE"



def test_initialization_correct_values_IP():
    """Test that InitializationPhase correctly initializes the attributes."""
    samples_x = np.array([[1.0, 2.0], [3.0, 4.0], [2,4], [5,4]])  # DxN matrix (4x2)
    samples_y = np.array([[10.0, 20.0], [30, 40], [4,3], [5., 6.]])             # 2D array with N elements
    
    init_phase = InitializationPhase(initial_samples_x=samples_x, initial_samples_y=samples_y)
    
    np.testing.assert_array_equal(init_phase.initial_samples_x, samples_x)
    np.testing.assert_array_equal(init_phase.initial_samples_y, samples_y)

def test_frozen_dataclass_immutability_IP():
    """Test that InitializationPhase is immutable."""
    samples_x = np.array([[1.0, 2.0], [3.0, 4.0], [2,4], [5,4]])  # DxN matrix (4x2)
    samples_y = np.array([[10.0, 20.0], [30, 40], [4,3], [5., 6.]])             # 2D array with N elements
    
    init_phase = InitializationPhase(initial_samples_x=samples_x, initial_samples_y=samples_y)
    
    with pytest.raises(FrozenInstanceError):
        init_phase.initial_samples_x = np.array([[0.0, 0.0], [0.0, 0.0]])

def test_invalid_types_IP():
    """Test that InitializationPhase raises an error when given invalid types."""
    with pytest.raises(TypeError):
        InitializationPhase(initial_samples_x="not an array", initial_samples_y=np.array([1.0, 2.0]))
    
    with pytest.raises(TypeError):
        InitializationPhase(initial_samples_x=np.array([[1.0, 2.0]]), initial_samples_y="not an array")

def test_invalid_shapes_IP():
    """Test that InitializationPhase handles invalid input shapes."""
    samples_x_invalid = np.array([1.0, 2.0, 3.0])  # Not a 2D array
    samples_y_invalid = np.array([[10.0, 20.0]])   # Not a 1D array
    
    with pytest.raises(ValueError):
        InitializationPhase(initial_samples_x=samples_x_invalid, initial_samples_y=np.array([10.0, 20.0]))
    
    with pytest.raises(ValueError):
        InitializationPhase(initial_samples_x=np.array([[1.0, 2.0], [3.0, 4.0]]), initial_samples_y=samples_y_invalid)

def test_shape_consistency_IP():
    """Test that InitializationPhase checks consistency between samples_x and samples_y."""
    samples_x = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
    samples_y_mismatch = np.array([10.0, 20.0, 30.0])  # Mismatched shape
    
    with pytest.raises(ValueError):
        InitializationPhase(initial_samples_x=samples_x, initial_samples_y=samples_y_mismatch)



def test_globalphase_initialization_correct_values_GP():
    """Test that GlobalPhase correctly initializes the attributes."""
    restart_x = np.array([[1.0, 2.0, 3.0]])           # 2-dimensional vector
    restart_y = np.array([[10.0, 20.0]])            # 1x2 matrix
    
    global_phase = GlobalPhase(restart_point_x=restart_x, restart_point_y=restart_y)
    
    np.testing.assert_array_equal(global_phase.restart_point_x, restart_x)
    np.testing.assert_array_equal(global_phase.restart_point_y, restart_y)

def test_frozen_dataclass_immutability_GP():
    """Test that GlobalPhase is immutable."""
    restart_x = np.array([[1.0, 2.0, 3.0]])
    restart_y = np.array([[10.0, 20.0]])
    # print(type(restart_x))
    # print(type(restart_y))
    global_phase = GlobalPhase(restart_point_x=restart_x, restart_point_y=restart_y)
    
    with pytest.raises(FrozenInstanceError):
        global_phase.restart_point_x = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(FrozenInstanceError):
        global_phase.restart_point_y = np.array([[0.0, 0.0]])

def test_invalid_types_GP():
    """Test that GlobalPhase raises an error when given invalid types."""
    # GlobalPhase(restart_point_x="not an array", restart_point_y=np.array([[10.0, 20.0]]))
    with pytest.raises(TypeError):
        GlobalPhase(restart_point_x="not an array", restart_point_y=np.array([[10.0, 20.0]]))
    
    with pytest.raises(TypeError):
        GlobalPhase(restart_point_x=np.array([1.0, 2.0, 3.0]), restart_point_y="not an array")

def test_invalid_shapes_GP():
    """Test that GlobalPhase raises an error for invalid input shapes."""
    restart_x_invalid = np.array([1.0, 2.0, 3.0])  # Not a 1D array
    restart_y_invalid = np.array([10.0, 20.0])       # Not a 1x2 matrix
    
    with pytest.raises(ValueError):
        GlobalPhase(restart_point_x=restart_x_invalid, restart_point_y=np.array([[10.0, 20.0]]))
    
    with pytest.raises(ValueError):
        GlobalPhase(restart_point_x=np.array([[1.0, 2.0, 3.0]]), restart_point_y=restart_y_invalid)

def test_restart_point_y_shape_GP():
    """Test that restart_point_y specifically requires a 1x2 matrix."""
    restart_x = np.array([[1.0, 2.0, 3.0]])
    restart_y_invalid = np.array([[10.0], [20.0]])  # Not 1x2
    
    with pytest.raises(ValueError):
        GlobalPhase(restart_point_x=restart_x, restart_point_y=restart_y_invalid)

def test_restart_point_x_shape_GP():
    """Test that restart_point_x requires a 1-dimensional vector."""
    restart_x_invalid = np.array([1.0, 2.0])  # 1D
    restart_y = np.array([[10.0, 20.0]])
    
    with pytest.raises(ValueError):
        GlobalPhase(restart_point_x=restart_x_invalid, restart_point_y=restart_y)



def test_localbest_initialization_correct_values_LB():
    """Test that LocalBest correctly initializes the attributes."""
    local_x = np.array([[1.0, 2.0, 3.0]])           # 1-dimensional vector
    local_y = np.array([[5.0, 10.0]])             # 1x2 matrix
    
    local_best = LocalBest(local_best_x=local_x, local_best_y=local_y)
    
    np.testing.assert_array_equal(local_best.local_best_x, local_x)
    np.testing.assert_array_equal(local_best.local_best_y, local_y)


def test_frozen_dataclass_immutability_LB():
    """Test that LocalBest is immutable."""
    local_x = np.array([[1.0, 2.0, 3.0]])
    local_y = np.array([[5.0, 10.0]])
    
    local_best = LocalBest(local_best_x=local_x, local_best_y=local_y)
    
    with pytest.raises(FrozenInstanceError):
        local_best.local_best_x = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(FrozenInstanceError):
        local_best.local_best_y = np.array([[0.0, 0.0]])


def test_invalid_types_LB():
    """Test that LocalBest raises an error when given invalid types."""
    with pytest.raises(TypeError):
        LocalBest(local_best_x="invalid", local_best_y=np.array([[5.0, 10.0]]))
    
    with pytest.raises(TypeError):
        LocalBest(local_best_x=np.array([[1.0, 2.0]]), local_best_y="invalid")


def test_invalid_shapes_LB():
    """Test that LocalBest raises an error for invalid input shapes."""
    local_x_invalid = np.array([1.0, 2.0, 3.0])  # Not a 1D array
    local_y_invalid = np.array([5.0, 10.0])        # Not a 1x2 matrix
    
    with pytest.raises(ValueError):
        LocalBest(local_best_x=local_x_invalid, local_best_y=np.array([[5.0, 10.0]]))
    
    with pytest.raises(ValueError):
        LocalBest(local_best_x=np.array([[1.0, 2.0, 3.0]]), local_best_y=local_y_invalid)


def test_local_best_y_shape_LB():
    """Test that local_best_y specifically requires a 1x2 matrix."""
    local_x = np.array([[1.0, 2.0, 3.0]])
    local_y_invalid = np.array([[5.0], [10.0]])  # Not 1x2
    
    with pytest.raises(ValueError):
        LocalBest(local_best_x=local_x, local_best_y=local_y_invalid)


def test_local_best_x_shape_LB():
    """Test that local_best_x requires a 1-dimensional vector."""
    local_x_invalid = np.array([1.0, 2.0, 3.0])  # Not 2D
    local_y = np.array([[5.0, 10.0]])
    # LocalBest(local_best_x=local_x_invalid, local_best_y=local_y)
    with pytest.raises(ValueError):
        LocalBest(local_best_x=local_x_invalid, local_best_y=local_y)

def test_localphase_initialization_correct_values_LP():
    """Test that LocalPhase correctly initializes the attributes."""
    region_support = np.array([[0.0, 1.0], [1.0, 2.0]])  # Shape (2, 2)
    local_x = np.array([[0.5, 1.5], [0.7, 1.7]])        # Shape (2, 2)
    local_y = np.array([[5.0, 10.0], [6.0, 12.0]])      # Shape (2, 2)
    
    local_phase = LocalPhase(region_support=region_support, 
                             local_phase_x=local_x, 
                             local_phase_y=local_y)
    
    np.testing.assert_array_equal(local_phase.region_support, region_support)
    np.testing.assert_array_equal(local_phase.local_phase_x, local_x)
    np.testing.assert_array_equal(local_phase.local_phase_y, local_y)


def test_frozen_dataclass_immutability_LP():
    """Test that LocalPhase is immutable."""
    region_support = np.array([[0.0, 1.0], [1.0, 2.0]])
    local_x = np.array([[0.5, 1.5]])
    local_y = np.array([[5.0, 10.0]])
    
    local_phase = LocalPhase(region_support=region_support, 
                             local_phase_x=local_x, 
                             local_phase_y=local_y)
    
    with pytest.raises(FrozenInstanceError):
        local_phase.region_support = np.array([[0.0, 2.0]])
    
    with pytest.raises(FrozenInstanceError):
        local_phase.local_phase_x = np.array([[0.0, 0.0]])
    
    with pytest.raises(FrozenInstanceError):
        local_phase.local_phase_y = np.array([[0.0, 0.0]])


def test_invalid_types_LP():
    """Test that LocalPhase raises an error when given invalid types."""
    with pytest.raises(TypeError):
        LocalPhase(region_support="invalid", 
                   local_phase_x=np.array([[0.5, 1.5]]), 
                   local_phase_y=np.array([[5.0, 10.0]]))
    
    with pytest.raises(TypeError):
        LocalPhase(region_support=np.array([[0.0, 1.0]]), 
                   local_phase_x="invalid", 
                   local_phase_y=np.array([[5.0, 10.0]]))
    
    with pytest.raises(TypeError):
        LocalPhase(region_support=np.array([[0.0, 1.0]]), 
                   local_phase_x=np.array([[0.5, 1.5]]), 
                   local_phase_y="invalid")


def test_invalid_shapes_LP():
    """Test that LocalPhase raises an error for invalid input shapes."""
    region_support_invalid = np.array([0.0, 1.0])  # Not (d, 2)
    local_x_invalid = np.array([0.5, 1.5])         # Not (n, d)
    local_y_invalid = np.array([[5.0], [10.0]])    # Not (n, 2)
    
    with pytest.raises(ValueError):
        LocalPhase(region_support=region_support_invalid, 
                   local_phase_x=np.array([[0.5, 1.5]]), 
                   local_phase_y=np.array([[5.0, 10.0]]))
    
    with pytest.raises(ValueError):
        LocalPhase(region_support=np.array([[0.0, 1.0]]), 
                   local_phase_x=local_x_invalid, 
                   local_phase_y=np.array([[5.0, 10.0]]))
    
    with pytest.raises(ValueError):
        LocalPhase(region_support=np.array([[0.0, 1.0]]), 
                   local_phase_x=np.array([[0.5, 1.5]]), 
                   local_phase_y=local_y_invalid)


def test_region_support_shape_LP():
    """Test that region_support specifically requires a (d, 2) matrix."""
    region_support_invalid = np.array([0.0, 1.0])  # Not (d, 2)
    local_x = np.array([[0.5, 1.5]])
    local_y = np.array([[5.0, 10.0]])
    
    with pytest.raises(ValueError):
        LocalPhase(region_support=region_support_invalid, 
                   local_phase_x=local_x, 
                   local_phase_y=local_y)


def test_local_phase_x_shape_LP():
    """Test that local_phase_x requires an (n, d) matrix."""
    local_x_invalid = np.array([0.5, 1.5])  # Not (n, d)
    region_support = np.array([[0.0, 1.0]])
    local_y = np.array([[5.0, 10.0]])
    
    with pytest.raises(ValueError):
        LocalPhase(region_support=region_support, 
                   local_phase_x=local_x_invalid, 
                   local_phase_y=local_y)


def test_local_phase_y_shape_LP():
    """Test that local_phase_y requires an (n, 2) matrix."""
    local_y_invalid = np.array([[5.0], [10.0]])  # Not (n, 2)
    region_support = np.array([[0.0, 1.0]])
    local_x = np.array([[0.5, 1.5]])
    
    with pytest.raises(ValueError):
        LocalPhase(region_support=region_support, 
                   local_phase_x=local_x, 
                   local_phase_y=local_y_invalid)
        

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