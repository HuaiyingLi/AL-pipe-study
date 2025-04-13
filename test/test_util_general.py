"""Test file for util.general."""

import random

import numpy as np
import torch

from al_pipe.util.data import seed_all


def test_returned_randomstate():  # noqa: D103
    # Call seed_all with a specific seed and capture the returned RandomState.
    rs = seed_all(0)
    # Create an expected RandomState using the same seed (0).
    expected_rs = np.random.RandomState(0)
    # Check that they produce the same sequence of numbers.
    np.testing.assert_array_equal(rs.rand(5), expected_rs.rand(5))


def test_global_numpy_seed():  # noqa: D103
    # Reset the global numpy random seed by calling seed_all.
    seed_all(42)
    a = np.random.rand(5)
    # Reset again with the same seed.
    seed_all(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_global_python_random():  # noqa: D103
    seed_all(42)
    a = random.random()
    seed_all(42)
    b = random.random()
    assert a == b


def test_global_torch_seed():  # noqa: D103
    seed_all(42)
    a = torch.rand(5)
    seed_all(42)
    b = torch.rand(5)
    # Use torch.allclose to compare floating-point tensors
    assert torch.allclose(a, b)
