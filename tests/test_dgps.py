import numpy as np
import pytest
from src.dgps import sample_normal, sample_t, sigma_single_spike, sigma_multi_spike


@pytest.mark.parametrize("p, n", [(100, 20), (500, 100)])
def test_sample_normal_shape(p, n):
    """sample_normal() should return an array of shape (n, p)."""
    Sigma, _ = sigma_single_spike(p)
    X = sample_normal(Sigma=Sigma, n=n)
    assert X.shape == (n, p)


@pytest.mark.parametrize("p, n, nu", [(100, 20, 5), (500, 100, 7)])
def test_sample_t_shape(p, n, nu):
    """sample_t() should also return an array of shape (n, p)."""
    Sigma, _ = sigma_multi_spike(p)
    X = sample_t(Sigma=Sigma, nu=nu, n=n)
    assert X.shape == (n, p)


def test_sample_normal_reproducibility():
    """sample_normal() should produce identical results for the same seed, and different ones for different seeds."""
    Sigma, _ = sigma_single_spike(100)
    X1 = sample_normal(Sigma, n=5, seed=123)
    X2 = sample_normal(Sigma, n=5, seed=123)
    X3 = sample_normal(Sigma, n=5, seed=124)
    assert np.allclose(X1, X2)
    assert not np.allclose(X1, X3)


def test_sample_t_reproducibility():
    """sample_t() should be reproducible with the same seed."""
    Sigma, _ = sigma_single_spike(100)
    X1 = sample_t(Sigma, nu=5, n=5, seed=123)
    X2 = sample_t(Sigma, nu=5, n=5, seed=123)
    X3 = sample_t(Sigma, nu=5, n=5, seed=124)
    assert np.allclose(X1, X2)
    assert not np.allclose(X1, X3)