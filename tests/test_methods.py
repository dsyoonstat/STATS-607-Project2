import numpy as np
import pytest

from src.methods import compute_PC_subspace, compute_ARG_PC_subspace
from src.dgps import generate_basis, generate_reference_vectors, sigma_single_spike, sample_normal


def is_orthonormal(Q: np.ndarray, atol: float = 1e-10) -> bool:
    """
    Check column-orthonormality: Q^T Q ≈ I_k.
    """
    if Q.ndim != 2:
        return False
    k = Q.shape[1]
    G = Q.T @ Q
    return np.allclose(G, np.eye(k), atol=atol)


@pytest.mark.parametrize("p, n, m", [
    (100, 40, 1),
    (120, 50, 2),   # m <= n-1
])
def test_compute_PC_subspace(p, n, m):
    """
    compute_PC_subspace should return (p, m) with column-orthonormal basis.
    """
    # simple single-spike covariance just to get a PSD Sigma
    Sigma, _ = sigma_single_spike(p, coef=(1.0, 5.0))
    rng = np.random.default_rng(123)
    X = sample_normal(Sigma=Sigma, n=n, mu=np.zeros(p), seed=int(rng.integers(0, 2**31-1)))

    U_pc = compute_PC_subspace(samples=X, spike_number=m)
    assert U_pc.shape == (p, m)
    assert is_orthonormal(U_pc)


@pytest.mark.parametrize("p, n, m, r", [
    (100, 40, 1, 1),   # single spike target, single reference
    (120, 50, 2, 2),   # 2-dim target, two references
])
def test_compute_ARG_PC_subspace(p, n, m, r):
    """
    compute_ARG_PC_subspace should return (p, m). If orthonormal=True, columns should be orthonormal.
    """
    # data
    Sigma, _ = sigma_single_spike(p, coef=(1.0, 5.0))
    rng = np.random.default_rng(456)
    X = sample_normal(Sigma=Sigma, n=n, mu=np.zeros(p), seed=int(rng.integers(0, 2**31-1)))

    # references V from basis mixing
    E = generate_basis(p)  # (p,4)
    if r == 1:
        # one reference: mix of e1,e2
        A = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0]])  # (1,4)
    else:
        # two references
        A = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1/np.sqrt(2), 0.0, -1/np.sqrt(2), 0.0],
        ])  # (2,4)
    V = generate_reference_vectors(E, A)  # (p, r)

    # orthonormal=False → just shape check
    U_arg = compute_ARG_PC_subspace(samples=X, reference_vectors=V, spike_number=m, orthonormal=False)
    assert U_arg.shape == (p, m)

    # orthonormal=True → shape + orthonormality check
    U_arg_orth = compute_ARG_PC_subspace(samples=X, reference_vectors=V, spike_number=m, orthonormal=True)
    assert U_arg_orth.shape == (p, m)
    assert is_orthonormal(U_arg_orth)