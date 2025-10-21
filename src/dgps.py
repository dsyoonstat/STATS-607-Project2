# dgps.py
# - generate_basis(p)
# - sigma_single_spike(p, coefs)
# - sigma_multi_spike(p, coefs)
# - sample_normal(Sigma, n, mu, seed)
# - sample_t(Sigma, nu, n, mu, seed)
# - generate_reference_vectors(E, A)

from typing import Optional, Tuple
import numpy as np
from numpy.linalg import cholesky


# -------- Basis vectors generator --------

def generate_basis(p: int) -> np.ndarray:
    """
    Return a (p, 4) matrix whose columns are e1, e2, e3, e4 (each of length p).
    p must be divisible by 4.

    e1 = (1/sqrt(p))*[+ + + +]
    e2 = (1/sqrt(p))*[+ + - -]
    e3 = (1/sqrt(p))*[+ - - +]
    e4 = (1/sqrt(p))*[+ - + -]
    """
    if p % 4 != 0:
        raise ValueError("p must be divisible by 4.")
    q = p // 4
    s = 1.0 / np.sqrt(p)
    ones = np.ones
    e1 = s * np.concatenate([ones(q),  ones(q),  ones(q),  ones(q)])
    e2 = s * np.concatenate([ones(q),  ones(q), -ones(q), -ones(q)])
    e3 = s * np.concatenate([ones(q), -ones(q), -ones(q),  ones(q)])
    e4 = s * np.concatenate([ones(q), -ones(q),  ones(q), -ones(q)])
    return np.column_stack([e1, e2, e3, e4])  # shape (p, 4)


# -------- Covariance builders --------

def sigma_single_spike(p: int, coef: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sigma = c1 * p * e1 e1^T + c2 * I_p  (single-spike)
    Returns (Sigma, e1), where e1 is the principal spike (length p).
    """
    E = generate_basis(p)
    e1 = E[:, [0]]
    c1, c2 = coef
    Sigma = c1 * p * np.outer(e1, e1) + c2 * np.eye(p)
    return Sigma, e1


def sigma_multi_spike(
    p: int,
    coef: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sigma = c1 * p * e1 e1^T + c2 * p * e2 e2^T + c3 * I_p  (two spikes)
    Returns (Sigma, U_m), where U_m = [e1, e2] has shape (p, 2).
    """
    E = generate_basis(p)
    e1, e2 = E[:, 0], E[:, 1]
    c1, c2, c3 = coef
    Sigma = c1 * p * np.outer(e1, e1) + c2 * p * np.outer(e2, e2) + c3 * np.eye(p)
    U_m = np.column_stack([e1, e2])
    return Sigma, U_m


# -------- Samplers --------

def sample_normal(
    Sigma: np.ndarray,
    n: int,
    mu: Optional[np.ndarray] = None,
    seed: Optional[int] = 725
) -> np.ndarray:
    """
    Draw X ~ N(mu, Sigma), shape (n, p).
    If mu is None, uses zero mean (0_p).
    """
    p = Sigma.shape[0]
    if mu is None:
        mu = np.zeros(p)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, Sigma, size=n)


def sample_t(
    Sigma: np.ndarray,
    nu: int,
    n: int,
    mu: Optional[np.ndarray] = None,
    seed: Optional[int] = 725
) -> np.ndarray:
    """
    Draw X from multivariate t_nu(mu, S) with *population* Cov(X)=Sigma (nu>2).
    Matches MATLAB: mvtrnd((nu-2)/nu * Sigma, nu, n)
    Returns array of shape (n, p).
    """
    if nu <= 2:
        raise ValueError("nu must be > 2 for finite covariance.")
    p = Sigma.shape[0]
    if mu is None:
        mu = np.zeros(p)
    rng = np.random.default_rng(seed)
    # Choose S so that Cov[X] = Sigma
    S = ((nu - 2) / nu) * Sigma
    L = cholesky(S)
    Z = rng.normal(size=(n, p))
    w = rng.chisquare(df=nu, size=n)  # shape (n,)
    T = (Z @ L.T) / np.sqrt(w / nu)[:, None]  # each row scaled by sqrt(nu/w)
    return mu[None, :] + T


# -------- Reference vector generator --------

def generate_reference_vectors(E: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Multi-reference:
      E: (p, 4) matrix from generate_basis (columns are e1,e2,e3,e4)
      A: (r, 4) weight matrix; each row defines a reference vector as a mix of e1..e4
    Returns V of shape (p, r) whose columns are v1..vr.
    """
    # E (p x 4) @ A^T (4 x k) -> (p x k)
    return E @ A.T