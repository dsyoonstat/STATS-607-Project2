import numpy as np
import pytest
from src.metrics import compute_principal_angles


def test_identical_vectors():
    """If U and V span the same 1D direction, principal angle should be 0."""
    u = np.array([[1.0], [0.0]])   # (2,1)
    v = np.array([[1.0], [0.0]])   # identical direction
    theta = compute_principal_angles(u, v)
    assert np.allclose(theta, 0.0, atol=1e-12)


def test_opposite_vectors():
    """If U and V are opposite directions, the angle is also 0 because cos(θ)=|uᵀv|."""
    u = np.array([[1.0], [0.0]])
    v = np.array([[-1.0], [0.0]])
    theta = compute_principal_angles(u, v)
    # principal angles are always in [0, π/2]
    assert np.allclose(theta, 0.0, atol=1e-12)


def test_orthogonal_vectors():
    """If U and V are orthogonal, principal angle should be π/2."""
    u = np.array([[1.0], [0.0]])
    v = np.array([[0.0], [1.0]])
    theta = compute_principal_angles(u, v)
    assert np.allclose(theta, np.pi / 2, atol=1e-12)


def test_45_degree():
    """If V is rotated 45° from U, the principal angle should be π/4."""
    u = np.array([[1.0], [0.0]])
    v = np.array([[np.sqrt(2)/2], [np.sqrt(2)/2]])
    theta = compute_principal_angles(u, v)
    assert np.allclose(theta, np.pi / 4, atol=1e-12)


def test_partial_overlap():
    """
    For two 2D subspaces in R^3:
    U = span(e1, e2), V = span(e1, e3)
    -> One shared direction (angle 0), one orthogonal (π/2).
    """
    U = np.eye(3)[:, :2]  # [[1,0],[0,1],[0,0]]
    V = np.eye(3)[:, [0, 2]]  # [[1,0],[0,0],[0,1]]
    theta = compute_principal_angles(U, V)
    expected = np.array([0.0, np.pi / 2])
    assert np.allclose(np.sort(theta), expected, atol=1e-12)


def test_invariant_to_rotation():
    """
    Principal angles are invariant under global orthogonal transformations:
    if Q is orthogonal, then angles(U, V) == angles(Q@U, Q@V).
    """
    rng = np.random.default_rng(42)
    U = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    V = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    Q = np.linalg.qr(rng.normal(size=(5, 5)))[0]

    theta1 = compute_principal_angles(U, V)
    theta2 = compute_principal_angles(Q @ U, Q @ V)

    assert np.allclose(theta1, theta2, atol=1e-12)