import numpy as np

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Principal angle (radians) between two directions u, v in R^p:
        angle = arccos(|u^T v| / (||u|| ||v||))
    Robust to tiny norms via clipping.
    """
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    un = u / max(np.linalg.norm(u), 1e-32)
    vn = v / max(np.linalg.norm(v), 1e-32)
    c = float(np.clip(np.abs(np.dot(un, vn)), 0.0, 1.0))
    return float(np.arccos(c))


def compute_principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute principal angles (radians) between two subspaces spanned by columns of U and V.
    Returns angles sorted (θ1 ≤ θ2 ≤ ...), where cos(θi) are the singular values of U^T V.
    Assumes U, V have orthonormal columns.
    """
    # SVD of U^T V: singular values are cosines of principal angles
    S = np.linalg.svd(U.T @ V, full_matrices=False, compute_uv=False)
    S = np.clip(S, 0.0, 1.0)
    return np.arccos(S)  # sorted descending cos -> ascending angles