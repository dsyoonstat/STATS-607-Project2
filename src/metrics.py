# metrics.py
# - compute_principal_angles(U, V)

import numpy as np

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