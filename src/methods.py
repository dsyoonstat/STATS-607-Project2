from typing import Optional, Tuple
import numpy as np

def compute_ARG_PC_subspace(
    samples: np.ndarray,
    reference_vectors: np.ndarray,
    spike_number: int,
    orthonormal: bool = False
) -> np.ndarray:
    """
    Compute ARG (Adaptive Reference Guided) PC subspace basis under a spike covariance structure.
    The code is optimized for high-dimensional setting using the Gram matrix.

    Parameters
    ----------
    samples : (n, p) array
        Rows = observations, columns = features.
    reference_vectors : (p, r) array
        Columns are reference directions in R^p.
    spike_number : int
        Number of spikes (target subspace dimension).
    orthonormal : bool, default False
        If True, QR-orthonormalize the output.

    Returns
    -------
    U_ARG : (p, spike_number) array
        Basis of the ARG PC subspace (orthonormalized if requested).
    """
    X = np.asarray(samples)
    reference_vectors = np.asarray(reference_vectors)

    n, p = X.shape
    p_V, r = reference_vectors.shape
    if p != p_V:
        raise ValueError("Dimension mismatch: samples have p features; reference_vectors must have p rows.")
    if not (1 <= spike_number <= min(n - 1, p)):
        raise ValueError(f"spike_number must be in [1, min(n-1, p)]; got {spike_number}, n={n}, p={p}")

    # 1) Center data
    Xc = X - X.mean(axis=0, keepdims=True)  # (n, p)

    # 2) Compute gram matrix G
    G = (Xc @ Xc.T) / float(n)

    # 3) Full symmetric eigen-decomposition of G (small: n x n)
    w, Q = np.linalg.eigh(G)             # w: (n,), Q: (n, n)
    idx_desc = np.argsort(w)[::-1]       # descending
    w = w[idx_desc]
    Q = Q[:, idx_desc]

    # 4) Nonzero spectrum size k = min(n-1, p) due to centering (rank ≤ n-1)
    k = min(n - 1, p)
    # top-m eigenvalues/vectors of S equal top-m of G
    lam_spike = w[:spike_number].copy()            # (spike_number,)
    Q_spike   = Q[:, :spike_number]                # (n, spike_number)

    # 5) Recover eigenvectors of S: U_spike = Xc^T Q_spike / sqrt(n * lam_spike)
    denom = np.sqrt(np.maximum(lam_spike * float(n), 1e-32))  # (spike_number,)
    U_spike = (Xc.T @ Q_spike) / denom[None, :]               # (p, spike_number)

    # 6) l_tilde = mean of non-spiked eigenvalues among the nonzero spectrum
    if k > spike_number:
        l_tilde = float(np.mean(w[spike_number:k]))
    else:
        l_tilde = 0.0

    # 7) Apply (I - P_V) to U_spike without forming P_V:
    #    (I - P_V)U_spike = U_spike - V * solve(V^T V, V^T U_spike)
    Gv  = reference_vectors.T @ reference_vectors          # (r, r)
    VtU = reference_vectors.T @ U_spike                    # (r, spike_number)
    Y   = _spd_solve(Gv, VtU)                              # (r, spike_number)
    M   = U_spike - reference_vectors @ Y                  # (p, spike_number)

    # 8) Compute U_ARG = (S_m - l_tilde I) M,
    #    with S_m M = U_spike * (diag(lam_spike) * (U_spike^T M))
    UtM   = U_spike.T @ M                                   # (spike_number, spike_number)
    S_m_M = U_spike @ (lam_spike[:, None] * UtM)            # (p, spike_number)
    U_ARG = S_m_M - l_tilde * M                             # (p, spike_number)

    # 9) Orthonormalize if requested
    if orthonormal:
        U_ARG, _ = np.linalg.qr(U_ARG, mode="reduced")

    return U_ARG


# --- helpers ---

def _spd_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve A X = B for X where A is symmetric positive (semi)definite.
    Try Cholesky; fall back to solve; last resort pinv for numerical safety.
    """
    try:
        L = np.linalg.cholesky(A)
        Z = np.linalg.solve(L, B)
        X = np.linalg.solve(L.T, Z)
        return X
    except np.linalg.LinAlgError:
        # Not strictly PD; try generic solve
        try:
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # Very ill-conditioned; use pseudoinverse
            return np.linalg.pinv(A) @ B
        

def compute_PC_subspace(samples: np.ndarray, spike_number: int) -> np.ndarray:
    """
    Compute PC subspace basis (top principal components) using the Gram trick.
    Optimized for high-dimensional settings.

    Parameters
    ----------
    samples : (n, p) array
        Rows are observations, columns are features.
    spike_number : int
        Target subspace dimension.

    Returns
    -------
    U_PC : (p, spike_number) array
        Orthonormal basis of the top-`spike_number` PC subspace in R^p.
    """
    X = np.asarray(samples)
    if X.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got {X.ndim}D.")
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least two observations (n >= 2).")
    max_m = min(n - 1, p)  # rank(Xc) <= n-1 after centering
    if not (1 <= spike_number <= max_m):
        raise ValueError(f"`spike_number` must be in [1, {max_m}] for n={n}, p={p}; got {spike_number}.")

    # 1) Center data
    Xc = X - X.mean(axis=0, keepdims=True)  # (n, p)

    # 2) Compute Gram matrix and do eigen-decomposition
    G = (Xc @ Xc.T) / float(n)              # (n, n), SPSD
    w, Q = np.linalg.eigh(G)                # ascending
    idx = np.argsort(w)[::-1]               # to descending
    w = w[idx]
    Q = Q[:, idx]

    # 3) Take top-`spike_number` nonzero spectrum
    lam = w[:spike_number].copy()           # (spike_number,)
    Qm  = Q[:, :spike_number]               # (n, spike_number)

    # 4) Recover feature-space eigenvectors (principal directions)
    #    U_m = Xc^T Qm / sqrt(n * lam)
    denom = np.sqrt(np.maximum(lam * float(n), 1e-32))  # guard tiny eigvals
    U_m = (Xc.T @ Qm) / denom[None, :]      # (p, spike_number)

    # 5) Re-orthonormalize for numerical stability
    U_PC, _ = np.linalg.qr(U_m, mode="reduced")  # (p, spike_number)
    return U_PC