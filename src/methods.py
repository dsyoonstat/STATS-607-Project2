# methods.py
# - compute_ARG_PC_subspace(samples, reference_vectors, spike_number, orthonormal)
# - compute_PC_subspace(samples, spike_number)
# - compute_negative_ridge_discriminants(samples, reference_vectors, spike_number, normalize)

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
    The code is optimized for high-dimensional setting using the Gram matrix.

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


def compute_negative_ridge_discriminants(
    samples: np.ndarray,
    reference_vectors: np.ndarray,
    spike_number: int,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute negatively ridged discriminant vector d_j for each reference vector v_j using:
        d_j := - l_tilde * (S_m - l_tilde I_p)^{-1} v_j,
      where S_m = sum_{i=1}^m \hat{lambda}_i \hat{u}_i \hat{u}_i^T
            l_tilde = mean of non-spike eigenvalues among the nonzero spectrum.

    This implementation uses the dual (Gram) approach and a spectral formula that
    avoids forming/inverting any p×p matrices:
        Let U ∈ R^{p×m}, Λ = diag(lam) with top-m eigenpairs of S,
        a := U^T V,  R := (I - U U^T) V,
        denom := lam - l_tilde  (elementwise),
        then  D = R - l_tilde * U @ (a / denom).

    Parameters
    ----------
    samples : (n, p) ndarray
        Data matrix (rows = observations).
    reference_vectors : (p, r) ndarray
        Columns are reference directions v_j in R^p.
    spike_number : int
        Target subspace dimension m.
    normalize : bool, default False
        If True, normalize each column of the output to unit L2 norm.

    Returns
    -------
    D : (p, r) ndarray
        Columns are d_j for each reference vector v_j.

    Notes
    -----
    - When l_tilde == 0, the formula reduces to D = (I - U U^T) V automatically.
    - We guard tiny denominators (lam_i - l_tilde) with a small epsilon to avoid blow-ups.
    """
    X = np.asarray(samples)
    V = np.asarray(reference_vectors)

    if X.ndim != 2 or V.ndim != 2:
        raise ValueError("`samples` and `reference_vectors` must be 2D arrays.")
    n, p = X.shape
    pV, r = V.shape
    if pV != p:
        raise ValueError("Dimension mismatch: `reference_vectors` must have p rows to match `samples`.")
    if n < 2:
        raise ValueError("Need at least two observations (n >= 2).")
    if not (1 <= spike_number <= min(n - 1, p)):
        raise ValueError(f"`spike_number` must be in [1, {min(n-1, p)}]; got {spike_number}.")

    # 1) Center
    Xc = X - X.mean(axis=0, keepdims=True)  # (n, p)

    # 2) Gram matrix and eigendecomposition (ascending from eigh)
    G = (Xc @ Xc.T) / float(n)              # (n, n)
    w, Q = np.linalg.eigh(G)                # w ascending
    idx = np.argsort(w)[::-1]               # to descending
    w = w[idx]
    Q = Q[:, idx]

    # 3) Nonzero spectrum size and top-m eigenpairs
    k = min(n - 1, p)
    lam = w[:spike_number].copy()           # (m,)
    Qm  = Q[:, :spike_number]               # (n, m)

    # 4) Recover U (feature-space eigenvectors of S)
    #    U = Xc^T Qm / sqrt(n * lam)
    denom = np.sqrt(np.maximum(lam * float(n), 1e-32))
    U = (Xc.T @ Qm) / denom[None, :]        # (p, m)

    # 5) l_tilde = mean of non-spike eigenvalues among the nonzero spectrum
    l_tilde = float(np.mean(w[spike_number:k])) if k > spike_number else 0.0

    # 6) Compute D using spectral formula: D = R - l_tilde * U @ (a / (lam - l_tilde))
    #    a = U^T V, R = (I - U U^T) V
    a = U.T @ V                              # (m, r)
    R = V - U @ a                            # (p, r)

    if l_tilde == 0.0:
        D = R.copy()                         # degenerates nicely
    else:
        denom2 = np.maximum(lam - l_tilde, 1e-32)  # (m,), guard tiny denominators
        Ua = U @ (a / denom2[:, None])       # (p, r)
        D = R - l_tilde * Ua                 # (p, r)

    if normalize:
        # column-wise L2 normalization with safety
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        D = D / norms

    return D