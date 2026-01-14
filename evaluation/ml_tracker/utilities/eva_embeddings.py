from typing import Sequence, Optional, Dict

import numpy as np

import warnings

from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

MetricDict = Dict[str, float]

def _sample_pairs(n: int, num_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample (i, j) pairs with i != j, length = num_pairs.
    """
    if n < 2 or num_pairs <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    i = rng.integers(0, n, size=num_pairs, dtype=np.int64)
    j = rng.integers(0, n, size=num_pairs, dtype=np.int64)

    # Ensure i != j by shifting any collisions; keeps uniformity in practice.
    mask = (i == j)
    if np.any(mask):
        j[mask] = (j[mask] + 1) % n
    return i, j


def _safe_row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize rows of X with numerical safety.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _distribution_scale_metrics(
    X: np.ndarray,
    *,
    num_pairs: int = 20_000,
    random_state: Optional[int] = None,
    cosine_on_unit_sphere: bool = True,
    eps: float = 1e-12,
) -> MetricDict:
    """
    Distribution/scale probes:
      - vector norm statistics
      - sampled pairwise cosine & Euclidean distance statistics (memory-safe)

    Pairs are sampled uniformly over ordered pairs with i != j.
    """
    N = X.shape[0]
    metrics: MetricDict = {}
    if N == 0:
        return metrics

    # Norm stats
    norms = np.linalg.norm(X, axis=1)
    metrics.update(
        {
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std(ddof=0)),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
        }
    )

    if N >= 2:
        # Ordered-pair cap for consistency with sampling scheme
        max_pairs = N * (N - 1)
        m = int(min(num_pairs, max_pairs))
        if m > 0:
            rng = np.random.default_rng(random_state)
            i, j = _sample_pairs(N, m, rng)

            # Precompute for distance identity
            row_norm2 = np.einsum("nd,nd->n", X, X)
            dot_ij = np.einsum("nd,nd->n", X[i], X[j])

            # Cosine similarities
            if cosine_on_unit_sphere:
                Z = _safe_row_normalize(X, eps=eps)
                cos = np.einsum("nd,nd->n", Z[i], Z[j])
            else:
                ni = np.sqrt(row_norm2[i])
                nj = np.sqrt(row_norm2[j])
                denom = np.maximum(ni * nj, eps)
                cos = dot_ij / denom

            # Euclidean distances via ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            sq = np.maximum(row_norm2[i] + row_norm2[j] - 2.0 * dot_ij, 0.0)
            d = np.sqrt(sq)

            def q(arr: np.ndarray, qv: float) -> float:
                return float(np.quantile(arr, qv)) if arr.size else float("nan")

            metrics.update(
                {
                    "cos_mean": float(cos.mean()),
                    "cos_std": float(cos.std(ddof=0)),
                    "cos_q05": q(cos, 0.05),
                    "cos_q50": q(cos, 0.50),
                    "cos_q95": q(cos, 0.95),
                    "dist_mean": float(d.mean()),
                    "dist_std": float(d.std(ddof=0)),
                    "dist_q05": q(d, 0.05),
                    "dist_q50": q(d, 0.50),
                    "dist_q95": q(d, 0.95),
                }
            )

    return metrics


def _geometry_spectrum_metrics(
    X: np.ndarray,
    *,
    center: bool = True,
    explained_var_cut: float = 0.90,
    eps: float = 1e-12,
) -> MetricDict:
    """
    Geometry/spectrum metrics from covariance eigenvalues via SVD:

      • Effective rank (spectral entropy): exp(-∑ p_i log p_i), p_i = λ_i / ∑ λ
      • Participation ratio: (∑ λ)^2 / ∑ λ^2
      • Isotropy ratio: λ_max / mean(λ)
      • Explained-variance@k: smallest k reaching `explained_var_cut`
      • Condition number: λ_max / λ_min

    Uses λ = S^2 / (N - 1), where S are singular values of (optionally centered) X.
    """
    X = np.asarray(X, dtype=np.float64)  # promote for stable SVD/entropy

    N, D = X.shape
    if N < 2 or D == 0:
        warnings.warn("Not enough samples/features to compute spectrum metrics.", RuntimeWarning)
        return {}

    Xc = X - X.mean(axis=0, keepdims=True) if center else X

    try:
        S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        Xj = Xc + 1e-8 * np.random.default_rng(0).standard_normal(size=Xc.shape)
        S = np.linalg.svd(Xj, full_matrices=False, compute_uv=False)

    eig = (S ** 2) / max(N - 1, 1)  # eigenvalues of sample covariance
    total = float(np.sum(eig))

    if not np.isfinite(total) or total <= eps:
        warnings.warn("Covariance spectrum is (near) zero; geometry metrics are ill-defined.", RuntimeWarning)
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "isotropy_ratio": float("nan"),
            f"explained_k@{int(round(explained_var_cut * 100))}%": 0,
            "cond_number": float("nan"),
        }

    p = eig / total
    ent = -np.sum(p * np.log(np.maximum(p, eps)))
    effective_rank = float(np.exp(ent))

    participation_ratio = float((total ** 2) / max(float(np.sum(eig ** 2)), eps))

    lam_max = float(np.max(eig))
    lam_min = float(np.min(eig))
    lam_mean = float(np.mean(eig))
    isotropy_ratio = float(lam_max / max(lam_mean, eps))

    eig_sorted = np.sort(eig)[::-1]
    csum = np.cumsum(eig_sorted)
    target = explained_var_cut * total
    k = int(np.searchsorted(csum, target, side="left") + 1)
    k = int(np.clip(k, 1, eig_sorted.size))

    cond_number = float(lam_max / max(lam_min, eps))

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "isotropy_ratio": isotropy_ratio,
        f"explained_k@{int(round(explained_var_cut * 100))}%": k,
        "cond_number": cond_number,
    }

def _uniformity_metric(
    X: np.ndarray,
    *,
    t: float = 2.0,
    num_pairs: int = 20_000,
    normalize: bool = True,
    random_state: Optional[int] = None,
    eps: float = 1e-12,
) -> MetricDict:
    """
    Uniformity on the hypersphere (Wang & Isola, 2020):
        U_t(Z) = log E_{i≠j} [ exp( -t * ||z_i - z_j||^2 ) ]

    When `normalize=True`, distances are computed on L2-normalized rows (recommended).
    Pairs are sampled uniformly over ordered pairs with i != j.
    """
    N = X.shape[0]
    if N < 2:
        warnings.warn("Need at least 2 samples to compute uniformity.", RuntimeWarning)
        return {}

    max_pairs = N * (N - 1)  # ordered pairs cap
    m = int(min(num_pairs, max_pairs))
    if m <= 0:
        return {}

    rng = np.random.default_rng(random_state)
    i, j = _sample_pairs(N, m, rng)

    if normalize:
        Z = _safe_row_normalize(X, eps=eps)
        # On unit sphere: ||x - y||^2 = 2 - 2 * (x·y)
        cos = np.einsum("nd,nd->n", Z[i], Z[j])
        sq = np.maximum(2.0 - 2.0 * cos, 0.0)
    else:
        row_norm2 = np.einsum("nd,nd->n", X, X)
        dot_ij = np.einsum("nd,nd->n", X[i], X[j])
        sq = np.maximum(row_norm2[i] + row_norm2[j] - 2.0 * dot_ij, 0.0)

    # Stable log-mean-exp
    a = -t * sq
    a_max = float(np.max(a))
    u = a_max + float(np.log(np.mean(np.exp(a - a_max))))
    return {"uniformity_log_mean_exp": u}


def _retrieval_metrics(
    X: np.ndarray, y: np.ndarray, top_k: Sequence[int] = (1, 5, 10)
) -> MetricDict:
    """Compute top‑k retrieval accuracy & mean‑AP on pooled clip embeddings.

    *Self‑similarities are excluded* from MAP by dropping the query item from
    both `y_true` and `scores`, preventing –∞ issues in `average_precision_score`.
    """
    sim = cosine_similarity(X).astype(np.float64)

    # Set diagonal to very small finite value so argsort logic stays intact but finite
    np.fill_diagonal(sim, -1.0)

    # Top‑k hits --------------------------------------------------------------
    indices = np.argsort(sim, axis=1)[:, ::-1]  # descending similarity
    metrics: MetricDict = {}
    for k in top_k:
        hits = np.any(y[indices[:, :k]] == y[:, None], axis=1)
        metrics[f"retrieval_top{k}_acc"] = float(np.mean(hits))

    # Mean Average Precision --------------------------------------------------
    average_precisions = []
    for i, (y_i, sim_i) in enumerate(zip(y, sim)):
        # Binary relevance (exclude self)
        rel = (y == y_i).astype(int)
        rel[i] = 0  # self‑match excluded

        scores = sim_i
        scores = np.delete(scores, i)
        rel = np.delete(rel, i)

        # Skip if no other positive exists for this query
        if rel.sum() == 0:
            continue

        try:
            ap = average_precision_score(rel, scores)
            average_precisions.append(ap)
        except ValueError:
            # All scores identical? fall back to 0‑AP
            average_precisions.append(0.0)

    metrics["retrieval_map"] = float(np.mean(average_precisions)) if average_precisions else float("nan")
    return metrics


def _center_columns(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray, *, center: bool = True) -> float:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (same examples)")

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if center:
        X = _center_columns(X)
        Y = _center_columns(Y)

    C  = X.T @ Y
    XX = X.T @ X
    YY = Y.T @ Y

    numer = np.linalg.norm(C, ord="fro") ** 2
    denom = np.linalg.norm(XX, ord="fro") * np.linalg.norm(YY, ord="fro")
    if denom == 0.0:
        return float("nan")
    val = numer / denom
    return {"linear_cka": float(val)}

# ---------- RBF CKA (nonlinear; O(N^2) memory/time) ----------

def _pairwise_sq_dists(Z: np.ndarray) -> np.ndarray:
    norms = np.sum(Z * Z, axis=1, keepdims=True)
    G = Z @ Z.T
    D = norms + norms.T - 2.0 * G
    np.maximum(D, 0.0, out=D)
    return D

def _center_gram(K: np.ndarray) -> np.ndarray:
    mean_row = K.mean(axis=0, keepdims=True)
    mean_col = K.mean(axis=1, keepdims=True)
    mean_all = K.mean()
    return K - mean_row - mean_col + mean_all

def rbf_cka(X: np.ndarray, Y: np.ndarray, *, gamma: float | str = "median") -> float:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (same examples)")

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    DX = _pairwise_sq_dists(X)
    DY = _pairwise_sq_dists(Y)

    if isinstance(gamma, str):
        if gamma != "median":
            raise ValueError("rbf_cka: gamma must be 'median' or a positive float")
        triX = DX[np.triu_indices_from(DX, k=1)]
        medX = np.median(np.sqrt(triX)) if triX.size else 0.0
        if medX <= 0.0:
            return float("nan")
        gamma_x = 1.0 / (2.0 * medX * medX)

        triY = DY[np.triu_indices_from(DY, k=1)]
        medY = np.median(np.sqrt(triY)) if triY.size else 0.0
        if medY <= 0.0:
            return float("nan")
        gamma_y = 1.0 / (2.0 * medY * medY)
    else:
        if not (isinstance(gamma, (int, float)) and gamma > 0):
            raise ValueError("rbf_cka: gamma must be 'median' or a positive float")
        gamma_x = gamma_y = float(gamma)

    # FIX: remove invalid dtype kwarg; inputs are already float64
    K = np.exp(-gamma_x * DX)
    L = np.exp(-gamma_y * DY)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    numer = (Kc * Lc).sum()
    denom = np.sqrt((Kc * Kc).sum()) * np.sqrt((Lc * Lc).sum())
    if denom == 0.0:
        return float("nan")
    val = numer / denom
    return {"rbf_cka": float(val)}
