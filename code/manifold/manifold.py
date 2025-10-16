import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar
import warnings


def simple_two_NN(data, k=2, metric='euclidean', n_jobs=None):
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    
    if data.shape[0] < k + 1:
        raise ValueError(f"Not enough samples. Need at least {k + 1} samples, got {data.shape[0]}")
    
    n_samples, n_features = data.shape

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=n_jobs)
    nbrs.fit(data)
    
    distances, indices = nbrs.kneighbors(data)
    
    distances = distances[:, 1:]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratios = distances[:, 1] / distances[:, 0]
    
    valid_mask = np.isfinite(ratios) & (ratios > 0) & (distances[:, 0] > 0)
    ratios = ratios[valid_mask]
    
    if len(ratios) == 0:
        raise ValueError("No valid distance ratios found. Data might be degenerate.")
    
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios)
    
    if mean_log_ratio <= 0:
        warnings.warn("Mean log ratio is non-positive. Data might be degenerate or have very low intrinsic dimension.")
        return 1.0
    
    intrinsic_dim = 1.0 / mean_log_ratio
    return intrinsic_dim


def two_NN(data, k=2, metric='euclidean', n_jobs=None):
    """
    References:
    -----------
    Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
    "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"
    Scientific Reports
    """
    
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    
    if data.shape[0] < k + 1:
        raise ValueError(f"Not enough samples. Need at least {k + 1} samples, got {data.shape[0]}")
    
    n_samples, n_features = data.shape
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=n_jobs)
    nbrs.fit(data)
    
    distances, indices = nbrs.kneighbors(data)
    
    distances = distances[:, 1:]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratios = distances[:, 1] / distances[:, 0]
    
    valid_mask = np.isfinite(ratios) & (ratios > 1.0) & (distances[:, 0] > 1e-15)
    ratios = ratios[valid_mask]
    
    if len(ratios) < 10:
        raise ValueError(f"Too few valid ratios ({len(ratios)}). Need at least 10 for reliable fitting.")
    
    sorted_ratios = np.sort(ratios)
    N = len(sorted_ratios)
    
    ranks = np.arange(1, N + 1)
    
    
    empirical_cdf = ranks / N
    log_ratios = np.log(sorted_ratios)
    log_survival = np.log(1 - empirical_cdf)
    
    valid_fit_mask = (1 - empirical_cdf) > 0.05
    
    if np.sum(valid_fit_mask) < 5:
        raise ValueError("Too few points for reliable linear fitting")
    
    x_fit = log_ratios[valid_fit_mask]  # log(μ_i)
    y_fit = log_survival[valid_fit_mask]  # log(1 - i/N)
    
    A = np.vstack([x_fit, np.ones(len(x_fit))]).T
    slope, intercept = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    
    intrinsic_dim = -slope
    
    if intrinsic_dim <= 0:
        warnings.warn(f"Negative intrinsic dimension ({intrinsic_dim:.3f}) - may indicate issues with data or method")
        intrinsic_dim = abs(intrinsic_dim) 
    
    return float(intrinsic_dim)


def compute_block_means(data, block_size=100):
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    
    n_samples, n_features = data.shape
    
    if n_samples % block_size != 0:
        raise ValueError(f"Number of samples ({n_samples}) must be divisible by block_size ({block_size})")
    
    n_blocks = n_samples // block_size
    
    reshaped = data.reshape(n_blocks, block_size, n_features)
    block_means = np.mean(reshaped, axis=1)
    
    return block_means
