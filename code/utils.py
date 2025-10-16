import numpy as np
import cvxpy as cp
from scipy.optimize import linear_sum_assignment
from scipy import linalg as scipy_linalg
import warnings

# ============================================================================
# METRICS FOR MATRIX COMPARISON
# ============================================================================

def _frobenius_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    return float(np.linalg.norm(A - B, 'fro'))

def frobenius_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _frobenius_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _frobenius_distance(A, B)

def _spectral_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    return float(np.linalg.norm(A - B, 2))

def spectral_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _spectral_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _spectral_distance(A, B)

def _nuclear_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    _, s, _ = np.linalg.svd(A - B, full_matrices=False)
    return float(np.sum(s))

def nuclear_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _nuclear_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _nuclear_distance(A, B)

def _entrywise_p_distance(A, B, p=2):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    if p == 1:
        return float(np.sum(np.abs(A - B)))
    elif p == 2:
        return float(np.sqrt(np.sum((A - B)**2)))
    elif p == np.inf:
        return float(np.max(np.abs(A - B)))
    else:
        return float(np.sum(np.abs(A - B)**p)**(1/p))

def entrywise_p_distance(A, B, p=2):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _entrywise_p_distance(A[i], B[i], p)
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _entrywise_p_distance(A, B, p)

def _singular_value_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    _, s_A, _ = np.linalg.svd(A, full_matrices=False)
    _, s_B, _ = np.linalg.svd(B, full_matrices=False)
    
    max_len = max(len(s_A), len(s_B))
    s_A_padded = np.pad(s_A, (0, max_len - len(s_A)), 'constant')
    s_B_padded = np.pad(s_B, (0, max_len - len(s_B)), 'constant')
    
    return float(np.linalg.norm(s_A_padded - s_B_padded))

def singular_value_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _singular_value_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _singular_value_distance(A, B)

def _trace_norm_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    trace_A = np.trace(A.T @ A)
    trace_B = np.trace(B.T @ B)
    
    return float(np.abs(np.sqrt(trace_A) - np.sqrt(trace_B)))

def trace_norm_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _trace_norm_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _trace_norm_distance(A, B)

def _procrustes_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)
    
    norm_A = np.linalg.norm(A_centered, 'fro')
    norm_B = np.linalg.norm(B_centered, 'fro')
    
    if norm_A > 1e-10 and norm_B > 1e-10:
        A_normalized = A_centered / norm_A
        B_normalized = B_centered / norm_B
    else:
        A_normalized = A_centered
        B_normalized = B_centered
    
    H = A_normalized.T @ B_normalized
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    B_aligned = B_normalized @ R
    distance = np.linalg.norm(A_normalized - B_aligned, 'fro')
    
    return float(distance)

def procrustes_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _procrustes_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _procrustes_distance(A, B)

def _eigenvalue_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    if A.shape[0] == A.shape[1]:
        # Square matrices: use direct eigenvalues
        gram_A = A
        gram_B = B
    else:
        # Non-square matrices: use A A^T and B B^T to preserve row structure
        gram_A = A @ A.T
        gram_B = B @ B.T
    
    eig_A = np.linalg.eigvals(gram_A)
    eig_B = np.linalg.eigvals(gram_B)
    
    eig_A = eig_A[np.lexsort((eig_A.imag, eig_A.real))]
    eig_B = eig_B[np.lexsort((eig_B.imag, eig_B.real))]
    
    return float(np.linalg.norm(eig_A - eig_B))

def eigenvalue_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _eigenvalue_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _eigenvalue_distance(A, B)

def _condition_number_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cond_A = np.linalg.cond(A)
        cond_B = np.linalg.cond(B)
    
    if np.isinf(cond_A):
        cond_A = 1e16
    if np.isinf(cond_B):
        cond_B = 1e16
    
    return float(np.abs(cond_A - cond_B))

def condition_number_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _condition_number_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _condition_number_distance(A, B)

def _ky_fan_distance(A, B, k=1):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    _, s, _ = np.linalg.svd(A - B, full_matrices=False)
    k = min(k, len(s))
    return float(np.sum(s[:k]))

def ky_fan_distance(A, B, k=1):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _ky_fan_distance(A[i], B[i], k)
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _ky_fan_distance(A, B, k)

def _schatten_p_distance(A, B, p=2):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    _, s, _ = np.linalg.svd(A - B, full_matrices=False)
    
    if p == np.inf:
        return float(np.max(s))
    elif p == 1:
        return float(np.sum(s))
    else:
        return float(np.sum(s**p)**(1/p))

def schatten_p_distance(A, B, p=2):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _schatten_p_distance(A[i], B[i], p)
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _schatten_p_distance(A, B, p)

def _matrix_log_distance(A, B, regularization=1e-12):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    if A.shape[0] == A.shape[1]:
        # Square matrices: use directly with regularization
        A_reg = A + regularization * np.eye(A.shape[0])
        B_reg = B + regularization * np.eye(B.shape[0])
    else:
        # Non-square matrices: use A A^T and B B^T to preserve row structure
        gram_A = A @ A.T + regularization * np.eye(A.shape[0])
        gram_B = B @ B.T + regularization * np.eye(B.shape[0])
        A_reg = gram_A
        B_reg = gram_B
    
    log_A = scipy_linalg.logm(A_reg)
    log_B = scipy_linalg.logm(B_reg)
    
    return float(np.linalg.norm(log_A - log_B, 'fro'))

def matrix_log_distance(A, B, regularization=1e-12):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _matrix_log_distance(A[i], B[i], regularization)
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _matrix_log_distance(A, B, regularization)

def _grassmannian_distance(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)
    
    _, s, _ = np.linalg.svd(Q_A.T @ Q_B, full_matrices=False)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    
    return float(np.linalg.norm(np.sin(angles)))

def grassmannian_distance(A, B):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _grassmannian_distance(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _grassmannian_distance(A, B)

def _bures_distance(A, B, regularization=1e-12):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    
    if A.shape[0] == A.shape[1]:
        # Square matrices: use directly with regularization
        A_reg = A + regularization * np.eye(A.shape[0])
        B_reg = B + regularization * np.eye(B.shape[0])
    else:
        # Non-square matrices: use A A^T and B B^T to preserve row structure
        gram_A = A @ A.T + regularization * np.eye(A.shape[0])
        gram_B = B @ B.T + regularization * np.eye(B.shape[0])
        A_reg = gram_A
        B_reg = gram_B
    
    sqrt_A = scipy_linalg.sqrtm(A_reg)
    sqrt_B = scipy_linalg.sqrtm(B_reg)
    
    product = sqrt_A @ B_reg @ sqrt_A
    sqrt_product = scipy_linalg.sqrtm(product)
    
    trace_A = np.trace(A_reg)
    trace_B = np.trace(B_reg)
    trace_sqrt = 2 * np.trace(sqrt_product).real
    
    distance_squared = trace_A + trace_B - trace_sqrt
    return float(np.sqrt(np.maximum(0, distance_squared)))

def bures_distance(A, B, regularization=1e-12):
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _bures_distance(A[i], B[i], regularization)
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _bures_distance(A, B, regularization)

def vector_correlation(A, B):
    if len(A) != len(B):
        raise ValueError("Vectors A and B must have the same length.")
    
    A_mean = A - np.mean(A)
    B_mean = B - np.mean(B)
    
    numerator = np.sum(A_mean * B_mean)
    denominator = np.sqrt(np.sum(A_mean**2) * np.sum(B_mean**2))

    return numerator / np.maximum(denominator, 1e-10)

def _columnwise_correlation(A, B):
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrices A and B must have the same number of columns.")
    
    correlations = []
    for i in range(A.shape[1]):
        corr = vector_correlation(A[:, i], B[:, i])
        correlations.append(corr)
    metric = np.mean(np.abs(np.array(correlations)))

    return float(metric)

def columnwise_correlation(A, B):
    # process batch dimension if present
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _columnwise_correlation(A[i], B[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _columnwise_correlation(A, B)

def _optimal_correlation(A, B, return_correlations=False):
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrices A and B must have the same number of columns.")

    correlations = np.zeros((A.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            correlations[i, j] = vector_correlation(A[:, i], B[:, j])
    cost_matrix = -np.abs(correlations)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    selected_correlations = correlations[row_ind, col_ind]
    metric = np.mean(np.abs(selected_correlations))
    if return_correlations:
        return selected_correlations, col_ind
    else:
        return float(metric)
    
def optimal_correlation(A, B, return_correlations=False):
    # process batch dimension if present
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] != B.shape[0]:
            raise ValueError("When A and B have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _optimal_correlation(A[i], B[i], return_correlations=return_correlations)
            batch_metrics.append(batch_metric)
        if return_correlations:
            # Return list of tuples (selected_correlations, col_ind) for each batch
            return [(batch_metric[0], batch_metric[1]) for batch_metric in batch_metrics]
        else:
            return float(np.mean(np.array(batch_metrics)))
    else:
        return _optimal_correlation(A, B, return_correlations=return_correlations)

def _internal_optimal_correlation(A):
    if A.shape[1] < 2:
        raise ValueError("Matrix A must have at least two columns for internal optimal correlation.")

    correlations = np.zeros((A.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if i != j:  # Ensure we don't compute the correlation between the same vectors
                correlations[i, j] = vector_correlation(A[:, i], A[:, j])
            else:
                correlations[i, j] = 0.
    cost_matrix = -np.abs(correlations)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    selected_correlations = correlations[row_ind, col_ind]
    metric = np.mean(np.abs(selected_correlations))

    return float(metric)

def internal_optimal_correlation(A):
    # process batch dimension if present
    if len(A.shape) == 3:
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _internal_optimal_correlation(A[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _internal_optimal_correlation(A)

def _weighted_optimal_correlation(A, B, W):
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrices A and B must have the same number of columns.")
    if A.shape[1] != W.shape[1]:
        raise ValueError("Weight vector W must have the same length as the number of columns in A and B.")
    if len(W.shape) < 2:
        raise ValueError("Weight vector W must be 2-dimensional.")
    W_ = np.sum(np.abs(W), axis=0) 
    W_ = W_ / np.sum(W_) 
    selected_correlations, col_ind = optimal_correlation(A, B, return_correlations=True)
    weighted_correlations = selected_correlations * W_[col_ind]
    metric = np.mean(np.abs(weighted_correlations)) * W.shape[1]
    return float(metric)

def weighted_optimal_correlation(A, B, W):
    # process batch dimension if present
    if len(A.shape) == 3 and len(B.shape) == 3 and len(W.shape) == 3:
        if A.shape[0] != B.shape[0] or A.shape[0] != W.shape[0]:
            raise ValueError("When A, B, and W have a batch dimension, they must have the same batch size.")
        batch_metrics = []
        for i in range(A.shape[0]):
            batch_metric = _weighted_optimal_correlation(A[i], B[i], W[i])
            batch_metrics.append(batch_metric)
        return float(np.mean(np.array(batch_metrics)))
    else:
        return _weighted_optimal_correlation(A, B, W)


def util_metric(A, B=None, metric_name="FRO", **kwargs):

    if metric_name == "FRO": # frobenius
        if B is None:
            raise ValueError("frobenius metric requires both A and B")
        return frobenius_distance(A, B)

    elif metric_name == "SPEC": # spectral
        if B is None:
            raise ValueError("spectral metric requires both A and B")
        return spectral_distance(A, B)
    
    elif metric_name == "NUCL": # nuclear
        if B is None:
            raise ValueError("nuclear metric requires both A and B")
        return nuclear_distance(A, B)
    
    elif metric_name == "LP": # entrywise_p
        if B is None:
            raise ValueError("entrywise_p metric requires both A and B")
        p = kwargs.get('p', 2)
        return entrywise_p_distance(A, B, p=p)
    
    elif metric_name == "SV": # singular_value
        if B is None:
            raise ValueError("singular_value metric requires both A and B")
        return singular_value_distance(A, B)
    
    elif metric_name == "TRACE": # trace_norm
        if B is None:
            raise ValueError("trace_norm metric requires both A and B")
        return trace_norm_distance(A, B)
    
    elif metric_name == "PROC": # procrustes
        if B is None:
            raise ValueError("procrustes metric requires both A and B")
        return procrustes_distance(A, B)

    elif metric_name == "EIG": # eigenvalue
        if B is None:
            raise ValueError("eigenvalue metric requires both A and B")
        return eigenvalue_distance(A, B)

    elif metric_name == "COND": # condition_number
        if B is None:
            raise ValueError("condition_number metric requires both A and B")
        return condition_number_distance(A, B)

    elif metric_name == "KY_FAN": # ky_fan
        if B is None:
            raise ValueError("ky_fan metric requires both A and B")
        k = kwargs.get('k', 1)
        return ky_fan_distance(A, B, k=k)

    elif metric_name == "S_P": # schatten_p
        if B is None:
            raise ValueError("schatten_p metric requires both A and B")
        p = kwargs.get('p', 2)
        return schatten_p_distance(A, B, p=p)

    elif metric_name == "M_LOG": # matrix_log
        if B is None:
            raise ValueError("matrix_log metric requires both A and B")
        regularization = kwargs.get('regularization', 1e-12)
        return matrix_log_distance(A, B, regularization=regularization)
    
    elif metric_name == "GRASS": # grassmannian
        if B is None:
            raise ValueError("grassmannian metric requires both A and B")
        return grassmannian_distance(A, B)

    elif metric_name == "BURES": # bures
        if B is None:
            raise ValueError("bures metric requires both A and B")
        regularization = kwargs.get('regularization', 1e-12)
        return bures_distance(A, B, regularization=regularization)
    
    elif metric_name == "COR": # columnwise_correlation
        if B is None:
            raise ValueError("columnwise_correlation metric requires both A and B")
        return columnwise_correlation(A, B)
    
    elif metric_name == "oCOR": # optimal_correlation
        if B is None:
            raise ValueError("optimal_correlation metric requires both A and B")
        return_correlations = kwargs.get('return_correlations', False)
        return optimal_correlation(A, B, return_correlations=return_correlations)
    
    elif metric_name == "weightCOR": # weighted_optimal_correlation
        if B is None:
            raise ValueError("weighted_optimal_correlation metric requires both A and B")
        W = kwargs.get('W')
        if W is None:
            raise ValueError("weighted_optimal_correlation requires W parameter")
        return weighted_optimal_correlation(A, B, W)
    
    elif metric_name == "interCOR": # internal_optimal_correlation
        return internal_optimal_correlation(A)
    
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}. Available metrics: {available_metrics}")

available_metrics = [
            "FRO", "SPEC", "NUCL", "LP", "SV", "TRACE", "PROC", "EIG", "COND",
            "KY_FAN", "S_P", "M_LOG", "GRASS", "BURES", "COR", "oCOR", "weightCOR", "interCOR"
        ]


def solve_with_norm(U, A, norm='fro'):
    if len(U.shape) != 2 or len(A.shape) != 2:
        raise ValueError("Matrices U and A must be 2-dimensional.")
    m = U.shape[1]
    p = A.shape[1]
    E = cp.Variable((p, m))
    residual = U - A @ E

    if norm == 'fro':
        objective = cp.Minimize(cp.norm(residual, 'fro'))
    elif norm == 'l1':
        objective = cp.Minimize(cp.norm(residual, 1))
    elif norm == 'linf':
        objective = cp.Minimize(cp.norm(residual, 'inf'))
    elif norm == '2':
        objective = cp.Minimize(cp.norm(residual, 2))
    elif norm == 'nuc':
        objective = cp.Minimize(cp.norm(residual, 'nuc'))
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    
    problem = cp.Problem(objective)
    problem.solve()
    return E.value
# A = np.array([[1, 10, 10], [1, 12, 4]]).T
# B = np.array([[1, 12, 4], [1, 10, 10]]).T
# W = np.array([[10, 1, 5], [0.1, 8, 6]]).T
# print("Column-wise correlation:", columnwise_correlation(A, B))
# print("Optimal correlation:", optimal_correlation(A, B))
# print("Internal optimal correlation:", internal_optimal_correlation(A))
# print("Weighted optimal correlation:", weighted_optimal_correlation(A, B, W))