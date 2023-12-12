import numpy as np
from tqdm import tqdm
import sparsegraph as sg


def katz_centrality(
    graph: sg.SparseGraph,
    *,
    alpha: float = 0.1,
    beta: float = 1,
    max_iter: int = 10000,
    tol: float = 1.0e-6,
    normalized: bool = True,
    verbose: bool = False,
):
    r"""
    Find Katz centrality defined by :math:`C_{\textrm{Katz}}(i)  = \sum_j  (I_{ij} + \alpha A_{ij} + \alpha^2 A_{ij}^2 + \alpha^3 A_{ij}^3 + \dots)`

    Parameters
    ----------
    graph:
        A SparseGraph graph.
    alpha:
        The attenuation factor. The attenuation factor must satisfy :math:`\alpha < \frac{1}{\max(\lambda_1, \dots, \lambda_n)}` or the algorithm will not converge.
    beta:
        The constant factor. Defaults to 1.
    max_iter:
        The maximum number of iterations to perform before raising a RuntimeError. Defaults to 10,000.
    tol:
        The tolerance for convergence. Defaults to 1.0e-6.
    normalized:
        If ``True`` the centrality scores are normalized.
    verbose:
        If ``True`` a progress bar is displayed during the power iteration.
    """
    A = graph.adjacency.transpose()
    n = graph.size
    e = np.ones((n, 1))
    last = e.copy()

    for _ in tqdm(range(max_iter), disable=not verbose, total=None):
        current = alpha * A.dot(last) + beta * e
        error = sum((abs(current[i] - last[i]) for i in range(n)))
        if error < n * tol:
            centrality = current.flatten().tolist()
            if normalized:
                norm = np.sign(sum(centrality)) * np.linalg.norm(centrality)
                return centrality / norm
            else:
                return centrality
        last = current.copy()

    raise RuntimeError(
        f"Power iteration failed to converge after {max_iter} iterations"
    )
