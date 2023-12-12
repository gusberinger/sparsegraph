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
