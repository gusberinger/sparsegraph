import sparsegraph as sg
import numpy as np
import numpy.typing as npt
import random


def estimate_closeness_centrality(
    graph: sg.SparseGraph, *, k: int = 10**4
) -> npt.NDArray[np.float64]:
    r"""
    Finds an estimate of closeness centrality defined by :math:`C_{\textrm{closeness}}(v) = \frac{n-1}{\sum_{u}d(u,v)}.`
    where :math:`d(u,v)` is the shortest path distance between :math:`u` and :math:`v`.

    Parameters
    ----------
    graph:
        A SparseGraph graph.
    k:
        The number of random starting points to use. A larger value will produce a more accurate estimate.

    References
    ----------
    Eppstein and Wang (2000). Fast approximation of centrality. https://doi.org/10.48550/arXiv.cs/0009005
    """
    n = graph.size
    total = np.zeros(n)
    for _ in range(k):
        i = random.randint(0, n - 1)
        sssp = sg.alg.distance_from(graph, i)
        total += (n / (k * (n - 1))) * sssp
    return 1 / total
