import sparsegraph as sg
import numpy as np
import numpy.typing as npt
import random


def estimate_closeness_centrality(
    graph: sg.SparseGraph, k: int
) -> npt.NDArray[np.float64]:
    """
    Implements https://doi.org/10.48550/arXiv.cs/0009005 Eppstein and Wang (2000)
    """
    n = graph.size
    total = np.zeros(n)
    for _ in range(k):
        i = random.randint(0, n - 1)
        sssp = sg.alg.distance_from(graph, i)
        total += (n / (k * (n - 1))) * sssp
    return 1 / total
