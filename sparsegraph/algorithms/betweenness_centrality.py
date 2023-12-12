import numpy as np
from tqdm import tqdm
import sparsegraph as sg
from collections import deque


def betweenness_centrality(
    graph: sg.SparseGraph,
    *,
    normalized=False,
    verbose=False,
):
    r"""
    Finds the betweenness centrality defined by :math:`C_{\textrm{betweenness}}(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}` where :math:`\sigma_{st}` is the number of shortest paths from :math:`s` to :math:`t` and :math:`\sigma_{st}(i)` is the number of shortest paths from :math:`s` to :math:`t` that pass through :math:`i`.

    Parameters
    ----------
    graph:
        The sparse adjacency matrix of the graph.

    normalized:
        If ``True`` the centrality scores are normalized.
    
    verbose:
        If ``True`` a progress bar is displayed during the power iteration.

    References
    ----------
    Brandes (2000). A faster algorithm for betweenness centrality, https://doi.org/10.1080/0022250X.2001.9990249.
    """
    indices = graph.adjacency.indices
    indptr = graph.adjacency.indptr
    size = graph.adjacency.shape[0]
    scores = np.zeros(size)

    for s in tqdm(range(size), disable=not verbose):
        S = []
        P = [[] for _ in range(size)]
        sigma = np.zeros(size)
        sigma[s] = 1
        d = np.ones(size) * -1
        d[s] = 0
        Q = deque()
        Q.append(s)
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in indices[indptr[v] : indptr[v + 1]]:
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = np.zeros(size)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] = delta[v] + sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                scores[w] += delta[w]

    scores /= 2

    if normalized:
        return (scores / ((size - 1) * (size - 2))) * 2
    else:
        return scores
