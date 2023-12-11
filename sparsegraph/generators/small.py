import numpy as np
import sparsegraph as sg
import scipy.sparse as sp


def house_graph() -> sg.SparseGraph:
    house_graph_adj = sp.csr_array(
        (
            np.ones(12),
            np.array([1, 2, 0, 3, 0, 3, 4, 1, 2, 4, 2, 3]),
            np.array([0, 2, 4, 7, 10, 12]),
        ),
        shape=(5, 5),
    )
    return sg.SparseGraph(house_graph_adj, labels=["A", "B", "C", "D", "E"])
