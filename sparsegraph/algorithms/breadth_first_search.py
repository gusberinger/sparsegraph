import numpy as np
from tqdm import tqdm
import sparsegraph as sg
import numpy.typing as npt


def distance_from(
    graph: sg.SparseGraph, start_node_idx: int, *, verbose=False
) -> npt.NDArray[np.int64]:
    """
    Breadth first search to find the distance from a start node to all other nodes in the graph.
    """
    N = graph.adjacency.shape[0]
    predecessors = np.empty(N, dtype=int)
    node_list = np.empty(N, dtype=int)
    levels: npt.NDArray[np.int64] = np.empty(N, dtype=int)
    indptr = graph.adjacency.indptr
    indices = graph.adjacency.indices

    predecessors.fill(-1)
    node_list.fill(-1)
    levels.fill(-1)

    node_list[0] = start_node_idx
    levels[start_node_idx] = 0
    i_nl = 0
    i_nl_end = 1

    pbar = tqdm(total=N, disable=not verbose)
    while i_nl < i_nl_end:
        pnode = node_list[i_nl]
        pbar.update(1)
        for i in range(indptr[pnode], indptr[pnode + 1]):
            cnode = indices[i]
            if cnode == start_node_idx:
                continue
            elif predecessors[cnode] == -1:
                node_list[i_nl_end] = cnode
                predecessors[cnode] = pnode
                levels[cnode] = levels[pnode] + 1
                i_nl_end += 1
        i_nl += 1

    pbar.close()
    return levels
