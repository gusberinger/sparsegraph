import numpy as np
from typing import Generic, List, TypeVar
from scipy import sparse
import networkx as nx

T = TypeVar("T")


class SparseGraph(Generic[T]):
    def __init__(self, adjacency: sparse.csr_matrix, labels: List[T]) -> None:
        self.size = adjacency.shape[0]
        self.adjacency = adjacency
        if len(labels) != self.size:
            raise ValueError("Number of labels must match number of nodes in graph.")
        self.labels = labels
        self._in_degree = None
        self._out_degree = None

    @staticmethod
    def __delete_from_csr__(
        mat: sparse.csr_matrix, indices: List[int]
    ) -> sparse.csr_matrix:
        """
        Parameters
        ---
        mat:
            The csr sparse adjacency matrix of the graph.
        indices:
            The indices of the nodes in the graph that will be removed.

        Returns
        ---
        mat:
            The sparse adjacency matrix of the graph with the nodes removed.
        """
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Matrix must be square.")

        mask = np.ones(mat.shape[0], dtype=bool)
        mask[indices] = False
        return mat[mask][:, mask]

    def get_largest_component(
        self, directed: bool = True, connection: str = "strong"
    ) -> "SparseGraph":
        """
        Parameters
        ---
        matrix:
            The sparse adjacency matrix of the graph.
        labels:
            The labels of each node in the graph.
            The index of the list corresponds to the index in the adjacency matrix.
        direct:
            If ``True`` the graph is treated as directed.
        connection:
            If ``"strong"``, the connected components will all be strongly connected together.
            If ``"weak"`` the connected components will be weakly connected.
            If ``directed == False`` the parameter is ignored.

        Returns
        ---
        sub_matrix:
            The sparse adjacency matrix of the largest strongly connected subgraph.
        sub_labels:
            The labels of each node in the subgraph.
        """
        _, component_labels = sparse.csgraph.connected_components(
            self.adjacency, directed=directed, connection=connection
        )
        unique_component_labels, count = np.unique(component_labels, return_counts=True)
        largest = unique_component_labels[np.argmax(count)]
        indices = np.where(component_labels == largest)[0]
        unconnected_indices = list(set(range(self.adjacency.shape[0])) - set(indices))
        subgraph = self.remove_indices(unconnected_indices)
        return subgraph

    def compute_degree(self) -> None:
        """
        Fills in the degrees for self.in_degree and self.out_degree
        """
        self._in_degree = np.array(np.sum(self.adjacency, axis=0))[0]
        self._out_degree = np.array(np.sum(self.adjacency, axis=1).transpose())[0]

    def in_degree(self, index: int) -> int:
        if self._in_degree is None:
            raise ValueError(
                "Degrees not computed. Must call self.compute_degree on instance first."
            )
        else:
            return self._in_degree[index]

    def out_degree(self, index: int) -> int:
        if self._out_degree is None:
            raise ValueError(
                "Degrees not computed. Must call self.compute_degree on instance first."
            )
        else:
            return self._out_degree[index]

    def outgoing_neighbors(self, index: int) -> np.ndarray:
        nodes = self.adjacency[index, :].toarray()[0]
        return np.nonzero(nodes)[0]

    def incoming_neighbors(self, index: int) -> np.ndarray:
        nodes = self.adjacency[:, index].toarray()[0]
        return np.nonzero(nodes)[0]

    def remove_indices(self, indices: List[int]) -> "SparseGraph":
        new_adjacency = self.__delete_from_csr__(self.adjacency, indices)
        new_labels = [label for i, label in enumerate(self.labels) if i not in indices]
        return SparseGraph(new_adjacency, new_labels)

    def get_value(self, index: int) -> T:
        return self.labels[index]

    def to_networkx(self):
        return nx.from_scipy_sparse_array(self.adjacency, create_using=nx.DiGraph)
