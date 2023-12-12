import numpy as np
import pytest
import sparsegraph as sg
import networkx as nx


class TestSparseGraph:
    def test_labels_same_length(self):
        adj = sg.generators.house_graph().adjacency
        with pytest.raises(ValueError):
            sg.SparseGraph(adj, labels=["a", "b"])

    def test_get_largest_component_is_same_if_connected(self):
        graph = sg.generators.house_graph()
        largest = graph.get_largest_component()
        assert np.all(largest.adjacency.data == graph.adjacency.data)
        assert np.all(largest.adjacency.indices == graph.adjacency.indices)
        assert np.all(largest.adjacency.indptr == graph.adjacency.indptr)
        assert largest.labels == graph.labels

    def test_indegree(self):
        pass

    def test_outdegree(self):
        pass

    def test_get_label(self):
        nx_graph = nx.house_graph()
        sg_graph = sg.from_networkx(nx_graph)
        for i in range(5):
            assert sg_graph.get_label(i) == i
