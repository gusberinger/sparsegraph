import networkx as nx
import sparsegraph as sg
import pytest


class TestKatzCentrality:
    testing_graphs = [
        nx.watts_strogatz_graph(1000, 4, 0.05),
        nx.house_graph(),
        nx.star_graph(10),
    ]

    def test_katz_centrality(self):
        for nx_graph in self.testing_graphs:
            sg_graph = sg.from_networkx(nx_graph)
            sg_values = sg.alg.katz_centrality(sg_graph, alpha=0.1)
            nx_values = nx.katz_centrality(nx_graph, alpha=0.1).values()
            for sg_val, nx_val in zip(sg_values, nx_values):
                assert sg_val == pytest.approx(nx_val)
