import networkx as nx
import sparsegraph as sg
import pytest


class TestBetweennessCentrality:
    testing_graphs = [
        nx.watts_strogatz_graph(1000, 4, 0.05),
        nx.house_graph(),
        nx.star_graph(10),
    ]

    def test_betweenness_centrality_non_normal(self):
        for nx_graph in self.testing_graphs:
            sg_graph = sg.from_networkx(nx_graph)
            sg_values = sg.alg.betweenness_centrality(sg_graph, normalized=False)
            nx_values = nx.betweenness_centrality(nx_graph, normalized=False).values()
            for sg_val, nx_val in zip(sg_values, nx_values):
                assert sg_val == pytest.approx(nx_val)
