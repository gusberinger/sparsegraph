import networkx as nx
import sparsegraph as sg
import pytest


class TestCentrality:
    testing_graphs = [
        nx.watts_strogatz_graph(100, 4, 0.05),
        nx.house_graph(),
    ]

    def test_closeness_centrality(self):
        for nx_graph in self.testing_graphs:
            sg_graph = sg.from_networkx(nx_graph)
            closeness_measures = sg.alg.estimate_closeness_centrality(sg_graph)
            nx_closeness_measures = nx.closeness_centrality(nx_graph)
            for sg_val, nx_val in zip(
                closeness_measures, nx_closeness_measures.values()
            ):
                # the algoritm is only an estimate so we allow for more error
                assert sg_val == pytest.approx(nx_val, rel=0.1)
