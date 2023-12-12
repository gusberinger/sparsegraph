import networkx as nx
import sparsegraph as sg
import pytest


class TestRadiusDiameter:
    testing_graphs = [
        nx.watts_strogatz_graph(50, 4, 0.05),
        nx.house_graph(),
        nx.star_graph(10),
    ]

    def test_radius_diameter(self):
        for nx_graph in self.testing_graphs:
            sg_graph = sg.from_networkx(nx_graph)
            radius, diameter = sg.alg.estimate_radius_and_diameter(sg_graph, k=4000)
            nx_radius = nx.radius(nx_graph)
            nx_diameter = nx.diameter(nx_graph)
            assert diameter == pytest.approx(nx_diameter)
            assert radius == pytest.approx(nx_radius)

            sg_radius = sg.alg.estimate_radius(sg_graph, k=1000)
            assert sg_radius == pytest.approx(nx_radius)
            sg_diameter = sg.alg.estimate_diameter(sg_graph, k=1000)
            assert sg_diameter == pytest.approx(nx_diameter)
