import networkx as nx
import sparsegraph as sg
import pytest


class TestRadiusDiameter:
    testing_graphs = [
        nx.watts_strogatz_graph(100, 4, 0.05),
        nx.house_graph(),
        nx.star_graph(10),
    ]

    def test_radius_diameter(self):
        for nx_graph in self.testing_graphs:
            sg_graph = sg.from_networkx(nx_graph)
            radius, diameter = sg.alg.estimate_radius_and_diameter(sg_graph, 1000)
            nx_radius = nx.radius(nx_graph)
            nx_diameter = nx.diameter(nx_graph)
            pytest.approx(diameter, nx_diameter)
            pytest.approx(radius, nx_radius)

            sg_radius = sg.alg.estimate_radius(sg_graph, 1000)
            pytest.approx(sg_radius, nx_radius)
            sg_diameter = sg.alg.estimate_diameter(sg_graph, 1000)
            pytest.approx(sg_diameter, nx_diameter)
