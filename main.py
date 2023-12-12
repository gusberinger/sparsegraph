import networkx as nx
import sparsegraph as sg
import pytest


testing_graphs = [
    nx.watts_strogatz_graph(300, 4, 0.05),
    nx.house_graph(),
    nx.star_graph(10),
]


for nx_graph in testing_graphs:
    sg_graph = sg.from_networkx(nx_graph)
    sg_values = sg.alg.betweenness_centrality(sg_graph, normalized=True)
    nx_values = list(nx.betweenness_centrality(nx_graph, normalized=True).values())
    print(sg_values)
    print(nx_values)
    for sg_val, nx_val in zip(sg_values, nx_values):
        assert sg_val == pytest.approx(nx_val)
