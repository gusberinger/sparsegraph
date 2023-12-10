import networkx as nx
import sparsegraph as sg


def to_networkx(self):
    return nx.from_scipy_sparse_array(self.adjacency, create_using=nx.DiGraph)


def from_networkx(graph: nx.Graph | nx.DiGraph | nx.MultiDiGraph | nx.MultiGraph):
    adj = nx.to_scipy_sparse_array(graph)
    labels = list(graph.nodes)
    return sg.SparseGraph(adj, labels)
