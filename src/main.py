import sparsegraph as sg
import scipy.sparse as sp
import networkx as nx
import itertools


def infinite_letters(*, max_yield: int | None = None):
    """
    A,B,C,...,Z,AA,AB,...,ZZ,AAA,AAB,...
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    yielded = 0
    for i in itertools.count(1):
        for combination in itertools.product(letters, repeat=i):
            yield "".join(combination)
            yielded += 1
            if max_yield is not None and yielded == max_yield:
                return


def main():
    # create a random graph with 10 nodes and conver it to a csr_matrix
    graph = nx.gnp_random_graph(1000, 0.5, directed=True)
    csr = nx.to_scipy_sparse_array(graph, format="csr")
    labels = list(infinite_letters(max_yield=1000))
    graph = sg.SparseGraph(csr, labels)
    subgraph = graph.get_largest_component()
    print(subgraph.labels)
    sg.alg.katz_centrality(subgraph)


if __name__ == "__main__":
    main()
