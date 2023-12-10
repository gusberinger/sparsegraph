import json
from pathlib import Path

import sparsegraph as sg
import scipy.sparse as sp
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from bidict import bidict
import pickle


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


def alt_main():
    data_path = Path(__file__).parent / "data"
    adj = sp.load_npz(data_path / "wiki_graph.npz")

    with open(data_path / "wiki_id_to_idx.json") as f:
        print()
        page_id_to_idx = bidict(json.load(f))

    with open(data_path / "wiki_id_to_title.pickle", "rb") as f:
        page_id_to_title = pickle.load(f)

    page_ids_sorted_by_idx = sorted(
        page_id_to_idx.keys(), key=lambda k: page_id_to_idx[k]
    )
    titles_sorted_by_idx = [
        page_id_to_title[int(page_id)] for page_id in page_ids_sorted_by_idx
    ]
    # exit()

    graph = sg.SparseGraph(adj, titles_sorted_by_idx)
    print("loaded data")
    page_ids = [24299020]
    for page_id in page_ids:
        page_idx = page_id_to_idx[str(page_id)]
        # page_title = page_id_to_title[page_id]
        label = graph.get_label(page_idx)
        for i, neighbor in enumerate(graph.incoming_neighbors(page_idx)):
            neighbor_label = graph.get_label(neighbor)
            print(f"{label} -> {neighbor_label}")
            if i == 10:
                break


def main():
    graph = sg.alg.distance_from(sg.generators.house_graph(), 0)
    nx_house = nx.house_graph()
    sg_house = sg.from_networkx(nx_house)

    print(sg.alg.estimate_radius_and_diameter(sg_house, 10))
    print(nx.algorithms.distance_measures.radius(nx_house))


if __name__ == "__main__":
    main()
