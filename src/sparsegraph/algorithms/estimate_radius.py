import random
import numpy as np
import sparsegraph as sg
import sys


def estimate_radius_and_diameter(graph: sg.SparseGraph, k: int) -> tuple[int, int]:
    """
    Implements https://doi.org/10.1007/11764298_9 Boitmanis et. al (2006)
    :param graph: graph to estimate radius and diameter of
    :param k: number of nodes to sample
    :return: (radius, diameter)
    """
    n = graph.size
    distances = np.full(n, sys.maxsize, dtype=np.int64)
    diameter_estimate = -sys.maxsize
    radius_estimate = sys.maxsize
    for i in range(1, k + 1):
        if i == 1:
            v = random.randint(0, n - 1)
        else:
            # choose furthest from set of proccessed nodes
            v = int(np.argmax(distances))

        bfs_result = sg.alg.distance_from(graph, v)  # bfs
        # distances = list(map(min, zip(bfs_result, distances)))
        distances = np.minimum(bfs_result, distances)
        diameter_estimate = max(diameter_estimate, max(bfs_result))
        radius_estimate = min(diameter_estimate, max(bfs_result))
    return radius_estimate, diameter_estimate


def estimate_radius(graph: sg.SparseGraph, k: int) -> int:
    return estimate_radius_and_diameter(graph, k)[0]


def estimate_diameter(graph: sg.SparseGraph, k: int) -> int:
    return estimate_radius_and_diameter(graph, k)[1]
