import math
import multiprocessing as mp
import os
import pickle
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Edge:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"{self.start} -> {self.end}"


class EdgeIntersector:
    def __init__(self):
        self.edges = []

    def add_edge(self, new_edge: Edge) -> bool:
        """Try to add a new edge to the graph and check for intersections.
        Return True if the edge is added.
        """
        for edge in self.edges:
            if self.intersects(edge, new_edge):
                return False
        self.edges.append(new_edge)
        return True

    def intersects(self, edge1: Edge, edge2: Edge) -> bool:
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = edge1.start, edge1.end
        C, D = edge2.start, edge2.end
        if (A == C).all() or (A == D).all() or (B == C).all() or (B == D).all():
            return False
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def gen_gabriel_graph(data_path: str, seed: int, n: int, k1: int, k2: int):
    """Generate a Gabriel Graph with population.

    Args:
        data_path (str): path to store the generated graph
        seed (int): random seed
        n (int): number of nodes in the graph
        k1, k2 (int): range of nearest neighbors to connect
    """

    def dist(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    np.random.seed(seed)
    os.makedirs(data_path, exist_ok=True)

    # Generate points
    center = (0.5, 0.5)
    std_dev = 0.15
    points = np.clip(np.random.normal(center, std_dev, size=(n, 2)), 0, 1)
    eu_dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    # Check for planarity
    intersector = EdgeIntersector()

    # Gabriel graph
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=points[i])
    for i in range(n):
        for j in range(i):
            xm = 0.5 * (points[i][0] + points[j][0])
            ym = 0.5 * (points[i][1] + points[j][1])
            M = (xm, ym)
            d = dist(M, points[i])
            for k in range(0, n):
                if xm - d < points[k][0] < xm + d and ym - d < points[k][1] < ym + d:
                    if dist(M, points[k]) < d:
                        break
            else:
                G.add_edge(i, j, length=eu_dist[i, j])
                assert intersector.add_edge(Edge(points[i], points[j]))

    # Additional edges
    for i in range(n):
        # Sort distances and get indices of k nearest neighbors
        deg = max(np.random.randint(k1, k2 + 1) - G.degree(i), 0)
        indices = np.argsort(eu_dist[i])[1 : deg + 1]  # Exclude self (distance of 0)
        for j in indices:
            if j not in G.neighbors(i):
                if intersector.add_edge(Edge(points[i], points[j])):
                    G.add_edge(i, j, length=eu_dist[i, j])

    # Compute the distance matrix
    p = dict(nx.shortest_path_length(G))
    distance_m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_m[i, j] = p[i][j] if j in p[i] else math.inf

    # Compute the population values for each node
    city_pop = np.random.uniform(1, 100, n)  # noise
    eigenvector_centrality = nx.eigenvector_centrality(G, 50000)
    for i in range(n):
        city_pop[i] += max(0, np.random.normal(eigenvector_centrality[i] * 50000, 1000))
    for i, pop in enumerate(city_pop):
        G.nodes[i]["city_pop"] = pop

    # Save data
    pickle.dump(G, open(f"{data_path}/G.pkl", "wb"))
    pickle.dump(distance_m, open(f"{data_path}/distance_m.pkl", "wb"))
    pickle.dump(city_pop, open(f"{data_path}/city_pop.pkl", "wb"))
    plot_gabriel_graph(G, city_pop, data_path)

    return city_pop, distance_m, points


def plot_gabriel_graph(G, city_pop, data_path: str):
    node_sizes = np.sqrt(city_pop)
    norm = mcolors.Normalize(vmin=min(city_pop), vmax=max(city_pop))
    cmap = plt.cm.get_cmap("plasma")
    node_colors = [cmap(norm(cent)) for cent in city_pop]

    pos_dict = nx.get_node_attributes(G, "pos")

    plt.figure()
    nx.draw(G, pos_dict, node_size=node_sizes, node_color=node_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="City Population")
    plt.savefig(data_path + "/graph.pdf", bbox_inches="tight", pad_inches=0.1)

    plt.figure()
    city_pop = sorted(city_pop, reverse=True)
    plt.plot(city_pop)
    plt.xlabel("Rank")
    plt.ylabel("Population")
    plt.savefig(data_path + "/population_rank.pdf", bbox_inches="tight", pad_inches=0.1)


def batch_gen(data_path: str, n: int, city_num: int, k1: int = 3, k2: int = 6):
    """Generate Gabriel Graphs in batch.

    Args:
        data_path (str): path to store the generated graphs
        n (int): number of nodes in the graph
        city_num (int): number of graphs to generate
        k1, k2 (int, optional): range of nearest neighbors to connect (default: {3, 6})
    """

    data_path = f"{data_path}/Gabriel/{n}_{k1}_{k2}_{city_num}"

    with mp.Pool(10) as pool:
        for i in range(city_num):
            pool.apply_async(
                gen_gabriel_graph,
                args=(f"{data_path}/{i}/", int(time.time() + i), n, k1, k2),
            )
        pool.close()
        pool.join()


if __name__ == "__main__":
    batch_gen("./data/", 100, 10)
    batch_gen("./data/", 100, 1000)
