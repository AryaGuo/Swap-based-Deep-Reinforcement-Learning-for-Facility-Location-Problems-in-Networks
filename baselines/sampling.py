import math
import os
import pickle
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from results import PMPSolution


class DensitySampling:
    def __init__(self, exp):
        self.exp = exp

    def sample(self, city_pop, p):
        density = np.reshape(city_pop**self.exp, -1)
        density = density / np.sum(density)
        facility_list = np.random.choice(
            np.prod(city_pop.shape), size=p, p=density, replace=False
        )
        return facility_list

    def solve(self, p, iter_num, city_pop, distance_m):
        start = time.time()
        best_sol = None
        for _ in range(iter_num):
            sol = PMPSolution(facility_list=self.sample(city_pop, p), time=math.nan)
            sol.eval(city_pop, distance_m)
            if best_sol is None or sol.cost < best_sol.cost:
                best_sol = sol
        best_sol.time = time.time() - start
        return best_sol


class RandomSolver:
    def __init__(self):
        pass

    def solve(self, p, iter_num, city_pop, distance_m):
        start = time.time()
        best_sol = None
        for _ in range(iter_num):
            sol = PMPSolution(
                facility_list=np.random.choice(
                    np.prod(city_pop.shape), size=p, replace=False
                ),
                time=math.nan,
            )
            sol.eval(city_pop, distance_m)
            if best_sol is None or sol.cost < best_sol.cost:
                best_sol = sol
        best_sol.time = time.time() - start
        return best_sol


def run_density_sampling(dataset, data_path, iter_num, exp, **kwargs):
    if type(exp) is str:
        exp = eval(exp)
    name = f"density_{round(exp,2)}_{iter_num}"
    sol_path = data_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    ds = DensitySampling(exp)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = ds.solve(p, iter_num, city_pop, distance_m)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return name


def run_kmeans(dataset, save_path, iter_num, **kwargs):
    name = f"kmeans_{iter_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    for batch in dataset:
        city_id, city_pop, p, distance_m, coordinates = batch[:5]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            est = KMeans(p, n_init=iter_num)
            start = time.time()
            est.fit(coordinates, sample_weight=city_pop.reshape(-1))
            facility_list = pairwise_distances_argmin(est.cluster_centers_, coordinates)

            sol = PMPSolution(np.asarray(facility_list, dtype=int), time.time() - start)
            sol.eval(city_pop, distance_m)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_random(dataset, save_path, iter_num, **kwargs):
    name = f"random_{iter_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSolver()
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, iter_num, city_pop, distance_m)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path
