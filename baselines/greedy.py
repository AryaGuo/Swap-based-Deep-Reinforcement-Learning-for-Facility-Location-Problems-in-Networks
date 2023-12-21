import os
import pickle
import time

import numpy as np

from results import PMPSolution
from utils import get_cost


class GreedySolver:
    def __init__(self):
        self.p = None
        self.facility_list = None
        self.candidates = None

    def step(self, city_pop, distance_m):
        min_cost = None
        min_i = None
        for i in self.candidates:
            total_cost = get_cost(self.facility_list + [i], distance_m, city_pop)
            if min_cost is None or total_cost < min_cost:
                min_cost = total_cost
                min_i = i
        self.facility_list.append(min_i)
        self.candidates.remove(min_i)

    def solve(self, p, city_pop, distance_m):
        start = time.time()
        self.p = p
        self.facility_list = []
        self.candidates = list(range(np.prod(city_pop.shape)))
        for _ in range(p):
            self.step(city_pop, distance_m)
        sol = PMPSolution(self.facility_list, time.time() - start)
        sol.eval(city_pop, distance_m)
        return sol


def run_greedy(dataset, save_path, **kwargs):
    name = "greedy_addition"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GreedySolver()
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path
