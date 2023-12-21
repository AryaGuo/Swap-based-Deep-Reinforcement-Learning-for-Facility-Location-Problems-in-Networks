import os
import pickle
import time

import numpy as np
from scipy.spatial.distance import cdist

from results import PMPSolution
from utils import get_cost_details


class MaranzanaSolver:
    def __init__(self, iter_num, swap_num):
        self.p = None
        self.facility_list = None
        self.iter_num = iter_num
        self.swap_num = swap_num

    def step(self, city_pop, distance_m, coordinates):
        city_pop = city_pop.flatten()
        ret = False
        pop_list = get_cost_details(self.facility_list, distance_m, city_pop)[0]
        for i in range(self.p):
            if sum(city_pop[pop_list[i]]) == 0:
                continue

            if coordinates is not None:
                mass_center = np.average(
                    coordinates[pop_list[i]], axis=0, weights=city_pop[pop_list[i]]
                )
                distances = cdist([mass_center], coordinates[pop_list[i]])
                new_facility = pop_list[i][np.argmin(distances)]
            else:
                local_dist = (distance_m * city_pop.flatten())[
                    np.ix_(pop_list[i], pop_list[i])
                ]
                new_facility = pop_list[i][np.argmin(np.sum(local_dist, axis=1))]

            if new_facility != self.facility_list[i]:
                ret = True
                self.facility_list[i] = new_facility
        return ret

    def solve(self, p, city_pop, distance_m, coordinates):
        start = time.time()
        best_sol = None
        self.p = p
        for _ in range(self.iter_num):
            self.facility_list = np.random.choice(
                np.prod(city_pop.shape), size=p, replace=False
            )
            for _ in range(self.swap_num):
                if not self.step(city_pop, distance_m, coordinates):
                    break
            sol = PMPSolution(self.facility_list, time.time() - start)
            sol.eval(city_pop, distance_m)
            if best_sol is None or sol.cost < best_sol.cost:
                best_sol = sol
        best_sol.time = time.time() - start
        return best_sol


def run_maranzana(dataset, save_path, iter_num, swap_num, **kwargs):
    name = f"maranzana_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = MaranzanaSolver(iter_num, swap_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, coordinates = batch[:5]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, coordinates)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path
