import math
import os
import pickle
import time

import numpy as np
import torch
import torch_geometric.data as geom_data

from baselines.sampling import DensitySampling
from ppo import PPOLightning
from results import PMPSolution
from utils import cal_voronoi, get_cost, get_cost_details, to_device


class SwapSolver:
    def __init__(self, iter_num):
        self.iter_num = iter_num

    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        raise NotImplementedError

    def solve(self, p, city_pop, distance_m, swap_num, **kwargs):
        start = time.time()
        best_sol = None
        for _ in range(self.iter_num):
            facility_list = DensitySampling(2 / 3).sample(city_pop, p)
            sol = self.solve_reloc(
                city_pop, p, distance_m, facility_list, reloc_step=swap_num, **kwargs
            )
            if best_sol is None or sol.cost < best_sol.cost:
                best_sol = sol
        best_sol.time = time.time() - start
        return best_sol


class RandomSwapSolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        facility_list_ = facility_list.copy()

        for i in range(self.iter_num):
            facility_list = facility_list_.copy()
            mask = np.ones(np.prod(city_pop.shape), dtype=np.bool)
            mask[facility_list] = 0
            swaps = []

            for j in range(reloc_step):
                fac_in_indices = np.where(mask == 1)[0]

                fac_out_idx = np.random.choice(range(len(facility_list)))
                fac_out = facility_list[fac_out_idx]
                fac_in = np.random.choice(fac_in_indices)
                facility_list[fac_out_idx] = fac_in
                cost = get_cost(facility_list, distance_m, city_pop)

                mask[fac_in] = 0
                mask[fac_out] = 1

                swaps.append((fac_out, fac_in))
                if best_sol is None or cost < best_sol.cost:
                    best_sol = PMPSolution(facility_list, math.nan, cost)
                    best_sol.swaps = swaps

            cost = get_cost(facility_list, distance_m, city_pop)

        best_sol.time = time.time() - start
        return best_sol


class GreedySwapSolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        mask = np.ones(np.prod(city_pop.shape), dtype=np.bool)
        mask[facility_list] = 0
        swaps = []

        for j in range(reloc_step):
            fac_in_indices = np.where(mask == 1)[0]
            min_cost = get_cost(facility_list, distance_m, city_pop)
            best_action = None

            for i, fac_out in enumerate(facility_list):
                for fac_in in fac_in_indices:
                    facility_list_ = facility_list.copy()
                    facility_list_[i] = fac_in
                    cost = get_cost(facility_list_, distance_m, city_pop)
                    if cost < min_cost:
                        min_cost = cost
                        best_action = (fac_out, fac_in)
                    del facility_list_

            if best_action == None:
                break
            fac_out, fac_in = best_action
            facility_list[np.where(facility_list == fac_out)[0]] = fac_in
            mask[fac_in] = 0
            mask[fac_out] = 1
            swaps.append(best_action)
        best_sol = PMPSolution(facility_list, time.time() - start, min_cost)
        best_sol.swaps = swaps
        return best_sol


class VSCASolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        mask = np.ones(np.prod(city_pop.shape), dtype=np.bool)
        mask[facility_list] = 0

        swaps = []
        for j in range(reloc_step):
            pop_list, costs, total_cost = get_cost_details(
                facility_list, distance_m, city_pop
            )[:3]
            swap_out_idx = np.argmin(costs)
            swap_in_idx = np.argmax(costs)
            fac_out = facility_list[swap_out_idx]
            fac_in = None

            for c in pop_list[swap_in_idx]:
                if mask[c] == 0:
                    continue
                facility_list[swap_out_idx] = c
                total_cost_ = get_cost(facility_list, distance_m, city_pop)
                if total_cost_ < total_cost:
                    total_cost = total_cost_
                    fac_in = c

            if fac_in is None:
                facility_list[swap_out_idx] = fac_out
                break
            facility_list[swap_out_idx] = fac_in
            mask[fac_in] = 0
            mask[fac_out] = 1
            swaps.append((fac_out, fac_in))

        best_sol = PMPSolution(facility_list, time.time() - start, total_cost)
        best_sol.swaps = swaps
        return best_sol


class PPOSwapSolver(SwapSolver):
    def __init__(self, iter_num, ckpt, device):
        super().__init__(iter_num)
        self.model = (
            PPOLightning.load_from_checkpoint(ckpt, mode="test").float().to(device)
        )

    def _get_fac_data(
        self,
        city_pop,
        p,
        distance_m,
        facility_list,
        coordinates,
        coordinates_norm,
        road_net_data,
        mask,
    ):
        def PolyArea(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        total_pop = np.sum(city_pop)
        pop_list_c, costs_c, total_cost_c, vor_indices_c = get_cost_details(
            facility_list, distance_m, city_pop
        )
        pop_facility = np.zeros(p)
        for i, indices in enumerate(pop_list_c):
            pop_facility[i] = np.sum(city_pop[np.asarray(indices)])

        vor = cal_voronoi(
            city_pop,
            facility_list,
            coordinates,
            (
                min(coordinates[:, 0]),
                max(coordinates[:, 0]),
                min(coordinates[:, 1]),
                max(coordinates[:, 1]),
            ),
        )

        # the boundary of each polygon
        poly_areas = np.zeros(p)
        for i in range(p):
            region_id = vor.point_region[i]
            region = vor.regions[region_id]
            region = [r for r in region if r != -1]
            region_x = [vor.vertices[r][0] for r in region]
            region_y = [vor.vertices[r][1] for r in region]
            region_x.append(region_x[0])
            region_y.append(region_y[0])
            area = PolyArea(region_x, region_y)
            poly_areas[i] = area

        fac_feat = np.concatenate(
            (
                pop_facility[:, None] / total_pop,
                np.asarray(costs_c)[:, None] / total_cost_c,
                poly_areas[:, None] / np.sum(poly_areas),
            ),
            axis=1,
        )
        node_fac_feat = np.zeros((np.prod(np.shape(city_pop)), fac_feat.shape[1]))
        node_fac_feat[facility_list] = fac_feat

        # GNN: [n, 10]
        node_feat = np.concatenate(
            (
                coordinates_norm,
                city_pop.reshape(-1, 1) / total_pop,
                mask.reshape(-1, 1),
                np.arange(np.prod(np.shape(city_pop)))[:, None],
                vor_indices_c[:, None],
                distance_m[np.arange(np.prod(np.shape(city_pop))), vor_indices_c, None],
                node_fac_feat,
            ),
            axis=1,
        )

        fac_data = geom_data.Data(
            x=torch.tensor(node_feat, dtype=torch.float32),
            edge_index=road_net_data.edge_index,
            edge_attr=road_net_data.edge_attr,
        )

        return fac_data

    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        facility_list_ = facility_list.copy()
        coordinates = kwargs["coordinates"]
        road_net_data = kwargs["road_net_data"]
        coordinates_norm = (coordinates - coordinates.min(axis=0)) / (
            coordinates.max(axis=0) - coordinates.min(axis=0)
        )

        for i in range(self.iter_num):
            facility_list = facility_list_.copy()
            mask = np.ones(np.prod(city_pop.shape), dtype=np.bool)
            mask[facility_list] = 0
            swaps = []

            for j in range(reloc_step):
                state = {
                    "mask": mask,
                    "fac_data": self._get_fac_data(
                        city_pop,
                        p,
                        distance_m,
                        facility_list,
                        coordinates,
                        coordinates_norm,
                        road_net_data,
                        mask,
                    ),
                }
                to_device(state, self.model.device)
                with torch.no_grad():
                    action = self.model(state)[1]
                fac_out, fac_in = action.cpu().numpy()
                fac_out_index = np.where(facility_list == fac_out)[0]
                facility_list[fac_out_index] = fac_in
                cost = get_cost(facility_list, distance_m, city_pop)

                mask[fac_in] = 0
                mask[fac_out] = 1

                swaps.append((fac_out, fac_in))
                if best_sol is None or cost < best_sol.cost:
                    best_sol = PMPSolution(facility_list, math.nan, cost)
                    best_sol.swaps = swaps

            cost = get_cost(facility_list, distance_m, city_pop)

        best_sol.time = time.time() - start
        return best_sol


def run_random_swap(dataset, save_path, iter_num, swap_num, **kwargs):
    name = f"random_swap_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_greedy_swap(dataset, save_path, iter_num, swap_num, **kwargs):
    name = f"greedy_swap_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GreedySwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_VSCA(dataset, save_path, iter_num, swap_num, **kwargs):
    name = f"vsca_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = VSCASolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_ppo_swap(dataset, save_path, iter_num, swap_num, ckpt, device, **kwargs):
    name = f'ppo_swap_{iter_num}_{swap_num}_{kwargs["name"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = PPOSwapSolver(iter_num, ckpt, device)
    for batch in dataset:
        city_id, city_pop, p, distance_m, coordinates, road_net_data = batch[:6]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(
                p,
                city_pop,
                distance_m,
                swap_num,
                coordinates=coordinates,
                road_net_data=road_net_data,
            )
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_random_swap_reloc(dataset, save_path, iter_num, **kwargs):
    name = f"random_swap_{iter_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(city_pop, p, distance_m, facility_list, int(p / 2))
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_greedy_swap_reloc(dataset, save_path, **kwargs):
    name = "greedy_swap"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GreedySwapSolver(None)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(city_pop, p, distance_m, facility_list, int(p / 2))
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_VSCA_reloc(dataset, save_path, **kwargs):
    name = f"VSCA"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = VSCASolver(None)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(city_pop, p, distance_m, facility_list, int(p / 2))
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_ppo_swap_reloc(dataset, save_path, iter_num, ckpt, device, **kwargs):
    name = f'ppo_swap_{iter_num}_{kwargs["name"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = PPOSwapSolver(iter_num, ckpt, device)
    for batch in dataset:
        (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
            facility_list,
        ) = batch
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop,
                p,
                distance_m,
                facility_list,
                int(p / 2),
                coordinates=coordinates,
                road_net_data=road_net_data,
            )
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_original(dataset, save_path, **kwargs):
    name = "original"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            cost = get_cost(facility_list, distance_m, city_pop)
            sol = PMPSolution(facility_list, 0, cost)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))
    return sol_path
