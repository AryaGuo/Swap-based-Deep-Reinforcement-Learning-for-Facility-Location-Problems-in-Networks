import os
import pickle
from itertools import product

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from results import PMPSolution


class GurobiSolver:
    def __init__(self):
        pass

    def solve(self, p, city_pop, distance_m, **kwargs):
        city_pop = city_pop.flatten()

        customers = list(range(np.prod(city_pop.shape)))
        facilities = list(range(np.prod(city_pop.shape)))

        num_customers = len(customers)
        num_facilities = len(facilities)
        cartesian_prod = list(product(range(num_customers), range(num_facilities)))

        shipping_cost = {}
        for c, f in cartesian_prod:
            shipping_cost[(c, f)] = (
                distance_m[customers[c], facilities[f]] * city_pop[customers[c]]
            )

        m = gp.Model("facility_location")
        for param, param_val in kwargs.items():
            m.setParam(param, param_val)

        select = m.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
        assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name="Assign")

        m.addConstr(select.sum() == p, name="Facility_limit")
        m.addConstrs(
            (assign[(c, f)] <= select[f] for c, f in cartesian_prod), name="Setup2ship"
        )
        m.addConstrs(
            (
                gp.quicksum(assign[(c, f)] for f in range(num_facilities)) == 1
                for c in range(num_customers)
            ),
            name="Demand",
        )

        m.setObjective(assign.prod(shipping_cost), GRB.MINIMIZE)

        m.optimize()

        if m.status != GRB.OPTIMAL:
            print("Optimization was stopped with status %d" % m.status)

        facility_list = []
        for facility in select.keys():
            if select[facility].x == 1:
                facility_list.append(facility)

        sol = PMPSolution(
            np.asarray(facility_list, dtype=int), time=m.Runtime, cost=m.objVal
        )

        return sol


def run_gurobi(dataset, save_path, **kwargs):
    if "TimeLimit" not in kwargs:
        name = f"gurobi_optimal"
    else:
        name = f'gurobi_{kwargs["TimeLimit"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GurobiSolver()
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(sol_path + f"/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, **kwargs)
            pickle.dump(sol, open(sol_path + f"/{city_id}_{p}.pkl", "wb"))

    return sol_path
