import math
import pickle

from utils import get_cost


class PMPSolution:
    def __init__(self, facility_list, time, cost=None):
        self.facility_list = facility_list
        self.time = time
        self.size = len(self.facility_list)
        self.cost = cost

    def eval(self, city_pop, distance_m):
        if self.cost is None:
            self.cost = get_cost(self.facility_list, distance_m, city_pop)


def save_avg(sol_path, dataset):
    costs = {}
    rtimes = {}
    for batch in dataset:
        city_id, p = batch[0], batch[2]
        sol = pickle.load(open(sol_path + f"/{city_id}_{p}.pkl", "rb"))
        costs.setdefault(p, []).append(sol.cost)
        rtimes.setdefault(p, []).append(sol.time)

    avg_cost = {}
    avg_rtime = {}
    for k, v in costs.items():
        avg_cost[k] = sum(v) / len(v)
    for k, v in rtimes.items():
        avg_rtime[k] = sum(v) / len(v)

    pickle.dump(avg_cost, open(sol_path + "/avg_costs.pkl", "wb"))
    pickle.dump(avg_rtime, open(sol_path + "/avg_time.pkl", "wb"))


def save_pmp_results(save_path, res_list, baseline, facility_range):
    # print cost to csv
    with open(save_path + "/cost.csv", "w") as f:
        f.write("facility number,")
        f.write("average,")
        for p in facility_range:
            f.write(f"{p},")
        f.write("\n")
        for name, fin in res_list.items():
            y = pickle.load(open(fin + "/avg_costs.pkl", "rb"))
            f.write(f"{name},")
            f.write(f"{sum(y.values())/len(y.values())},")
            for g in y.values():
                f.write(f"{g},")
            f.write("\n")

    # print running time to csv
    with open(save_path + "/run_time.csv", "w") as f:
        f.write("facility number,")
        f.write("average,")
        for p in facility_range:
            f.write(f"{p},")
        f.write("\n")
        for name, fin in res_list.items():
            y = pickle.load(open(fin + "/avg_time.pkl", "rb"))
            f.write(f"{name},")
            f.write(f"{sum(y.values())/len(y.values())},")
            for g in y.values():
                f.write(f"{g},")
            f.write("\n")

    # print optimality gap to csv
    if baseline is not None:
        optimal = pickle.load(open(res_list[baseline] + "/avg_costs.pkl", "rb"))
        with open(save_path + "/optimality_gap.csv", "w") as f:
            f.write("facility number,")
            f.write("average,")
            for p in facility_range:
                f.write(f"{p},")
            f.write("\n")
            for name, fin in res_list.items():
                y = pickle.load(open(fin + "/avg_costs.pkl", "rb"))
                gap = [
                    0
                    if math.isclose(y[i], optimal[i])
                    else ((y[i] - optimal[i]) / optimal[i])
                    for i in facility_range
                ]
                f.write(f"{name},")
                f.write(f"{sum(gap)/len(gap)},")
                for g in gap:
                    f.write(f"{g},")
                f.write("\n")


def save_reloc_results(save_path, res_list, baseline, facility_range):
    # print running time to csv
    with open(save_path + "/run_time.csv", "w") as f:
        f.write("facility number,")
        f.write("average,")
        for p in facility_range:
            f.write(f"{p},")
        f.write("\n")
        for name, fin in res_list.items():
            y = pickle.load(open(fin + "/avg_time.pkl", "rb"))
            f.write(f"{name},")
            f.write(f"{sum(y.values())/len(y.values())},")
            for g in y.values():
                f.write(f"{g},")
            f.write("\n")

    # print improvement ratio to csv
    init_cost = pickle.load(open(res_list[baseline] + "/avg_costs.pkl", "rb"))
    with open(save_path + "/improve_ratio.csv", "w") as f:
        f.write("facility number,")
        f.write("average,")
        for p in facility_range:
            f.write(f"{p},")
        f.write("\n")
        for name, fin in res_list.items():
            y = pickle.load(open(fin + "/avg_costs.pkl", "rb"))
            imp = [(init_cost[i] - y[i]) / init_cost[i] for i in facility_range]
            f.write(f"{name},")
            f.write(f"{sum(imp)/len(imp)},")
            for g in imp:
                f.write(f"{g},")
            f.write("\n")
