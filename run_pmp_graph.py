from baselines.greedy import run_greedy
from baselines.gurobi import run_gurobi
from baselines.maranzana import run_maranzana
from baselines.sampling import run_kmeans, run_random
from dataset import SynGraphDataset
from results import save_avg, save_pmp_results
from swap_solver import run_greedy_swap, run_ppo_swap, run_random_swap, run_VSCA
from utils import get_config, get_fac_num


def run_pmp_graph(config):
    data_path, fac_low, fac_high = (
        config["dataset"]["data_path"],
        config["dataset"]["fac_low"],
        config["dataset"]["fac_high"],
    )
    ds = SynGraphDataset(data_path, fac_low, fac_high)
    save_path = data_path + "/results_pmp/"

    costs_list = {}
    for k, v in config["methods"].items():
        run_fn = eval(v["run_fn"])
        del v["run_fn"]
        sol_path = run_fn(dataset=ds, save_path=save_path, **v)
        costs_list[k] = sol_path
        save_avg(sol_path, ds)

    baseline = "Gurobi" if "Gurobi" in costs_list else None
    facility_range = get_fac_num(ds.city_pops[0], fac_low, fac_high)
    save_pmp_results(save_path, costs_list, baseline, facility_range)


if __name__ == "__main__":
    config = get_config(["-c", "config/eval_pmp_graph.yaml"])
    run_pmp_graph(config)
