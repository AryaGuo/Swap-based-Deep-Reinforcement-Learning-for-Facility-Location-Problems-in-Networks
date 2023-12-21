from dataset import SynGraphImpDataset
from results import save_avg, save_reloc_results
from swap_solver import (
    run_greedy_swap_reloc,
    run_original,
    run_ppo_swap_reloc,
    run_random_swap_reloc,
    run_VSCA_reloc,
)
from utils import get_config, get_fac_num


def run_reloc_graph(config):
    data_path, fac_low, fac_high = (
        config["dataset"]["data_path"],
        config["dataset"]["fac_low"],
        config["dataset"]["fac_high"],
    )
    ds = SynGraphImpDataset(data_path, fac_low, fac_high)
    save_path = data_path + "/results_reloc/"

    costs_list = {}
    for k, v in config["methods"].items():
        run_fn = eval(v["run_fn"])
        del v["run_fn"]
        sol_path = run_fn(dataset=ds, save_path=save_path, **v)
        costs_list[k] = sol_path
        save_avg(sol_path, ds)

    baseline = "Original"
    facility_range = get_fac_num(ds.city_pops[0], fac_low, fac_high)
    save_reloc_results(save_path, costs_list, baseline, facility_range)


if __name__ == "__main__":
    config = get_config(["-c", "config/eval_reloc_graph.yaml"])
    run_reloc_graph(config)
