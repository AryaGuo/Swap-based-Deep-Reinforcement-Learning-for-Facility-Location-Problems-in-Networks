import os
import pickle

import networkx as nx
import numpy as np
import torch
import torch_geometric.data as geom_data
from torch.utils.data import Dataset

import utils
from baselines.sampling import DensitySampling

class SynGraphDataset(Dataset):
    def __init__(self, data_path: str, fac_low: float, fac_high: float):
        super().__init__()

        self.data_path = data_path
        self.city_num = int(data_path.rstrip("/").split("_")[-1])
        self.city_pops = []
        self.distance_m = []
        self.coordinates = []
        self.road_net_data = []

        for i in range(self.city_num):
            sub_path = data_path + "/" + str(i) + "/"

            G = pickle.load(open(sub_path + "G.pkl", "rb"))
            coordinates = np.array(list(nx.get_node_attributes(G, "pos").values()))
            city_pop = np.array(list(nx.get_node_attributes(G, "city_pop").values()))
            distance_m_i = pickle.load(open(sub_path + "distance_m.pkl", "rb"))

            max_dist = np.max(distance_m_i)
            distance_m_i /= max_dist
            edges = list(G.edges(data="length"))
            edge_index = np.array([(u, v) for u, v, _ in edges], dtype=int).T  # [2, m]
            edge_attr = np.array([e[-1] for e in edges], dtype=float)
            edge_attr = edge_attr.reshape(-1, 1) / max_dist  # [m, 1]

            road_net_data = geom_data.Data(
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            )
            self.city_pops.append(city_pop)
            self.distance_m.append(distance_m_i)
            self.coordinates.append(coordinates)
            self.road_net_data.append(road_net_data)

        self.facility_nums = utils.get_fac_num(self.city_pops[0], fac_low, fac_high)

    def __len__(self):
        return self.city_num * len(self.facility_nums)

    def __getitem__(self, index):
        city_id = index // len(self.facility_nums)
        p = self.facility_nums[index % len(self.facility_nums)]
        return (
            city_id,
            self.city_pops[city_id],
            p,
            self.distance_m[city_id],
            self.coordinates[city_id],
            self.road_net_data[city_id],
        )

    def get_city_iter(self):
        for i in range(self.city_num):
            yield i, self.city_pops[i], self.distance_m[i]


class SynGraphImpDataset(SynGraphDataset):
    # for each city & p, fix and dump an initial solution
    def __init__(self, data_path: str, fac_low: float, fac_high: float):
        super().__init__(data_path, fac_low, fac_high)
        self.init_dir = f"{self.data_path}/init/"

    def __getitem__(self, index):
        (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
        ) = super().__getitem__(index)

        if os.path.isfile(f"{self.init_dir}/{city_id}_{p}.pkl"):
            init_facility = pickle.load(
                open(f"{self.init_dir}/{city_id}_{p}.pkl", "rb")
            )
        else:
            init_facility = DensitySampling(2 / 3).sample(city_pop, p)
            os.makedirs(self.init_dir, exist_ok=True)
            pickle.dump(
                init_facility,
                open(f"{self.init_dir}/{city_id}_{p}.pkl", "wb"),
                pickle.HIGHEST_PROTOCOL,
            )
        return (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
            init_facility,
        )
