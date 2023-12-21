import gymnasium as gym
import numpy as np
import torch
import torch_geometric.data as geom_data

from baselines.sampling import DensitySampling
from dataset import SynGraphImpDataset
from utils import cal_voronoi, get_cost, get_cost_details


class SwapEnv(gym.Env):
    def __init__(
        self,
        dataset,
        data_path,
        fac_low=0.1,
        fac_high=0.1,
        episode_len=200,
    ):
        self._dataset = eval(dataset)(data_path, fac_low, fac_high)
        self._index_iter = iter(range(len(self._dataset)))
        self._index = None
        self._steps = None
        self._episode_len = episode_len

        # states
        self.city_pop = None
        self.p = None
        self.distance_m = None
        self.coordinates = None
        self.coordinates_norm = None
        self.road_net_data = None
        self.total_cost = None
        self.total_pop = None
        self.init_cost = None

        self.facility_list = None
        self.mask = None

    def _get_fac_data(self):
        def PolyArea(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        pop_list_c, costs_c, total_cost_c, vor_indices_c = get_cost_details(
            self.facility_list, self.distance_m, self.city_pop
        )
        pop_facility = np.zeros(self.p)
        for i, indices in enumerate(pop_list_c):
            pop_facility[i] = np.sum(self.city_pop[np.asarray(indices)])

        vor = cal_voronoi(
            self.city_pop, self.facility_list, self.coordinates_norm, (0, 1, 0, 1)
        )

        # the boundary of each polygon
        poly_areas = np.zeros(self.p)
        for i in range(self.p):
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
                pop_facility[:, None] / self.total_pop,
                np.asarray(costs_c)[:, None] / total_cost_c,
                poly_areas[:, None] / np.sum(poly_areas),
            ),
            axis=1,
        )
        node_fac_feat = np.zeros((np.prod(np.shape(self.city_pop)), fac_feat.shape[1]))
        node_fac_feat[self.facility_list] = fac_feat

        # GNN: [n, 10]
        node_feat = np.concatenate(
            (
                self.coordinates_norm,
                self.city_pop.reshape(-1, 1) / self.total_pop,
                self.mask.reshape(-1, 1),
                np.arange(np.prod(np.shape(self.city_pop)))[:, None],
                vor_indices_c[:, None],
                self.distance_m[
                    np.arange(np.prod(np.shape(self.city_pop))), vor_indices_c, None
                ],
                node_fac_feat,
            ),
            axis=1,
        )

        fac_data = geom_data.Data(
            x=torch.tensor(node_feat, dtype=torch.float32),
            edge_index=self.road_net_data.edge_index,
            edge_attr=self.road_net_data.edge_attr,
        )

        return fac_data

    def _get_obs(self):
        return {
            "population": self.city_pop,
            "facility_list": self.facility_list,
            "distance_m": self.distance_m,
            "mask": self.mask,
            "fac_data": self._get_fac_data(),
            "total_cost": self.total_cost,
        }

    def _get_info(self):
        return {"cost": self.total_cost}

    def reset(self):
        try:
            self._index = next(self._index_iter)
        except StopIteration:
            self._index_iter = iter(range(len(self._dataset)))
            self._index = next(self._index_iter)

        (
            _,
            self.city_pop,
            self.p,
            self.distance_m,
            self.coordinates,
            self.road_net_data,
            self.facility_list,
        ) = self._dataset[self._index]
        self.coordinates_norm = (self.coordinates - self.coordinates.min(axis=0)) / max(
            self.coordinates.max(axis=0) - self.coordinates.min(axis=0)
        )
        self.total_pop = np.sum(self.city_pop)

        self._steps = 0
        self.facility_list = DensitySampling(2 / 3).sample(self.city_pop, self.p)

        self.mask = np.ones(np.prod(self.city_pop.shape), dtype=np.bool)
        self.mask[self.facility_list] = 0

        self.total_cost = get_cost(self.facility_list, self.distance_m, self.city_pop)
        self.init_cost = self.total_cost

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        pre_cost = self.total_cost

        fac_out, fac_in = action
        if fac_out == fac_in:
            done = True
            return self._get_obs(), 0.0, done, False, self._get_info()

        assert self.mask[fac_out] == 0
        assert self.mask[fac_in] == 1
        self.facility_list = self.facility_list.copy()
        self.mask = self.mask.copy()

        self.facility_list[self.facility_list == fac_out] = fac_in
        self.mask[fac_out] = 1
        self.mask[fac_in] = 0
        self.total_cost = get_cost(self.facility_list, self.distance_m, self.city_pop)
        reward = (pre_cost - self.total_cost) / self.init_cost  # improvement ratio

        self._steps += 1
        truncated = self._steps == self._episode_len
        done = False  # allow negative reward
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, truncated, info
