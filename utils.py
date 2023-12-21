import argparse

import numpy as np
import torch
import yaml
from scipy import spatial
from torch_geometric.data import Data


def get_config(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="config/config.yaml",
    )
    args = parser.parse_args(args)

    with open(args.filename) as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_fac_num(city_pop, fac_low, fac_high, thresh=20):
    if type(city_pop) is int:
        n = city_pop
    elif type(city_pop) is np.ndarray:
        n = np.prod(city_pop.shape)
    x, y = int(n * fac_low), int(n * fac_high) + 1
    step = max(1, int((y - x + thresh - 1) / thresh))
    return range(x, y, step)


def to_device(state: dict, device: str):
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                state[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, Data) or isinstance(v, torch.Tensor):
                state[k].to(device)
    return state


def get_cost(facility_list, distance_m, city_pop):
    total_cost = np.sum(
        (distance_m[facility_list] * city_pop.flatten())[
            np.argmin(distance_m[facility_list], axis=0), np.arange(distance_m.shape[1])
        ]
    )
    return total_cost


def get_cost_details(facility_list, distance_m, city_pop):
    """Compute the cost and allocation of each demand points.

    Returns:
        pop_list: demand points allocated for each facility
        costs: regional cost for each facility
        total_cost: global cost
        point_indices: facility index for each demand point
    """

    wdist = (
        distance_m[facility_list] * city_pop.flatten()
    )  # i-th facility to j-th point
    pop_list = [[] for _ in range(len(facility_list))]
    costs = np.zeros(len(facility_list))

    point_indices = np.argmin(
        distance_m[facility_list], axis=0
    )  # [n] value in [0, m-1]
    for loc, point_index in enumerate(point_indices):
        pop_list[point_index].append(loc)
        costs[point_index] += wdist[point_index, loc]

    total_cost = sum(costs)
    return pop_list, costs, total_cost, point_indices


def voronoi_boundary(towers, bounding_box):
    def in_box(towers, bounding_box):
        return np.logical_and(
            np.logical_and(
                bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]
            ),
            np.logical_and(
                bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]
            ),
        )

    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )

    vor = spatial.Voronoi(points)

    p = len(points_center)
    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(p)]
    vor.filtered_ridge_points = [r for r in vor.ridge_points if r[0] < p and r[1] < p]
    return vor


def cal_voronoi(city_pop, facilities, coordinates=None, boundary=None):
    if coordinates is None:
        city_w, city_l = city_pop.shape
        facility_loc = []
        for block in facilities:
            b_x, b_y = np.unravel_index(block, city_pop.shape)
            b_x = b_x + 0.5
            b_y = b_y + 0.5
            facility_loc.append([b_x, b_y])
        points = np.array(facility_loc)
        vor = voronoi_boundary(points, (0, city_w + 0, 0, city_l + 0))
    else:
        points = np.array(coordinates[facilities])
        vor = voronoi_boundary(points, boundary)

    return vor
