import argparse, os, re, json, sys, pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.spatial import distance_matrix
from itertools import combinations

__rootdir__ = str(Path(__file__).resolve().parents[1])
sys.path.append(__rootdir__)
# from make_triangle_figure import make_triangle_figure, cmap_dict
from helper_functions import draw_triangle, test_plot_triang, rotation_matrix_from_vectors
from helper_functions import parse_output_path, parse_input_path, get_FRET_distance
from coordinate_recomposition.fret2coords import fret2coords


# --- Parameters ---
min2idx = {0: (0,1), 1: (0,2) ,2:(1,2)}


# --- Functions ---
def get_angle(v1, v2, p):
    vp1, vp2 = v1 - p, v2 - p
    ang = np.arccos(np.dot(vp1, vp2) / (np.linalg.norm(vp1) * np.linalg.norm(vp2)))
    if ang < np.pi:
        return ang
    else:
        return 2 * np.pi - ang


def get_bases(d1, d2):
    diffmat = np.abs(np.expand_dims(d1, (-1)) - d2)
    d1_min, d2_min = np.unravel_index(diffmat.argmin(), diffmat.shape)
    d1_base, d2_base = min2idx[d1_min], min2idx[d2_min]  # find the points that form the basis
    return d1_base, d2_base


def get_base_v2(d, target):
    return min2idx[np.argmin(d - target)]


def reorder_coordinates(coords, base_idx):
    """
    reorder coordinates of triangle so that (1) base is on bottom and (2) left-lower point is at origin.
    """
    # ensure top coord is last
    coords_out = np.copy(coords)
    top_coord_idx = np.setdiff1d(np.arange(3), base_idx)
    pivot_idx = np.concatenate((np.delete(np.arange(3), top_coord_idx), top_coord_idx))
    coords_out = coords_out[pivot_idx]

    # # ensure left angle is largest
    # ra = get_angle(coords_out[0], coords_out[2], coords_out[1])
    # la = get_angle(coords_out[1], coords_out[2], coords_out[0])
    # if la < ra:
    #     coords_out = coords_out[(1,0,2), :]
    #     coords_out[:, 0] *= -1  # note: reflect in y-axis will only work if dim-0 is not 000

    # translate so that left-lower point is at origin
    coords_out = coords_out - coords_out[0]

    # ensure base is parallel to x-axis
    target_base_vec = np.array([np.linalg.norm(coords_out[1]), 0.0, 0.0])
    rotmat = rotation_matrix_from_vectors(coords_out[1], target_base_vec)
    coords_out = np.dot(rotmat, coords_out.T).T

    # if top coord ends up under base, mirror in x-axis
    if coords_out[2,1] < 0:
        coords_out[:, 1] *= -1

    return coords_out


class TriangleCombination(object):
    def __init__(self, dye_dist, r0, fret_dict):
        self.coord_dict = None  # Dict of tuples per triangle ID: (coords, mirrored_coords)
        self.r0 = r0
        self.dye_dist = dye_dist
        self.fret_dict = fret_dict
        self.combine()

    def combine(self):

        # --- collect disjointed coordinates per triangle ---
        coord_dict = {}
        for fidx, fid in enumerate(self.fret_dict):
            lens = [get_FRET_distance(x, r0=self.r0) for x in self.fret_dict[fid]]
            coord_dict[fid] = fret2coords(lens, 3)[0]

        # --- find shared triangle bases by size ---
        dist_dict = {fid: distance_matrix(coord_dict[fid], coord_dict[fid])[np.triu_indices(3, 1)]
                     for fid in coord_dict}
        base_idx_dict = {dd: [] for dd in dist_dict}
        for di1, di2 in combinations(dist_dict, 2):
            b1, b2 = get_bases(dist_dict[di1], dist_dict[di2])
            base_idx_dict[di1].append(b1)
            base_idx_dict[di2].append(b2)
        base_dict = {}
        for bid in base_idx_dict:
            unique_bases, counts = np.unique(np.vstack(base_idx_dict[bid]), axis=0, return_counts=True)
            base_dict[bid] = unique_bases[np.argmax(counts)]

        # --- mirror coordinates ---
        baselen_list = []
        for cdi in coord_dict:
            coords = reorder_coordinates(coord_dict[cdi], base_dict[cdi])
            baselen_list.append(np.linalg.norm(coords[1]))
            coords_mirror = coords[(1, 0, 2), :]
            coords_mirror[:, 0] *= -1
            coords_mirror = coords_mirror - coords_mirror[0]
            coord_dict[cdi] = (coords, coords_mirror)
        self.coord_dict = coord_dict

        # --- construct base trapezoid RCC'R' ---
        R = np.mean(np.vstack([coord_dict[cdi][0][1] for cdi in coord_dict]), axis=0)
        C = np.array([0.0, 0.0, 0.0])
        CCp_vdist = 0.0
        CCp_hdist = self.dye_dist
        # CR_dist = np.linalg.norm(C - R)
        #
        # CCp_hdist = (CR_dist - self.base_length) / 2
        # CCp_vdist_sq = self.dye_dist ** 2 - CCp_hdist ** 2
        # if CCp_vdist_sq > 0:
        #     CCp_vdist = np.sqrt(self.dye_dist ** 2 - CCp_hdist ** 2)
        # else:
        #     CCp_vdist = 0.0
        #     CCp_hdist = self.dye_dist
        Cp = C + np.array([CCp_hdist, CCp_vdist, 0])
        Rp = Cp + np.array([np.median(baselen_list), 0, 0])
        # Rp = Cp + np.array([self.base_length, 0, 0])
        self.C, self.R, self.Cp, self.Rp = C, R, Cp, Rp
