import argparse, os, re, json, sys, pickle

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from copy import copy
import plotly.graph_objs as go

from pathlib import Path
# from scipy.spatial import distance_matrix
# from itertools import combinations
# from sklearn.linear_model import LinearRegression


__rootdir__ = str(Path(__file__).resolve().parents[1])
sys.path.append(__rootdir__)
# from make_triangle_figure import make_triangle_figure, cmap_dict
from helper_functions import draw_triangle, test_plot_triang, rotation_matrix_from_vectors
from helper_functions import parse_output_path, parse_input_path
from TriangleCombination import TriangleCombination, get_angle
from triangle_reconstruction.numpy2wavefront import create_obj_file

color_list = ['red', 'yellow', 'green', 'purple']
rgb_dict = {'red': (215 / 255, 25 / 255, 28 / 255, 1.0),
            'orange': (253/255, 174 / 255, 97 / 255, 1.0)}

def apply_rotation(rotmat, vec, origin):
    return np.dot(rotmat, (vec - origin).T).T + origin


def sin_fun(xr, p, h, v, x0, y0, sf):
    '''
    Generate RH sinusoid with defined parameters, starting at position (x0,y0)
    '''
    return (v * 0.5 * sf(2 * np.pi / p * (-xr - x0 - h)) + y0
            - v * 0.5 * sf(2 * np.pi / p * (-xr[0] - x0 - h)))


def get_pairwise_distmat(m1, m2):
    '''
    Get distance between each coordinate pair in 2 matrices
    '''
    m1_mod = np.tile(m1, (m2.shape[0], 1, 1))  # nb_waves x nb_points_circle x dims
    m2_mod = np.tile(m2, (m1.shape[0], 1, 1)).transpose((1,0,2))
    dd = np.linalg.norm(m1_mod - m2_mod, axis=-1)
    return dd



def plot_3d_array(wave_dict, return_plot=True, out_dir=None):
    # Create a plotly figure object
    fig = go.Figure()
    all_coords_list = []
    # Add a trace for the line plot
    for plt_id in wave_dict:
        col = wave_dict[plt_id]['color']
        for wv_id in wave_dict[plt_id]['coords']['wave']:
            wv_coords = wave_dict[plt_id]['coords']['wave'][wv_id]
            fig.add_trace(
                go.Scatter3d(
                    x=wv_coords[:, 0],
                    y=wv_coords[:, 1],
                    z=wv_coords[:, 2],
                    mode="lines",
                    line=dict(width=4, color=col),
                    name=plt_id + wv_id
                )
            )
            all_coords_list.append(wv_coords)
        for sc_id in wave_dict[plt_id]['coords']['scatter']:
            sc_coords = wave_dict[plt_id]['coords']['scatter'][sc_id]
            if sc_coords.ndim == 1:
                sc_coords = np.expand_dims(sc_coords, 0)
            fig.add_trace(
                go.Scatter3d(
                    x=sc_coords[:, 0],
                    y=sc_coords[:, 1],
                    z=sc_coords[:, 2],
                    mode="markers",
                    name=plt_id + sc_id,
                    marker=dict(color=col, size=5)
                )
            )
            all_coords_list.append(sc_coords)

    # Set the layout properties
    all_points = np.vstack(all_coords_list)
    all_min, all_max = np.min(all_points), np.max(all_points)
    fig.update_layout(
        title="3D Line Plot",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis",
        xaxis=dict(range=[all_min, all_max]),
        yaxis=dict(range=[all_min, all_max]),
        zaxis=dict(range=[all_min, all_max]),
                   aspectmode='cube')
    )

    if return_plot:
        # Display the figure
        fig.show()
    else:
        fig.write_html(f'{out_dir}triangles_3d.html')


def mirror_triangle(coords):
    coords = coords * np.array([-1, 1, 1])
    coords = coords - coords[1]
    coords = coords[[1,0,2], :]
    return coords


def model_triangle_positions(fret_dict, params_dict, out_dir, dna_shape, return_plot=True):

    # Reconstruct triangle positions in 2D
    tc = TriangleCombination(params_dict['dye_dist'], params_dict['r0'], fret_dict)

    # Generate a stack of helices with prescribed dimensions
    nt_per_p = 360 / params_dict['twist']
    p = nt_per_p * params_dict['axial_rise']  # dna period
    max_nt = max(list(params_dict['nt_dist'].values()))
    nb_periods = max_nt / nt_per_p
    max_len = nb_periods * p + 1
    # max_len = max([ max([np.linalg.norm(cd2) for cd2 in cd[0]]) for cd in tc.coord_dict.values()]) + 10  # longer helix length than the longest vector makes no sense
    xr = np.arange(tc.Cp[0], max_len + tc.Cp[0], 0.05)
    xr_per_p = len(xr) / ((max_len + tc.Cp[0]) / p)
    xr_per_nt = xr_per_p / nt_per_p
    std_wave_array = np.dstack([np.vstack((
            xr,  # x
            sin_fun(xr, p, h, params_dict['helix_diameter'], tc.Cp[0], tc.Cp[1], np.sin),  # y
            sin_fun(xr, p, h, params_dict['helix_diameter'], tc.Cp[0], tc.Cp[1], np.cos)) # z
        ).T
        for h in np.arange(0, p, 0.5)]) # different starting points in rows

    # Determine 3D positions of attachment points docking strands per wave
    tn_coord_dict = {}
    for tn in tc.coord_dict:
        tn_rise_idx = int(round(xr_per_nt * params_dict['nt_dist'][tn]))
        tn_coord_array = std_wave_array[tn_rise_idx, :, :]
        tn_coord_dict[tn] = tn_coord_array.T

    # Determine adjusted measured 3D position of dyes after rotation adjustment around x-axis
    d_circle_dict = {}
    d_circle_steps = np.arange(0, 2 * np.pi, np.pi / 180)
    x_cir_template = np.ones(len(d_circle_steps), dtype=float)
    for tn in tc.coord_dict:
        d_circle_dict[tn] = []
        for d_coord in tc.coord_dict[tn]:
            d_dist = d_coord[2,1] # distance point D to x-axis
            x_cir = x_cir_template * d_coord[2,0]
            y_cir = np.sin(d_circle_steps) * d_dist
            z_cir = np.cos(d_circle_steps) * d_dist
            d_circle_dict[tn].append(np.vstack((x_cir, y_cir, z_cir)).T)

    # minimize dye distances over angles R'C'A'
    angle_list = []
    for a in np.arange(np.pi / 6, np.pi, np.pi / 180):  # 1-deg steps through 30-90deg
        rotmat = rotation_matrix_from_vectors(np.array([np.cos(a), np.sin(a), 0]), np.array([1, 0, 0]))
        dd_list = []
        for dc in d_circle_dict:
            dd_list_cur_triangle = []
            for d_circle_coords_unrotated in d_circle_dict[dc]:
                d_circle_coords = apply_rotation(rotmat, d_circle_coords_unrotated, tc.Cp)
                dd = get_pairwise_distmat(d_circle_coords, tn_coord_dict[dc])
                dd_list_cur_triangle.append(dd.min(axis=1))
            dd_array_cur_triangle = np.vstack(dd_list_cur_triangle)
            dd_list.append(dd_array_cur_triangle)  # retain the value for which wave was closest to circle
        dd_array = np.dstack(dd_list)  # nb_reflections x nb_waves x nb_triangles
        dd_array_m1 = dd_array.min(axis=0)
        dd_min_array = dd_array_m1[np.argmin(dd_array_m1.sum(axis=1)), :]
        dd_mean = dd_min_array.mean()
        angle_list.append((dd_mean, dd_min_array, a, dd_array))
    mean_deviation, deviation_array, best_angle, dd_array = sorted(angle_list, key=lambda x: x[0])[0]
    best_wave_idx = dd_array.min(axis=0).sum(axis=-1).argmin()
    best_reflection_idx = dd_array[:, best_wave_idx, :].argmin(axis=0)

    # Construct best wave and plot
    rev_rotmat = rotation_matrix_from_vectors(np.array([1,0,0]), np.array([np.cos(best_angle), np.sin(best_angle), 0]))
    best_wave_normal = std_wave_array[:, :, best_wave_idx].copy()

    # Find the normal for the best wave
    hwp = best_wave_normal[int(xr_per_p / 2)]
    normal_start = ((hwp - best_wave_normal[0]) / 2)[1:]
    normal_vec = np.vstack([np.concatenate(([0], normal_start)), np.concatenate(([max_len], normal_start))])
    normal_vec = apply_rotation(rev_rotmat, normal_vec, tc.Cp)

    dna_wave = best_wave_normal.copy()
    dna_scaling_factor = 21.0 / params_dict['helix_diameter']
    dna_transl = best_wave_normal[:int(xr_per_p), 1:].mean(axis=0)
    dna_wave[:, 1:] = dna_wave[:, 1:] * dna_scaling_factor #+ dna_transl
    dna_wave[:, 1:] = dna_wave[:, 1:] - dna_wave[:int(xr_per_p), 1:].mean(axis=0) + dna_transl

    best_wave = apply_rotation(rev_rotmat, best_wave_normal, tc.Cp)
    dna_wave = apply_rotation(rev_rotmat, dna_wave, tc.Cp)

    obj_dict = {
        'best_wave': best_wave,
        'dna_wave': dna_wave,
        'best_wave_normal': normal_vec
    }

    wave_plot_dict = {
        'wave': {
            'color': 'grey',
            'coords': {
                'wave': {'_': best_wave}, 'scatter': {}
        }},
        'dna': {
            'color': 'black',
            'coords': {
                'wave': {'_': dna_wave}, 'scatter': {}
            }},
    }

    # Dict for blender orbs
    orb_dict = {
        'Cp': {
            'coords': tc.Cp,
            'size': 2.0,
            'color': rgb_dict['red']
        },
        'Rp': {
            'coords': tc.Rp,
            'size': 2.0,
            'color': rgb_dict['red']
        }
    }
    # Add intermediate orbs for positions of nucleotides
    for ii, i in enumerate(range(0, len(xr), int(xr_per_nt))):
        orb_dict[f'm{ii}'] = {'coords': best_wave[i], 'size': 1.0, 'color': rgb_dict['orange']}

    dye_len_dict = {}
    for acd_idx, acd in enumerate(d_circle_dict):
        connect_coord = apply_rotation(rev_rotmat, tn_coord_dict[acd][best_wave_idx, :], tc.Cp)
        d_circle_best = d_circle_dict[acd][best_reflection_idx[acd_idx]]
        d = d_circle_best[np.argmin(np.linalg.norm(d_circle_best - connect_coord, axis=1))]

        triang_coords = np.vstack((d, tc.Cp, tc.Rp, d))
        dye_dist_coords = np.vstack((connect_coord, d))

        orb_dict[f'd{acd_idx}'] = {'coords': d, 'size': 2.0, 'color': rgb_dict['red']}
        obj_dict[acd] = triang_coords
        obj_dict[acd + '_dd'] = dye_dist_coords

        wave_plot_dict[acd] = {
            'color': color_list[acd_idx],
            'coords': {
                'wave': {'_dd': dye_dist_coords,
                         '_tr': triang_coords},
                'scatter': {'_pts': np.vstack((connect_coord, d))}
            }
        }
        dye_len_dict[acd] = np.linalg.norm(connect_coord - d)
    plot_3d_array(wave_plot_dict, return_plot, out_dir)

    # Save fitting results

    create_obj_file(obj_dict, f'{out_dir}triangle_reconstruction.obj', orb_dict)

    np.save(f'{out_dir}best_wave.npy', best_wave)
    wave_stats_dict = {'best_angle': float(best_angle),
                       'mean_deviation': float(mean_deviation),
                       'max_deviation': float(np.max(deviation_array)),
                       'best_wave_idx': int(best_wave_idx)}
    with open(f'{out_dir}wave_stats.json', 'w') as fh:
        json.dump(wave_stats_dict, fh)

    # get objective value
    dye_lens = list(dye_len_dict.values())
    # dye_distance_delta = np.sum(dye_lens) + np.max(dye_lens)
    # dye_distance_delta = np.max(dye_lens)  # minimax
    dye_distance_delta = np.sum(np.array(dye_lens) ** 2)  # sum
    # dye_distance_delta = np.abs(np.max(dye_lens) - np.min(dye_lens))  # force same dye distances

    return dye_distance_delta



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine triangles from iMAX FRET data.')
    parser.add_argument('--fret-json', type=str, required=True)
    parser.add_argument('--params', type=str, default=f'{__rootdir__}/triangle_reconstruction/dna_property_params/dna_property_params.json')
    parser.add_argument('--dna-shape', type=str, choices=['linear', 'sinusoid'], default='linear')
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    with open(args.fret_json, 'r') as fh:
        fret_dict = json.load(fh)
    with open(args.params, 'r') as fh:
        params_dict = json.load(fh)
    out_dir = parse_output_path(args.out_dir)
    ddd = model_triangle_positions(fret_dict, params_dict, out_dir, args.dna_shape, True)
