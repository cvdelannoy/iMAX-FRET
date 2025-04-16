import os, sys
import re
import shutil
from math import ceil

import prody as pr
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go

from datetime import datetime
from glob import glob
from pathlib import Path
from contextlib import contextmanager
from geometricus import MomentInvariants, SplitType
from scipy.stats import norm
from itertools import permutations, chain
from scipy.spatial import distance_matrix
from matplotlib.collections import LineCollection




wur_colors = ['#E5F1E4', '#3F9C35']
categorical_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
continuous_colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84',
                     '#fc8d59', '#ef6548', '#d7301f', '#990000']

r0_default = 54.8 # forster radius in Angstrom for Cy3-Cy5, according to Murphy et al. 2004


aa_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    'X': 'TAG'
}


default_cmap_dict = {'orange': mpl.colors.LinearSegmentedColormap.from_list('oranges_custom', ('#fee8c8', '#e34a33')),
             'blue': mpl.colors.LinearSegmentedColormap.from_list('blues_custom', ('#bdc9e1', '#045a8d')),
             'green': mpl.colors.LinearSegmentedColormap.from_list('greens_custom', ('#b2e2e2', '#006d2c')),
             'purple': mpl.colors.LinearSegmentedColormap.from_list('purples_custom', ('#bcbddc', '#756bb1'))}


# --- IO ---
def parse_input_path(in_dir, pattern=None, regex=None):
    if type(in_dir) != list: in_dir = [in_dir]
    out_list = []
    for ind in in_dir:
        if not os.path.exists(ind):
            raise ValueError(f'{ind} does not exist')
        if os.path.isdir(ind):
            ind = os.path.abspath(ind)
            if pattern is not None: out_list.extend(glob(f'{ind}/**/{pattern}', recursive=True))
            else: out_list.extend(glob(f'{ind}/**/*', recursive=True))
        else:
            if pattern is None: out_list.append(ind)
            elif pattern.strip('*') in ind: out_list.append(ind)
    if regex is not None:
        out_list = [fn for fn in out_list if re.search(regex, fn)]
    return out_list


def parse_output_path(out_dir, clean=False):
    out_dir = os.path.abspath(out_dir) + '/'
    if clean:
        shutil.rmtree(out_dir, ignore_errors=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

# ---
def make_atom_group(atom_dict, coordset=0, linker_length=0.0, atm_type='prody'):
    if type(atom_dict) in (list, np.ndarray):
        atom_dict = {i: atom for i, atom in enumerate(atom_dict)}
    dye_coords_list = []
    if atm_type == 'prody':
        coords_list = [np.vstack((atom_dict[resi][0].getCoordsets(coordset)[0],
                                  atom_dict[resi][1].getCoordsets(coordset)[0]))
                       for resi in sorted(atom_dict)]
        for coords in coords_list:
            # extend dye position in direction of CB
            v = coords[1] - coords[0]
            uv = v / np.linalg.norm(v)
            dye_coords_list.append(coords[0] + uv * linker_length)

        out_group = pr.atomgroup.AtomGroup('Protein')
        out_group.addCoordset(np.vstack(dye_coords_list))
    elif atm_type == 'numpy':
        coords_list = list(atom_dict.values())
        out_group = pr.atomgroup.AtomGroup('Protein')
        out_group.addCoordset(np.vstack(coords_list))
    else:
        raise ValueError(f'Not a recognized atom type: {atm_type}')
    out_group.setNames(['CA'] * len(atom_dict))
    out_group.setResnums(list(atom_dict))
    out_group.setResnames(['TYR'] * len(atom_dict))
    return out_group

def np2embedded(coord_dict, embedder, feature_mode='both'):
    """
    Convert dict containing imax fret data to features, optionally including embedding of recomposed coordinates
    :param coord_dict:
    :param embedder:
    :param feature_mode: include only [fret] values or [both] fret values and embedded coordinates
    :return:
    """
    embedding_list = []
    for tagged_resn in sorted(coord_dict):
        invars = []
        coord_list = [np.array([[0.0,0.0,0.0]])]  if coord_dict[tagged_resn]['recomposed_coords'] is None else coord_dict[tagged_resn]['recomposed_coords']
        fp_list = []
        if coord_dict[tagged_resn]['recomposed_coords'] is not None:
            for ci, coord in enumerate(coord_dict[tagged_resn]['recomposed_coords']):
                coord = coord[0]
                if coord.ndim == 1: coord = np.expand_dims(coord, 0)
                atm_group = make_atom_group(coord, atm_type='numpy')
                invars.append(MomentInvariants.from_prody_atomgroup(str(ci), atm_group, split_type=SplitType.RADIUS,
                                                                    split_size=embedder[tagged_resn].radius))
                fp_list.append(coord_dict[tagged_resn]['fingerprint'][ci])
        emb = embedder[tagged_resn].embed(invars)
        if feature_mode == 'fret':
            feature_mat = np.vstack(fp_list)  # note: could use more examples if just saving fp without running above code, but we want to remove variation due to script failing
        elif feature_mode == 'both':
            feature_mat = np.hstack((emb.embedding, np.vstack(fp_list)))
        else:
            raise ValueError(f'{feature_mode} is not a valid feature_mode')
        embedding_list.append(feature_mat)
    embedding_vec = np.hstack(embedding_list)
    return embedding_vec


def get_FRET_efficiency(dist, r0=None):
    """
    return FRET efficiency for a given distance, between cy3-cy5
    """
    if r0 is None:
        return 1.0/((dist / r0_default) ** 6 + 1)
    return 1.0/((dist / r0) ** 6 + 1)


def get_FRET_distance(efret, limits=None, r0=None):
    efret = np.clip(efret, 0.01, 0.99)  # avoid nans
    d = (efret ** -1 - 1) ** (1/6)
    d *= r0_default if r0 is None else r0
    if limits is None:
        return d
    return np.clip(d, limits[0], limits[1])

def get_sigma(p, res):
    """
    Find std deviation for given threshold and resolution
    """
    return -1 * res / norm.ppf(p * 0.5, 0, 1)


def print_timestamp(txt):
    ts = str(datetime.now().strftime('%H:%M:%S'))
    print(f'{ts}: {txt}')

def get_aligned_rmsd(c1, c2):
    rmsd_list = []
    c1_centered = c1 - np.mean(c1, axis=0)
    for c1_order in permutations(np.arange(len(c1)), len(c1)):
        c1_reordered = c1_centered.copy()[c1_order, :]
        for mirror_dims in ([], [0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]):
            c1_mirrored = c1_reordered.copy()
            for i in mirror_dims: c1_mirrored[:, i] *= -1
            c1_mapped = kabsch(c1_mirrored, c2)
            rmsd_list.append(np.sqrt(np.mean((c1_mapped - c2) ** 2)))
    return min(rmsd_list)

def get_best_alignment(c1, c2):
    rmsd_list = []
    coord_list = []
    c1_centered = c1 - np.mean(c1, axis=0)
    for c1_order in permutations(np.arange(len(c1)), len(c1)):
        c1_reordered = c1_centered[c1_order, :].copy()
        for mirror_dims in ([], [0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]):
            c1_mirrored = c1_reordered.copy()
            for i in mirror_dims: c1_mirrored[:, i] *= -1
            c1_mapped = kabsch(c1_mirrored, c2)
            coord_list.append(c1_mapped)
            rmsd_list.append(np.sqrt(np.mean((c1_mapped - c2) ** 2)))
    return coord_list[np.argmin(rmsd_list)]

def kabsch(c1, c2):
    c1_centroid, c2_centroid = np.mean(c1, axis=0), np.mean(c2, axis=0)
    c1_centered, c2_centered = c1 - c1_centroid, c2 - c2_centroid

    h = np.matmul(c1_centered.T, c2_centered)
    u,s,vt = np.linalg.svd(h)
    v = vt.T
    d = np.sign(np.linalg.det(v @ u.T)).astype(int)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    r = v @ e @ u.T
    # r = vt @ u.T
    c1_mapped = np.matmul(c1_centered, r.T) + c2_centroid
    return c1_mapped


def plot_structure(coords_dict, fn=None, auto_open=True):
    color_list = ['blue', 'yellow', 'red', 'green', 'black']
    if len(coords_dict) > len(color_list):
        color_list * ceil(len(coords_dict) / len(color_list))
    trace_list = []
    for ci, cn in enumerate(coords_dict):
        cd = coords_dict[cn]
        cd = np.vstack((cd, cd[0,:]))
        trace_bb = go.Scatter3d(x=cd[:, 0],
                                y=cd[:, 1],
                                z=cd[:, 2],
                                name=cn,
                                marker=dict(size=5, color=color_list[ci])
                                )
        trace_list.append(trace_bb)
    pmin = np.min(np.vstack(coords_dict.values()))
    pmax = np.max(np.vstack(coords_dict.values()))
    layout = go.Layout(scene=dict(
        xaxis=dict(range=[pmin, pmax]),
        yaxis=dict(range=[pmin, pmax]),
        zaxis=dict(range=[pmin, pmax]),
        aspectmode='cube'
    )
    )
    fig = go.Figure(data=trace_list, layout=layout)
    if fn is None:
        py.offline.plot(fig, auto_open=auto_open)
    else:
        py.offline.plot(fig, filename=fn, auto_open=auto_open)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    v1_size, v2_size = len(vec1), len(vec2)
    vec1, vec2 = np.pad(vec1, (0, max(0, 3 - len(vec1)))), np.pad(vec2, (0, max(0, 3 - len(vec2))))
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix[:v1_size, :v2_size]


def draw_ruler(rlen, y_pos):
    cur_segments = [[(0, y_pos), (rlen, y_pos)]]
    for sub_l in np.arange(0, np.floor(rlen) + 1, 1):
        smod = 0.3 if sub_l % 10 else 0.6
        cur_segments.append([(sub_l, y_pos+0.055), (sub_l, y_pos - smod)])
    return cur_segments


def add_line_collection(segment_list, color_list, ax):
    for rs, rc in zip(segment_list, color_list):
        ax.add_collection(LineCollection(rs, colors=rc))
    coord_stack = np.vstack(list(chain.from_iterable(chain.from_iterable(segment_list))))
    y_min, y_max, x_min, x_max = (coord_stack[:,1].min(), coord_stack[:,1].max(),
                                  coord_stack[:,0].min(), coord_stack[:,0].max())
    ax.set_ylim((y_min - 0.1, y_max + 0.1))
    ax.set_xlim((x_min - 0.1, x_max + 0.1))



def plot_linecollections(segment_list, color_list, fig_dims, equalize_dims=False, ruler_dims=None):
    coord_stack = np.vstack(list(chain.from_iterable(chain.from_iterable(segment_list))))
    y_min, y_max, x_min, x_max = (coord_stack[:,1].min(), coord_stack[:,1].max(),
                                  coord_stack[:,0].min(), coord_stack[:,0].max())
    fig, ax = plt.subplots(figsize=fig_dims)
    for rs, rc in zip(segment_list, color_list):
        ax.add_collection(LineCollection(rs, colors=rc))
    if ruler_dims:
        ax.set_ylim((ruler_dims[0], ruler_dims[1]))
        ax.set_xlim((ruler_dims[0], ruler_dims[1]))
    elif equalize_dims:
        all_max, all_min = np.max((y_max, x_max)), np.min((y_min, x_min))
        ax.set_ylim((all_min - 0.1, all_max + 0.1))
        ax.set_xlim((all_min - 0.1, all_max + 0.1))
    else:
        ax.set_ylim((y_min-0.1, y_max+0.1))
        ax.set_xlim((x_min-0.1, x_max+0.1))
    plt.axis('off')
    plt.tight_layout()
    return fig


def get_ruler_segments(lens, cmap):
    colors = [cmap(x) for x in np.linspace(0, 1, len(lens))]
    ruler_segments = []
    ruler_colors = []
    for li, l in enumerate(lens):
        cur_segments = draw_ruler(l, li)
        ruler_segments.append(cur_segments)
        ruler_colors.append([colors[li]] * len(cur_segments))
        list(chain.from_iterable(cur_segments))

    # Add a legend
    ruler_legend = draw_ruler(10, li + 1)
    ruler_segments.append(ruler_legend)
    return ruler_segments, ruler_colors


def draw_ruler_stack(lens, cmap, svg_fn, ruler_dims=None):
    ruler_segments, ruler_colors = get_ruler_segments(lens, cmap)
    fig = plot_linecollections(ruler_segments, ruler_colors, (10, 10), True, ruler_dims)
    fig.savefig(svg_fn)
    plt.close(fig)
    return ruler_segments, ruler_colors


def plot_fp(fp, fig_dims, svg_fn, cmap=mpl.colors.LinearSegmentedColormap.from_list('oranges_custom', ('#fee8c8', '#e34a33'))):
    segments = [[(fv, 0), (fv, 1)] for fv in fp]
    colors = [cmap(x) for x in np.linspace(0, 1, len(segments))]
    fig, ax = plt.subplots(figsize=fig_dims)
    ax.add_collection(LineCollection(segments, colors=colors))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('FRET (E)')
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    plt.tight_layout()
    fig.savefig(svg_fn)
    plt.close(fig)

def draw_triangle(cm, cmap, ax):
    """
    Add a triangle of coordinates to a pyplot axis
    """
    cmidx_list = [np.linalg.norm(cm[i] - cm[(i + 1) % 3]) for i in range(len(cm))]
    dist_list = distance_matrix(cm, cm)[np.triu_indices(3,1)]
    ruler_segments, ruler_colors = get_ruler_segments(cmidx_list, cmap)

    ruler_triangle = []
    for ri, rs in enumerate(ruler_segments[:3]):
        ruler_coords = np.array(list(chain.from_iterable(rs)))
        ruler_coords -= ruler_coords[0]
        rlen = np.linalg.norm(ruler_coords[1])
        cmi = np.argmin(np.abs([rlen - cl for cl in cmidx_list]))

        # rotate and translate
        rotmat = rotation_matrix_from_vectors(ruler_coords[1], (cm[(cmi + 1) % 3] - cm[cmi])[:2])
        ruler_coords = np.dot(rotmat, ruler_coords.T).T + cm[cmi][:2]
        ruler_tuples = [tuple(x) for x in np.split(ruler_coords, len(rs), axis=0)]
        ruler_triangle.append(ruler_tuples)

    # add legend
    ruler_triangle.append(ruler_segments[-1])

    # add to axis
    add_line_collection(ruler_triangle, ruler_colors, ax)


def plot_fp_hist(fp_list, fig_dims, svg_fn, cmap=mpl.colors.LinearSegmentedColormap.from_list('oranges_custom', ('#fee8c8', '#e34a33'))):
    fp_array = np.vstack(fp_list)
    nb_values = fp_array.shape[1]
    colnames = [f'v{n}' for n in range(fp_array.shape[1])]
    colors = [cmap(x) for x in np.linspace(0, 1, fp_array.shape[1])]
    fp_df = pd.DataFrame(fp_array, columns=colnames)
    fp_df_melt = fp_df.melt()
    fig, ax = plt.subplots(nb_values, 1, figsize=fig_dims)
    for ci, cn in enumerate(colnames):
        sns.histplot(x='value', data=fp_df_melt.query(f'variable == "{cn}"'), bins=np.arange(0,1,0.01), ax=ax[ci])
        ax[ci].set_xlim([0, 1])
        ax[ci].set_xlabel('FRET (E)')
        # ax.set_yticks([])
        ax[ci].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.tight_layout()
    fig.savefig(svg_fn)
    plt.close(fig)


def test_plot_triang(coord_dict, cmap_dict=None):
    if cmap_dict is None: cmap_dict = default_cmap_dict
    # --- TEST ---
    fig, ax = plt.subplots(figsize=(10, 10))
    segment_list = []
    for cmidx, cmi in enumerate(coord_dict):
        cm = coord_dict[cmi].copy()
        # if len(cm) == 3:
        #     cm = np.vstack((cm, cm[-1, :]))
        draw_triangle(cm, cmap_dict[['blue', 'orange'][cmidx]], ax)
        segment_list.extend(cm)
    coord_stack = np.vstack(segment_list)
    y_min, y_max, x_min, x_max = (coord_stack[:, 1].min(), coord_stack[:, 1].max(),
                                  coord_stack[:, 0].min(), coord_stack[:, 0].max())
    ax.set_ylim((y_min - 0.1, y_max + 0.1))
    ax.set_xlim((x_min - 0.1, x_max + 0.1))

    plt.axis('off')
    plt.tight_layout()
    fig.show()
