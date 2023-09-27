import argparse, sys, pickle
from pathlib import Path
from scipy.spatial import distance_matrix
from itertools import combinations

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from recompose_coordinates import fill_idx_mat
from helper_functions import get_FRET_distance, plot_structure
from triangle_reconstruction.numpy2wavefront import create_obj_file

with open(f'{Path(__file__).parents[1].resolve()}/data/preindexed_dms.pkl', 'rb') as fh: dm_idx_dict = pickle.load(fh)


def fret2coords(dist_list, nb_dyes):
    m = np.zeros((nb_dyes, nb_dyes))
    coord_mats_list = []
    dm_list = [fill_idx_mat(dmi, dist_list) for dmi in dm_idx_dict[nb_dyes]]
    dm_check_list = []
    for dist_mat in dm_list:
        for i in range(0, nb_dyes):
            for j in range(0, nb_dyes):
                m[i, j] = (dist_mat[0, j] ** 2 + dist_mat[0, i] ** 2 - dist_mat[i, j] ** 2) / 2
        s, u = np.linalg.eig(m)
        if np.any(s < 0): continue  # negative semi-definite gram matrix --> no embedding possible
        u = u[:, np.argsort(s)[::-1]]
        s = np.sort(s)[::-1]
        coord_mat = np.zeros((nb_dyes,3))
        max_nb_dims = min(nb_dyes, 3)
        coord_mat[:, :max_nb_dims] = np.matmul(u[:, :3], np.diag(np.sqrt(s[:3])))
        coord_mats_list.append(coord_mat)
        dist_mat_rev = distance_matrix(coord_mat, coord_mat)
        dm_check_list.append(np.sum(np.square(dist_mat - dist_mat_rev)))
    if len(dm_check_list) == 0:
        return None  # no embedding possible
    return coord_mats_list[np.argmin(dm_check_list)], np.min(dm_check_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given list of fret values, reconstruct 3D coordinates of dyes')
    parser.add_argument('--fret', type=float, required=True, nargs='+')
    parser.add_argument('--nb-dyes', type=int, required=True)
    parser.add_argument('--blender-obj', type=str)
    args = parser.parse_args()
    dist_list = [get_FRET_distance(x) for x in args.fret]
    cm, err = fret2coords(dist_list,args.nb_dyes)
    plot_structure({'x': cm})
    print(cm)
    print('distance matrix:')
    print(distance_matrix(cm, cm))
    print(f'error: {err}')
    if args.blender_obj is not None:
        obj_dict = {f'e{ii}': [cm[i[0]], cm[i[1]]] for ii, i in enumerate(combinations(np.arange(len(cm)), 2))}
        orb_dict = {f'o{ii}': {'coords': c, 'size': 1.0, 'color': (215 / 255, 25 / 255, 28 / 255, 1.0)} for ii, c in enumerate(cm)}
        create_obj_file(obj_dict, args.blender_obj, orb_dict)
