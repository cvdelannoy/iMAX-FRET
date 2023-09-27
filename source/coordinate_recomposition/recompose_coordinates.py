import argparse, pickle, sys, os
import numpy as np
from os.path import dirname

from pathlib import Path
from itertools import permutations, combinations, combinations_with_replacement, chain
from math import factorial
from scipy.spatial import distance_matrix
import tqdm
from multiprocessing import Pool
sys.path.append(f'{dirname(Path(__file__).resolve())}/..')
from helper_functions import parse_output_path, get_FRET_distance, get_aligned_rmsd
with open(f'{dirname(Path(__file__).parents[0].resolve())}/preindexed_dms.pkl', 'rb') as fh: dm_idx_dict = pickle.load(fh)


def list2distmat(x, dm_size):
    out_mat = np.zeros((dm_size, dm_size))
    msk = np.invert(np.tri(dm_size, dm_size, 0, dtype=bool))
    out_mat[msk] = x
    out_mat += np.rot90(np.fliplr(out_mat))
    return out_mat


def coords2distlist(x):
    out_list = []
    nb_coords = len(x)
    for i in range(nb_coords):
        for j in range(nb_coords):
            if i >= j: continue
            out_list.append(np.linalg.norm(x[i] - x[j]))
    out_list.sort()
    return out_list


def fill_idx_mat(idx_mat, value_list):
    dm = np.zeros_like(idx_mat, dtype=float)
    for i, v in enumerate(value_list):
        dm[idx_mat == i] = v
    dm[np.diag_indices_from(dm)] = 0
    return dm


def get_grouped_nb_mats(nb_tags):
    out = 1
    for i in range(1, nb_tags):
        out *= factorial(i)
    return out


def dists2coords(dist_dict, grouped_peaks, true_coords, nb_free_dyes=None):
    # nb_free_dyes = int((1 + np.sqrt(1 + 8 * len(dist_dict))) / 2)
    if not nb_free_dyes:
        nb_free_dyes = len(dist_dict)
    m = np.zeros((nb_free_dyes, nb_free_dyes))
    coord_mats_list = []
    ev_sum_list = []
    if grouped_peaks:  # todo update to preindexed approach
        # assume that all FRET values can be grouped during recording based on their acceptor reference point
        dist_mat = np.zeros((nb_free_dyes, nb_free_dyes))
        resi_list = list(dist_dict)

        #  fix first line (reference)
        dist_mat[0, 1:] = list(dist_dict[resi_list[0]].values())
        dist_mat[1:, 0] = list(dist_dict[resi_list[0]].values())

        # Each next line: fix all - 1+i values and randomly combine the rest
        def recurse_distmat(dm_list, se_list, dd, depth, max_depth):
            out_dm_list, out_se_list = [], []
            dists = np.array(list(dd[resi_list[depth]].values()))
            for se, dm in zip(se_list, dm_list):
                for fd in dm[depth, :depth]:
                    # remove values that are likely filled by previous rows
                    candidate_errs = dists - fd
                    dists = np.delete(dists, np.argmin(candidate_errs))
                    se += min(candidate_errs)
                if not len(dists):
                    out_dm_list.append(dm)
                    out_se_list.append(se)
                for ordered_dists in permutations(dists, len(dists)):
                    # generate every order of values that are still free to be filled in
                    dm_new = dm.copy()
                    dm_new[depth, depth+1:] = ordered_dists
                    dm_new[depth + 1:, depth] = ordered_dists
                    out_dm_list.append(dm_new)
                    out_se_list.append(se)
            if depth < max_depth - 1:
                recurse_distmat(out_dm_list, out_se_list, dd, depth + 1, max_depth)
            return out_dm_list, out_se_list
        dm_list, se_list = recurse_distmat([dist_mat], [0], dist_dict, 1, nb_free_dyes)
    else:
        # do not assume any knowledge of origin of FRET values
        dist_list = []
        resi_list = list(dist_dict)
        for i1, i2 in combinations(resi_list, 2):
            dist_list.append(dist_dict[i1][i2])
        dist_list.sort()
        dm_list = [fill_idx_mat(dmi, dist_list) for dmi in dm_idx_dict[nb_free_dyes]]
    # rmsd_list = []
    dm_check_list = []
    for dist_mat in dm_list:
        for i in range(0, nb_free_dyes):
            for j in range(0, nb_free_dyes):
                m[i, j] = (dist_mat[0, j] ** 2 + dist_mat[0, i] ** 2 - dist_mat[i, j] ** 2) / 2
        s, u = np.linalg.eig(m)
        if np.any(s < 0):
            continue  # negative semi-definite gram matrix --> no embedding possible
        u = u[:, np.argsort(s)[::-1]]
        s = np.sort(s)[::-1]
        coord_mat = np.zeros((nb_free_dyes,3))
        max_nb_dims = min(nb_free_dyes, 3)
        coord_mat[:, :max_nb_dims] = np.matmul(u[:, :3], np.diag(np.sqrt(s[:3])))
        coord_mats_list.append(coord_mat)
        dist_mat_rev = distance_matrix(coord_mat, coord_mat)
        dm_check_list.append(np.sqrt(np.mean(np.square(dist_mat - dist_mat_rev))))
    if len(coord_mats_list) == 0:
        return np.array([[]]), np.inf  # none of the embeddings were possible
    return coord_mats_list[np.argmin(dm_check_list)], np.min(dm_check_list)


def l2d(ll):
    nbt = int((1 + np.sqrt(1 + 8 * len(ll))) / 2)
    out_dict = {i1: {i2: None for i2 in range(nbt) if i2 != i1} for i1 in range(nbt)}
    for i1 in range(nbt-1):
        for i2 in range(i1 + 1, nbt):
            x = ll[i1 + i2 - 1]
            out_dict[i1][i2] = x
            out_dict[i2][i1] = x
    return out_dict


def dist_list2dict(dl, nb_tags, grouped_peaks, coords, force_nb_dists=None):
    nb_dyes_expected = int(np.ceil((1 + np.sqrt(1 + 8 * len(dl))) / 2))
    nb_dists_expected = int((nb_dyes_expected ** 2 - nb_dyes_expected)  / 2)
    # Frequently, peaks are overlapping so that some seem missing. Try all combos and pick the one with the best score
    dl = np.array(dl)
    dl_list = []
    if force_nb_dists is not None:
        if len(dl) != force_nb_dists:
            raise ValueError(f'Nb dists is {len(dl)}, precisely {force_nb_dists} is enforced in this run')
    if len(dl) < nb_dists_expected:
        nb_missing_dists = nb_dists_expected - len(dl)
        dist_combos = (combinations(np.arange(len(dl)), nb_missing_dists) if nb_missing_dists <= len(dl) else
                       combinations_with_replacement(np.arange(len(dl)), nb_missing_dists))
        for x in dist_combos:
                doubles = [dl[xc] for xc in x]
                cur_dists = np.concatenate((dl, doubles))
                dl_list.append(l2d(cur_dists))
    else:
        dl_list.append(l2d(dl))
    cm_list = [dists2coords(dl_cur, grouped_peaks, coords) for dl_cur in dl_list]
    best_cm = sorted(cm_list, key=lambda x: x[1])[0]
    if best_cm[1] is np.inf:
        raise ValueError('Impossible coordinate combination!')
    return best_cm[0]


def parallel_recompose(args):
    pkl_fn, coord_dir, max_nb_tags, input_type, grouped_peaks, forced_nb_dists, log, fail_log = args
    with open(pkl_fn, 'rb') as fh:
        peak_dict = pickle.load(fh)
    rmsd_dict = {}
    nb_tags_list = [peak_dict['data'][tagged_resn]['nb_tags'] for tagged_resn in list(peak_dict['data'])]
    if all([t < 2 or t > max_nb_tags for t in nb_tags_list]):
        with open(fail_log, 'a') as fh:
            fh.write(f'{peak_dict["up_id"]}\t<2 or >{max_nb_tags} tags for all tagged residues\n')
        return
    for tagged_resn in peak_dict['data']:
        rmsd_dict[tagged_resn] = []
        nb_tags = peak_dict['data'][tagged_resn]['nb_tags']
        if nb_tags < 2 or nb_tags > max_nb_tags:
            peak_dict['data'][tagged_resn]['recomposed_coords'] = [np.zeros((1,3), dtype=float)] * peak_dict['nb_examples']
            if nb_tags == 0:
                peak_dict['data'][tagged_resn]['fingerprint'] = [np.zeros((1, 99), dtype=float)] * peak_dict['nb_examples']  # todo this depends on resolution which is hardcoded rn
            peak_dict['data'][tagged_resn]['rmsd'] = None
            peak_dict['data'][tagged_resn]['dl_diff'] = None
            continue

        # solution from: https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
        coord_mat_list, dl_diff_list = [], []
        for di, dl in enumerate(peak_dict['data'][tagged_resn][input_type]):
            if not len(dl): continue
            if input_type == 'fret':
                if type(dl) is dict:
                    dl = {i1: {i2: get_FRET_distance(dl[i1][i2]) for i2 in dl[i1]} for i1 in dl}
                    dl_list_fret = np.array(list(chain.from_iterable([[dl[i1][i2] for i2 in dl[i1]] for i1 in dl])))
                    dl_list_dist = np.array(list(chain.from_iterable([[peak_dict['data'][tagged_resn]['dist'][di][i1][i2] for i2 in dl[i1]] for i1 in dl])))
                elif type(dl) is list:
                    dl_fret = np.array(sorted(dl))
                    dl = dl_list_fret = dl_list_dist = np.array([get_FRET_distance(fv, (0,100)) for fv in dl_fret])
                dl_diff = np.mean(np.abs(dl_list_fret - dl_list_dist))
                dl_diff_list.append(dl_diff)
            if not len(dl) or np.any(np.isnan(dl)):
                continue
            try:
                if type(dl) is not dict:
                    cm = dist_list2dict(dl, nb_tags, grouped_peaks, peak_dict['data'][tagged_resn]['coords'], forced_nb_dists)
                else:
                    cm = dists2coords(dl, grouped_peaks, peak_dict['data'][tagged_resn]['coords'])
                coord_mat_list.append(cm)
            except Exception as e:
                with open(fail_log, 'a') as fh:
                    fh.write(f'{peak_dict["up_id"]}\tother error: {e}\n')
        rmsd_dict[tagged_resn] = [get_aligned_rmsd(recomposed_coords, true_coords)
                     for recomposed_coords, true_coords in zip(coord_mat_list, peak_dict['data'][tagged_resn]['coords'])]
        peak_dict['data'][tagged_resn]['recomposed_coords'] = coord_mat_list
        peak_dict['data'][tagged_resn]['rmsd'] = rmsd_dict[tagged_resn]
        peak_dict['data'][tagged_resn]['dl_diff'] = dl_diff_list
    with open(f'{coord_dir}{peak_dict["up_id"]}.pkl', 'wb') as fh:
        pickle.dump(peak_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
    rmsd_txt = ','.join([str(np.mean(rmsd_dict[tagged_resn])) if len(rmsd_dict[tagged_resn]) else 'nan'
                         for tagged_resn in rmsd_dict])
    nb_tags_txt = ','.join([str(t) for t in nb_tags_list])
    with open(log, 'a') as fh:
        fh.write(f'{peak_dict["up_id"]}\t{nb_tags_txt}\t{rmsd_txt}\n')


def main(input_type, fn_list, out_dir, grouped_peaks, max_nb_tags, forced_nb_dists, cores):
    fail_log = f'{out_dir}recompose_fail_log.txt'
    if os.path.exists(fail_log): os.unlink(fail_log)
    log = f'{out_dir}recompose_log.txt'
    with open(log, 'w') as fh: fh.write('pdb_id\tnb_tags\trmsd\n')
    coord_dir = parse_output_path(f'{out_dir}coords')

    arg_list = [(pkl_fn, coord_dir, max_nb_tags, input_type, grouped_peaks, forced_nb_dists, log, fail_log) for pkl_fn in fn_list]
    with Pool(cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(parallel_recompose, arg_list), total=len(arg_list)):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recompose dye coordinates from a set of FRET values or distances.')
    parser.add_argument('--peaks-dir', type=str, help='Directory of pickled dicts containing FRET and distance values,'
                                                      'made by get_FRET_values.py')
    parser.add_argument('--input-type', type=str, choices=['fret', 'dist'], default='dist',
                        help='Type of data to use as fingerprint, [fret] or [dist]ance [default: dist]')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--forced-nb-dists', type=int, default=None)
    parser.add_argument('--max-nb-tags', type=int, default=np.inf)
    parser.add_argument('--grouped-peaks', action='store_true',
                        help='Use additional information: distances from common reference point are marked as such.')
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    out_dir = parse_output_path(args.out_dir)
    fn_list = [fn for fn in Path(args.peaks_dir).iterdir() if fn.suffix == '.pkl']
    main(args.input_type, fn_list, out_dir, args.grouped_peaks, args.max_nb_tags, args.forced_nb_dists, args.cores)
