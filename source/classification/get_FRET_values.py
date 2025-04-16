import argparse, re, os, sys, pickle
from os.path import dirname
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial import distance_matrix
from scipy.stats import hmean
from itertools import combinations
import pandas as pd
import numpy as np
import prody as pr
import tqdm
import random
from copy import copy

sys.path.append(f'{dirname(Path(__file__).resolve())}/..')
from helper_functions import parse_output_path, get_FRET_efficiency, get_FRET_distance, make_atom_group, get_sigma

def average_dict_values(dict_list, harmonic=False):
    list_dict, out_dict = {}, {}
    for dl in dict_list:
        for k1 in dl:
            for k2 in dl[k1]:
                if k1 not in list_dict:
                    list_dict[k1], out_dict[k1] = {}, {}
                if k2 not in list_dict[k1]:
                    list_dict[k1][k2] = []
                list_dict[k1][k2].append(dl[k1][k2])
    for k1 in list_dict:
        for k2 in list_dict[k1]:
            if harmonic:
                out_dict[k1][k2] = hmean(list_dict[k1][k2])
            else:
                out_dict[k1][k2] = np.mean(list_dict[k1][k2])
    return out_dict




def simulate_FRET(atom_dict, repeats, efficiency, linker_length, noise_sigma):
    res = 0.01
    res_bins = np.arange(0.01, 1 + res, res)
    fret_list, dist_list, coords_list, fp_list = [], [], [], []
    for rep in range(repeats):
        if len(atom_dict) == 2:  # No point in simulating labeling efficiency; no FRET visible if either is missing
            cad = atom_dict
        else:
            cad = {atm: atom_dict[atm] for atm in atom_dict if random.random() < efficiency}
        nb_coordsets = min([len(list(atm[0].getCoordsets())) for atm in cad.values()])
        fret_sub_list, dist_sub_list, coords_sub_list = [], [], []
        for nc in range(nb_coordsets):
            ca_group = make_atom_group(cad, coordset=nc, linker_length=linker_length)
            coords = np.vstack([ca.getCoords() for ca in ca_group])
            fl, dl = atoms_to_FRET_values(ca_group, noise_sigma)
            fret_sub_list.append(fl)
            dist_sub_list.append(dl)
            coords_sub_list.append(coords)
        # fret_list.append(fret_mean := average_dict_values(fret_sub_list))
        dist_list.append(dist_mean := average_dict_values(dist_sub_list))
        fret_list.append(fret_dict := {k1: {k2: get_FRET_efficiency(dist_mean[k1][k2]) for k2 in dist_mean[k1]} for k1 in dist_mean})
        fret_fp = sorted([fret_dict[k[0]][k[1]] for k in list(combinations(fret_dict, 2))], reverse=True)
        fret_hist = np.clip((np.histogram(fret_fp, bins=res_bins)[0] > 0).astype(float), 0, 1)
        fp_list.append(fret_hist)
        coords_list.append(np.mean(np.dstack(coords_sub_list), axis=2))  # todo this will go wrong if imperfect labeling is added!
    return fret_list, dist_list, coords_list, fp_list


def atoms_to_FRET_values(atom_group, noise_sigma):
    fret_dict, dist_dict = {}, {}
    for i1, atm1 in enumerate(atom_group):
        fret_dict[atm1.getResnum()], dist_dict[atm1.getResnum()] = {}, {}
        for i2, atm2 in enumerate(atom_group):
            if atm1.getResnum() == atm2.getResnum(): continue
            # if i1 <= i2: continue  # only calculate upper-triangle combinations
            dist = np.linalg.norm(atm1.getCoords() - atm2.getCoords())
            dist_dict[atm1.getResnum()][atm2.getResnum()] = dist
            fret_dict[atm1.getResnum()][atm2.getResnum()] = (get_FRET_efficiency(dist) + np.random.normal(loc=0.0, scale=noise_sigma)).clip(1e-99, 1-1e-99)
    return fret_dict, dist_dict

def badness_test(coords):
    dm = distance_matrix(coords, coords)
    dm_bool = np.logical_or(dm > get_FRET_distance(0.2), dm < get_FRET_distance(0.95))
    dm_bool[np.eye(len(dm), dtype=bool)] = False
    bad_pairs = np.argwhere(dm_bool)
    return bad_pairs

def filter_distances(atom_dict, linker_length):
    atom_dict = copy(atom_dict)
    ca_group = make_atom_group(atom_dict, coordset=0, linker_length=linker_length)
    coords_list, resi_tuple = zip(*[(ca.getCoords(), ca.getResnum()) for ca in ca_group])
    resi_list = list(resi_tuple)
    coords = np.vstack(coords_list)
    while len(atom_dict) > 1 and len(bad_pairs := badness_test(coords)) > 0:
        bad_idx, bi_counts = np.unique(bad_pairs, return_counts=True)
        bad_idx = bad_idx[np.argsort(bi_counts)]
        # remove_resi = list(atom_dict.keys())[bad_idx[-1]]
        del atom_dict[resi_list.pop(bad_idx[-1])]
        ca_group = make_atom_group(atom_dict, coordset=0, linker_length=linker_length)
        coords = np.vstack([ca.getCoords() for ca in ca_group])
    return atom_dict


def parallel_get_FRET_values(args):
    (up_id, buried_df, af_dict, fail_log, repeats, efficiency, anchor_type,
     max_nb_tags, linker_length, noise_sigma, train_test, out_path, tagged_resn) = args
    try:
        if buried_df is None:
            bdf = None
        else:
            bdf = buried_df.query(f'up_id == "{up_id}"').copy()
            bdf.query('exposed == "exposed"', inplace=True)
            nb_tags = len(bdf)
            if nb_tags == 0:
                with open(fail_log, 'a') as fh: fh.write(f'{up_id}\tnot in buried_txt or no exposed tagged residues\n')
                return
            elif nb_tags > max_nb_tags:
                with open(fail_log, 'a') as fh: fh.write(f'{up_id}\tcontains too many tagged residues: {nb_tags}\n')
                return

        # Retrieve pdb file
        if pdb_fn_list := af_dict.get(up_id, False):
            prot_list = [pr.parsePDB(pdb_fn) for pdb_fn in pdb_fn_list]
        else:
            with open(fail_log, 'a') as fh: fh.write(f'{up_id}\tnot in pdb_dir\n')
            return

        pct90 = min(int(round(len(prot_list) * 0.9)), len(prot_list))
        if train_test == 'train':
            prot_list = prot_list[:pct90]
        elif train_test == 'test':
            prot_list = prot_list[pct90:]

        data_dict = {resn: {'nb_tags': 0, 'fret':[], 'dist': [], 'fingerprint': [], 'coords': []} for resn in tagged_resn}
        for prot in prot_list:
            for cur_tagged_resn in tagged_resn:
                # extract tagged CA atoms
                if bdf is None:
                    resn_resi_list = [(res.getResname(), res.getResnum())
                                      for res in prot.iterResidues() if res.getResname() == cur_tagged_resn]
                    if resn_resi_list[0][1] != 1:
                        resn_resi_list = [(cur_tagged_resn, 1)] + resn_resi_list
                    if not len(resn_resi_list): continue
                    resn_list, resi_list = zip(*resn_resi_list)
                else:
                    resn_list = [prot.select(f'resnum {ri}').getResnames()[0] for ri in bdf.resi]
                    resi_list = list(bdf.resi)
                if not all([resn in tagged_resn for resn in resn_list]):
                    with open(fail_log, 'a') as fh: fh.write(f'{up_id}\tnot all residues are of tagged type \n')
                    return
                resi_list = sorted(resi_list)
                atom_dict = {resi: (prot.select(f'resnum {resi} name CA'), prot.select(f'resnum {resi} name CB'))
                             for resi in resi_list}
                atom_dict = {resi: atom_dict[resi] for resi in atom_dict if atom_dict[resi][1] is not None}
                if not len(atom_dict): continue

                # filter out atoms too close or too far away from others to be accurately estimated
                # atom_dict = filter_distances(atom_dict, linker_length)

                if (anchor_type == 'tyr' and len(atom_dict) < 3) or len(atom_dict) < 2:
                    continue
                    # with open(fail_log, 'a') as fh: fh.write(f'{up_id}\tless than 2 exposed tagged residues\n')
                    # return

                if anchor_type == 'tyr':
                    repeats_per_anchor = repeats // len(atom_dict)
                    fret_list, dist_list, coords_list, fp_list = [], [], [], []
                    for resi in atom_dict:
                        cur_atom_dict = {ii: atom_dict[ii] for ii in atom_dict if ii != resi}
                        fl, dl, coords_list, fp = simulate_FRET(cur_atom_dict, repeats_per_anchor, efficiency, linker_length, noise_sigma)
                        fret_list.extend(fl)
                        dist_list.extend(dl)
                        fp_list.extend(fp)
                else:
                    fret_list, dist_list, coords_list, fp_list = simulate_FRET(atom_dict, repeats, efficiency, linker_length, noise_sigma)
                data_dict[cur_tagged_resn]['fret'].extend(fret_list)
                data_dict[cur_tagged_resn]['dist'].extend(dist_list)
                data_dict[cur_tagged_resn]['fingerprint'].extend(fp_list)
                data_dict[cur_tagged_resn]['coords'].extend(coords_list)
                data_dict[cur_tagged_resn]['nb_tags'] = max(data_dict[cur_tagged_resn]['nb_tags'], len(atom_dict))
        if all([data_dict[cur_tagged_resn]['nb_tags'] == 0 for cur_tagged_resn in tagged_resn]):
            with open(fail_log, 'a') as fh: fh.write(f'{up_id}\tNo tags for any of tagged residue types\n')
            return

        fret_dict = {
            'up_id': up_id,
            'anchor_type': anchor_type,
            'efficiency': efficiency,
            'nb_examples': len(prot_list),
            'data': data_dict
        }

        with open(f'{out_path}{up_id}.pkl', 'wb') as fh:
            pickle.dump(fret_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        with open(fail_log, 'a') as fh:
            fh.write(f'{up_id}\tother error: {e}\n')
            return
    return


def main(up_list_fn, buried_fn, pdb_dir, pdb_pattern, out_dir, efficiency, max_nb_tags,
         anchor_type, linker_length, repeats, resolution, train_test,
         cores):
    pr.LOGGER.delHandler(0)  # remove prody logger, avoid annoying prints to stderr
    noise_sigma = get_sigma(0.5, resolution)
    out_dir = parse_output_path(out_dir, clean=True)
    fp_dir = parse_output_path(out_dir + 'fingerprints', clean=True)
    pdb_dir = str(Path(pdb_dir).resolve()) + '/'
    fail_log = f'{out_dir}data_preparation_fail_log.txt'
    if buried_fn is None:
        buried_df = None
    else:
        buried_df = pd.read_csv(buried_fn, sep='\t', names=['up_id', 'exposed', 'resi'])

    af_dict = {}
    pdb_id_pattern = re.compile(pdb_pattern)
    for fn in os.listdir(pdb_dir):
        if not fn.endswith('.pdb') and not fn.endswith('.pdb.gz'): continue
        pdb_id = re.search(pdb_id_pattern, fn).group(0)
        if pdb_id in af_dict:
            af_dict[pdb_id].append(pdb_dir + fn)
        else:
            af_dict[pdb_id] = [pdb_dir + fn]

    if up_list_fn is None:
        up_list = list(af_dict)
    else:
        with open(up_list_fn, 'r') as fh: up_list = fh.read().splitlines()


    # af_dict = {re.search('(?<=AF-)[^-]+', fn).group(0):
    #                pdb_dir + fn for fn in os.listdir(pdb_dir) if fn.endswith('pdb.gz')}

    arg_list=[(up_id, buried_df, af_dict, fail_log, repeats, efficiency,
               anchor_type, max_nb_tags, linker_length, noise_sigma, train_test, fp_dir, args.tagged_resn)
              for up_id in up_list]
    with Pool(cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(parallel_get_FRET_values, arg_list),total=len(up_list)):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert structures to FRET values')
    # --- I/O ---
    parser.add_argument('--up-list', type=str, default=None,
                        help='text file with uniprot IDs for which to produce FRET values, 1 per line.')
    parser.add_argument('--buried-txt', type=str, default=None,
                        help='txt file detailing of each Y residue per structure whether it is buried.')
    parser.add_argument('--train-test', type=str, default='all', choices=['all', 'train', 'test'],
                        help='Either pass on [all], a 20 percent [test] set or an 80 percent [train] set [default: all]')
    parser.add_argument('--pdb-dir', type=str, required=True,
                        help='Directory pf pdb.gz files of structures')
    parser.add_argument('--pdb-pattern', type=str, default='(?<=AF-)[^-]+',
                        help='Pattern by which to parse the name of a pdb file. Defaults to pattern seen in Alphafold'
                             'filenames [default: (?<=AF-)[^-]+]')
    parser.add_argument('--out-dir', type=str, required=True)
    # --- simulation params ---
    parser.add_argument('--efficiency', type=float, default=1.0,
                        help='Labeling efficiency, P of labeling a targeted Tyr [default: 1.0]')
    parser.add_argument('--linker-length', type=float, default=0,
                        help='Distance between CA and dye, extended in direction of CB [default: 0]')
    parser.add_argument('--anchor-type', type=str, choices=['tyr', 'other'], default='other',
                        help='Define whether anchoring is taking up one of the tyrs [tyr], or by some other means,'
                             'e.g. N-terminus [other] [default: tyr]')
    parser.add_argument('--tagged-resn', type=str, nargs='+', default=['TYR'],
                        help='three-letter codes (capitals) of tagged residue types.')
    parser.add_argument('--resolution', type=float, default=0.01,
                        help='E_FRET resolution, will be used in parameterizing noise [default: 0.01]')
    parser.add_argument('--max-nb-tags', type=int, default=np.inf,
                        help='Maximum number of tyrs in structure if more, discard [default: no max]')
    # --- simulation size ---
    parser.add_argument('--repeats', type=int, default=50,
                        help='Number of times each protein is sampled [default: 50]')
    # --- misc ---
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()

    main(up_list_fn=args.up_list, buried_fn=args.buried_txt, pdb_dir=args.pdb_dir, pdb_pattern=args.pdb_pattern,
         repeats=args.repeats, efficiency=args.efficiency, max_nb_tags=args.max_nb_tags,
         anchor_type=args.anchor_type, linker_length=args.linker_length, resolution=args.resolution, train_test=args.train_test,
         out_dir=args.out_dir, cores=args.cores)
