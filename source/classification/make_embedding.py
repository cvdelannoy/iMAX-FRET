import argparse, sys, pickle

import numpy as np
import prody as pr

from os.path import dirname
from geometricus import GeometricusEmbedding, MomentInvariants, SplitType
from pathlib import Path

sys.path.append(f'{dirname(Path(__file__).resolve())}/..')
from helper_functions import parse_output_path, get_FRET_efficiency

def make_atom_group(atom_list):
    if atom_list.ndim == 1:
        atom_list = np.expand_dims(atom_list, 0)  # pickling squeezes numpy arrays!
    out_group = pr.atomgroup.AtomGroup('Protein')
    out_group.addCoordset(atom_list)

    out_group.setNames(['CA'] * len(atom_list))
    out_group.setResnums(np.arange(len(atom_list)))
    out_group.setResnames(['TYR'] * len(atom_list))
    return out_group


def main(fn_list, embedding_fn, resolution, radius, leave_out_fold, nb_folds):
    invar_dict = {}

    # Calculate shape-mers
    pi = 0
    y = []
    class_names = []
    for pkl_fn in fn_list:
        with open(pkl_fn, 'rb') as fh: peak_dict = pickle.load(fh)
        class_names.append(peak_dict['up_id'])
        for tagged_resn in peak_dict['data']:
            if peak_dict['data'][tagged_resn]['recomposed_coords'] is None:
                continue
            if leave_out_fold >= 0 and nb_folds >= 0:  # expect multiple examples per fold
                nb_examples_per_fold = len(peak_dict['data'][tagged_resn]['recomposed_coords']) // nb_folds
                fold_index = np.arange(leave_out_fold * nb_examples_per_fold, (leave_out_fold + 1) * nb_examples_per_fold)
            elif leave_out_fold >= 0: # expect 1 example per fold
                fold_index = np.array([leave_out_fold])
            else: # leave no fold out
                fold_index = np.array([])
            for cidx, coord_list in enumerate(peak_dict['data'][tagged_resn]['recomposed_coords']):
                if cidx in fold_index: continue
                atm_group = make_atom_group(coord_list[0])

                invars = MomentInvariants.from_prody_atomgroup(str(pi), atm_group,
                                                               split_type=SplitType.RADIUS,
                                                               split_size=radius)  # todo: would choosing split size based on FRET radius make sense?

                if tagged_resn in invar_dict:
                    invar_dict[tagged_resn].append(invars)
                else:
                    invar_dict[tagged_resn] = [invars]
                y.append(peak_dict['up_id'])
                pi += 1

    # Create embedding
    embedder_dict = {}
    for tagged_resn in invar_dict:
        embedder = GeometricusEmbedding.from_invariants(invar_dict[tagged_resn], resolution=resolution)
        embedder.embedding = None
        embedder.radius = radius
        embedder_dict[tagged_resn] = embedder
    with open(embedding_fn, 'wb') as fh:
        pickle.dump(embedder_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # reducer = umap.UMAP(metric="cosine", n_components=2)
    # reduced = reducer.fit_transform(embedder.embedding)
    #
    # y = np.array(y)
    # fig, ax = plt.subplots(figsize=(10,10))
    # for cn in class_names:
    #     plt.scatter(*np.hsplit(reduced[y == cn, :], 2), label=cn, edgecolor="black", linewidth=0.1, alpha=0.8)
    # plt.show(); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embed coordinates using caretta')
    parser.add_argument('--peaks-dir', type=str, required=True,
                        help='Directory of pickled dicts containing coordinates, as made by recompose_coordinates.py')
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='Resolution used in geometricus, higher is finer-grained [default: 10.0]')
    parser.add_argument('--radius', type=int, default=100,
                        help='Radius used to define invariants in A [default: 100]')
    parser.add_argument('--leave-out-fold', type=int, default=-1)
    parser.add_argument('--nb-folds', type=int, default=-1)
    parser.add_argument('--embedding', type=str, required=True)
    args = parser.parse_args()
    fn_list = [fn for fn in Path(args.peaks_dir).iterdir() if fn.suffix == '.pkl']
    main(fn_list, args.embedding, args.resolution, args.radius, args.leave_out_fold, args.nb_folds)
