import argparse, pickle, sys
import os.path
from os.path import dirname
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from geometricus import GeometricusEmbedding, MomentInvariants, SplitType, MomentType
import prody as pr
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from tempfile import NamedTemporaryFile
from xgboost import XGBClassifier

sys.path.append(f'{dirname(Path(__file__).resolve())}/..')
from helper_functions import parse_output_path, get_FRET_efficiency, make_atom_group, np2embedded


class XGBClassifier_txt(object):
    def __init__(self, xgb_txt, enc):
        self.xgb_txt = xgb_txt
        self.enc = enc
        self.classifier_type = 'xgb'

def load_xgb_classifier(cl):
    tf = NamedTemporaryFile(suffix='.json', delete=True)
    with open(tf.name, 'w') as fh: fh.write(cl.xgb_txt)
    mod = XGBClassifier()
    mod.load_model(tf.name)
    tf.close()
    return mod, cl.enc


def main(fn_list, embedder, classifier, leave_out_fold, nb_folds, feature_mode, out_csv):
    X, y, nb_tags_list, pi = [], [], [], 0
    pr_df_list = []
    for pkl_fn in fn_list:
        with open(pkl_fn, 'rb') as fh: peak_dict = pickle.load(fh)
        nb_tags = sum([peak_dict['data'][tr]['nb_tags'] for tr in peak_dict['data']])
        pr_df_list.append(pd.Series({'up_id': peak_dict['up_id'],
                                      'nb_tags': nb_tags}))
        Xc = np2embedded(peak_dict['data'], embedder, feature_mode)
        if leave_out_fold >= 0 and nb_folds >= 0:
            nb_examples_per_fold = len(Xc) // nb_folds
            msk = np.zeros(len(Xc), dtype=bool)
            msk[np.arange(leave_out_fold * nb_examples_per_fold, (leave_out_fold+1) * nb_examples_per_fold)] = True
        elif leave_out_fold >=0:
            msk = np.zeros(len(Xc), dtype=bool)
            msk[leave_out_fold] = True
        else:
            msk = np.ones(len(Xc), dtype=bool)
        Xc = Xc[msk]
        yc = [peak_dict['up_id']] * len(Xc)
        X.extend(Xc)
        y.extend(yc)
        nb_tags_list.extend([nb_tags] * len(yc))

    y = np.array(y)
    pr_df = pd.concat(pr_df_list, axis=1).T.set_index('up_id')

    if classifier.classifier_type == 'xgb':
        xgb_classifier, encoder = load_xgb_classifier(classifier)
        y_pred = encoder.inverse_transform(xgb_classifier.predict(np.vstack(X)))
    elif classifier.classifier_type == 'rf':
        y_pred = classifier.predict(X)
    elif classifier.classifier_type == 'svm':
        y_pred = classifier.predict(X)
    else:
        raise ValueError('Not a recognized classifier type')

    df = pd.DataFrame({'pdb_id': y, 'mod_id': leave_out_fold,
                       'nb_tags': nb_tags_list, 'pdb_id_pred': y_pred, 'pred': y_pred == y})
    df.to_csv(out_csv, index=False)

    # Evaluation todo: refactor
    # pr_df.loc[:, 'precision'] = precision_score(y, y_pred, labels=pr_df.index, average=None)
    # pr_df.loc[:, 'recall'] = recall_score(y, y_pred, labels=pr_df.index, average=None)
    # pr_df.loc[:, 'f1'] = 2 / (pr_df.precision ** -1 + pr_df.recall ** -1)
    # pr_df.loc[:, 'nb_tags_same_frac'] = pr_df.apply(lambda x: np.sum(x.nb_tags == pr_df.nb_tags) / len(pr_df), axis=1)
    # pr_df.to_csv(f'{out_dir}perf.csv')
    #
    # summary_txt = (f'accuracy: {accuracy_score(y, y_pred)}\n'
    #                f'precision: {precision_score(y, y_pred, average="micro")}\n'
    #                f'recall: {recall_score(y, y_pred, average="micro")}\n')
    # print(summary_txt)
    #
    # with open(f'{out_dir}summary_metrics.yaml', 'w') as fh:
    #     fh.write(summary_txt)
    #
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # sns.scatterplot(x='recall', y='precision', data=pr_df, ax=ax[0])
    # ll = min(pr_df.precision.min(), pr_df.recall.min()) - 0.01
    # ax[0].set_ylim(ll, 1); ax[0].set_xlim(ll, 1)
    # ax[0].set_aspect('equal')
    #
    # sns.lineplot(x='nb_tags', y='f1', data=pr_df, ax=ax[1])
    # sns.lineplot(x='nb_tags_same_frac', y='f1', data=pr_df, ax=ax[2])
    # for up_id, tup in pr_df.iterrows():
    #     ax[0].text(x=tup.recall, y=tup.precision, s=up_id, fontsize=5)
    # fig.savefig(f'{out_dir}pr_plot.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify samples given a classifier object.')
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--classifier', type=str, required=True)
    parser.add_argument('--peaks-dir', type=str, required=True,
                        help='Directory of pickled dicts containing coordinates, as made by recompose_coordinates.py')
    parser.add_argument('--out-csv', type=str, required=True)
    parser.add_argument('--leave-out-fold', type=int, default=-1)
    parser.add_argument('--nb-folds', type=int, default=-1)
    parser.add_argument('--feature-mode', type=str, choices=['fret', 'both'], default='both')
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    # out_dir = parse_output_path(args.out_dir)

    if os.path.isfile(args.peaks_dir):
        fn_list = [args.peaks_dir]
    else:
        fn_list = [fn for fn in Path(args.peaks_dir).iterdir() if fn.suffix == '.pkl']
    with open(args.embedding, 'rb') as fh:
        embedding = pickle.load(fh)
    with open(args.classifier, 'rb') as fh:
        classifier = pickle.load(fh)

    main(fn_list, embedding, classifier, args.leave_out_fold, args.nb_folds, args.feature_mode, args.out_csv)
