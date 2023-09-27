import argparse, pickle, sys
from os.path import dirname
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from tempfile import NamedTemporaryFile
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.append(f'{dirname(Path(__file__).resolve())}/..')
from helper_functions import np2embedded


class XGBClassifier_txt(object):
    def __init__(self, xgb_txt, enc):
        self.xgb_txt = xgb_txt
        self.enc = enc
        self.classifier_type = 'xgb'


def main(fn_list, embedder, classifier_type, classifier_fn, leave_out_fold, nb_folds, feature_mode, cores):
    X, y, pi = [], [], 0
    for pkl_fn in fn_list:
        with open(pkl_fn, 'rb') as fh: peak_dict = pickle.load(fh)
        Xc = np2embedded(peak_dict['data'], embedder, feature_mode)
        msk = np.ones(len(Xc), dtype=bool)
        if leave_out_fold >= 0 and nb_folds >= 0:
            nb_examples_per_fold = len(Xc) // nb_folds
            msk[np.arange(leave_out_fold * nb_examples_per_fold, (leave_out_fold+1) * nb_examples_per_fold)] = False
        elif leave_out_fold >= 0:
            msk[leave_out_fold] = False
        Xc = Xc[msk]
        yc = [peak_dict['up_id']] * len(Xc)
        X.extend(Xc)
        y.extend(yc)
    if classifier_type == 'xgb':
        encoder = LabelEncoder().fit(y)
        y_enc = encoder.transform(y)
        X_stacked = np.vstack(X)
        if len(encoder.classes_) > 2:
            obj_str = 'multi:softmax'
        else:
            obj_str = 'binary:logistic'
        mod = XGBClassifier(use_label_encoder=False, objective=obj_str, eval_metric='mlogloss',
                            n_estimators=500, n_jobs=cores)
        mod.fit(X_stacked, y_enc)
        tf = NamedTemporaryFile(suffix='.json', delete=True)
        mod.save_model(tf.name)
        with open(tf.name, 'r') as fh:
            mod_txt = fh.read()
        tf.close()
        classifier_out = XGBClassifier_txt(mod_txt, encoder)
        # out_tuple = (mod_txt, encoder)
        with open(classifier_fn, 'wb') as fh: pickle.dump(classifier_out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    elif classifier_type == 'rf':
        rfc = RandomForestClassifier(n_estimators=750)
        rfc.fit(X, y)
        rfc.classifier_type = 'rf'
        with open(classifier_fn, 'wb') as fh: pickle.dump(rfc, fh, protocol=pickle.HIGHEST_PROTOCOL)
    elif classifier_type == 'svm':
        mod = SVC().fit(X, y)
        mod.classifier_type = 'svm'
        with open(classifier_fn, 'wb') as fh:
            pickle.dump(mod, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError('Not a recognized classifier type')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier of choice on embedded data.')
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--classifier-type', type=str, default='rf',
                        help='Type of classifier, choose from random forest [rf], [svm] ... [default: rf]')
    parser.add_argument('--peaks-dir', type=str, required=True,
                        help='Directory of pickled dicts containing coordinates, as made by recompose_coordinates.py')
    parser.add_argument('--classifier-out', type=str, required=True)
    parser.add_argument('--leave-out-fold', type=int, default=-1)
    parser.add_argument('--nb-folds', type=int, default=-1)
    parser.add_argument('--feature-mode', type=str, choices=['fret', 'both'], default='both')
    parser.add_argument('--cores', type=int, default=4,
                        help='Only has effect if multiprocessing is available for classifier')
    args = parser.parse_args()

    fn_list = [fn for fn in Path(args.peaks_dir).iterdir() if fn.suffix == '.pkl']
    with open(args.embedding, 'rb') as fh:
        embedding = pickle.load(fh)
    main(fn_list, embedding, args.classifier_type, args.classifier_out, args.leave_out_fold, args.nb_folds, args.feature_mode, args.cores)