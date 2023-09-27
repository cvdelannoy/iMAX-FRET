import argparse, sys
from os.path import dirname
from pathlib import Path
from jinja2 import Template
from snakemake import snakemake
import numpy as np
import pandas as pd

__basedir__ = f'{dirname(Path(__file__).resolve().parents[0])}/source/'
sys.path.append(__basedir__)
from helper_functions import parse_output_path
from plot_triangle_classification import main as ptc

parser = argparse.ArgumentParser(description='Run triangle classifier pipeline.')
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--test-dir', type=str, required=True)  # todo make optional

# --- params ---
parser.add_argument('--efficiency', type=float, default=1.0,
                        help='Labeling efficiency, P of labeling a targeted Tyr [default: 1.0]')
parser.add_argument('--grouped-peaks', action='store_true',
                    help='Use additional information: distances from common reference point are marked as such.')
parser.add_argument('--classifier-type', type=str, default='rf',
                        help='Type of classifier, choose from random forest [rf], [svm] or xgboost [xgb] ... [default: rf]')
parser.add_argument('--geom-resolution', type=float, default=10.0,
                        help='Resolution used in geometricus, higher is finer-grained [default: 10.0]')
parser.add_argument('--radius', type=int, default=80,
                        help='Radius used to define invariants in A [default: 80]')
parser.add_argument('--input-type', type=str, choices=['fret', 'dist'], default='fret',
                        help='Type of data to use as fingerprint, [fret] or [dist]ance [default: fret]')
parser.add_argument('--feature-mode', type=str, choices=['fret', 'both'], default='both',
                    help='[default: both]')
parser.add_argument('--max-nb-tags', type=int, default=5)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--nb-folds', type=int, default=10)
parser.add_argument('--cores', type=int, default=4)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()

out_dir = parse_output_path(args.out_dir, clean=False)
classify_dir = parse_output_path(out_dir + 'classification')

with open(f'{__basedir__}triangle_classifier/run_pipeline_triangle_classifier_trainTest.sf', 'r') as fh:
    sm_txt = fh.read()

sm_out = Template(sm_txt).render(
    # inputs
    __basedir__=__basedir__,
    out_dir=out_dir,
    in_dir=args.in_dir,
    test_dir=args.test_dir,
    classify_dir=classify_dir,
    # params
    grouped_peaks=args.grouped_peaks,
    feature_mode=args.feature_mode,
    classifier_type=args.classifier_type,
    geom_resolution=args.geom_resolution,
    radius=args.radius,
    input_type=args.input_type,
    max_nb_tags=args.max_nb_tags,
    repeats=args.repeats,
    efficiency=args.efficiency,
    nb_folds=args.nb_folds
)

sf_fn = f'{out_dir}run_pipeline.sf'
with open(sf_fn, 'w') as fh: fh.write(sm_out)

snakemake(sf_fn, cores=args.cores, dryrun=args.dryrun)

# --- analysis ---

# cross-validation
result_df = pd.read_csv(f'{classify_dir}prediction_results_all.csv', index_col=None)
accuracy = np.mean(result_df.pred)
acc_std = np.std(result_df.groupby('mod_id').pred.mean())
print(f'overall accuracy CV: {accuracy.round(3)}±{acc_std.round(3)}')
analysis_dir = parse_output_path(f'{out_dir}analysis_cv')
with open(f'{analysis_dir}summary_stats_cv.yaml', 'w') as fh:
    fh.write(f'accuracy: {accuracy}\n'
             f'accuracy_std: {acc_std}\n'
             f'nb_samples: {len(result_df)}\n')
ptc(result_df, analysis_dir)

# test set
result_df = pd.read_csv(f'{classify_dir}prediction_results_test.csv', index_col=None)
accuracy = np.mean(result_df.pred)
acc_std = np.std(result_df.groupby('mod_id').pred.mean())
print(f'overall accuracy test: {accuracy.round(3)}±{acc_std.round(3)}')
analysis_dir = parse_output_path(f'{out_dir}analysis_test')
with open(f'{analysis_dir}summary_stats_test.yaml', 'w') as fh:
    fh.write(f'accuracy: {accuracy}\n'
             f'accuracy_std: {acc_std}\n'
             f'nb_samples: {len(result_df)}\n')
ptc(result_df, analysis_dir, bootstrap=100)
